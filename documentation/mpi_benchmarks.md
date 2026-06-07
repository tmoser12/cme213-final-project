# MPI Transfer Benchmarks (2-GPU Speculative Decoding)

Quick benchmark data for MPI send/recv between two GPUs on the Stanford HPCC `gpu-turing` partition.
The wire payloads here are identical to the production protocol now in
`runtime/speculative/mpi_protocol.py` (used by `runtime/speculative/mpi_coordinator.py`); this doc's
numbers come from the `test_mpi/` micro-benchmark of the same payloads.

**Implementation:** `test_mpi/benchmark.py`  
**Run:** `bash test_mpi/run_mpi.sh --slurm-gpu --benchmark`

---

## Wire payloads (production semantics)

| Direction | Fields | Types | Size (γ=4, vocab=152064) |
|-----------|--------|-------|--------------------------|
| **Draft → target** | `draft_token_ids`, `draft_logits` | `int32[γ]`, `float16[γ+1, vocab]` | **1,520,656 B (~1.45 MiB)** |
| **Target → draft** | `n_accepted`, `bonus_token`, `prefix_len`, `cache_pos_after` | `int32[4]` | **16 B** |

Notes:

- Default **γ = 4**, **vocab = 152064** (Qwen2.5-7B from `runtime/core/configs/qwen2.5-7b.yaml`).
- Draft logits are sent as a `uint8` view of the fp16 buffer — direct `float16` `Send`/`Recv` segfaults on NVHPC Open MPI (see `test_mpi/README.md`).
- Target **never** MPIs its `p` logits to draft. Only draft `q` crosses to target; target `p` stays local (D2H on target rank only for host sampling).

---

## Benchmark methodology

- **2 MPI ranks**, 1 SLURM task, 2 GPUs: rank 0 (target) → `cuda:0`, rank 1 (draft) → `cuda:1`.
- **Draft rank:** allocate fp16 logits and int32 token ids on GPU → **D2H** → blocking MPI send via `send_draft_payload()`.
- **Target rank:** blocking MPI recv via `recv_draft_payload()` → optional **H2D** of logits to `cuda:0` (enabled by default).
- **Target → draft:** `send_target_result()` / `recv_target_result()` for the 16-byte sync metadata.
- Default: 5 warmup iterations, 30 timed trials. Timings use `MPI.Wtime()`.

---

## Results (hpcc-gpu-5-2, 2026-06-06)

Hardware: 2× Quadro RTX 6000 (Turing), NVHPC Open MPI 4.1.x, γ=4, vocab=152064, 30 trials.

### Draft → target (dominant payload, ~1.45 MiB)

| Stage | Mean | p50 | Min | Max | Effective BW |
|-------|------|-----|-----|-----|--------------|
| Draft D2H (ids + logits) | 0.39 ms | 0.30 ms | 0.27 ms | 1.05 ms | 3.9 GB/s |
| Draft MPI send (blocking) | 0.19 ms | 0.18 ms | 0.16 ms | 0.23 ms | 8.0 GB/s |
| Target MPI recv | 0.68 ms | 0.61 ms | 0.54 ms | 1.36 ms | 2.2 GB/s |
| Target H2D (logits) | 0.24 ms | 0.23 ms | 0.22 ms | 0.26 ms | 6.5 GB/s |
| **Draft-side total (D2H + send)** | **0.58 ms** | 0.49 ms | 0.44 ms | 1.26 ms | 2.6 GB/s |
| **Target-side total (recv + H2D)** | **0.92 ms** | 0.84 ms | 0.76 ms | 1.61 ms | 1.7 GB/s |

### Target → draft (sync metadata, 16 B)

| Stage | Mean | p50 |
|-------|------|-----|
| Target MPI send | 0.002 ms | 0.002 ms |
| Draft MPI recv | 0.006 ms | 0.005 ms |

### Full speculative iteration (approx)

Sum of draft-side total + target-side total + both tiny sync messages:

| Metric | Value |
|--------|-------|
| **Total bytes** | 1,520,672 B (~1.45 MiB) |
| **Mean latency** | **1.51 ms** |
| **p50 latency** | 1.35 ms |
| **Effective BW** | ~1.0 GB/s (end-to-end) |

---

## Interpretation

- The **draft → target logits transfer** dominates: ~1.45 MiB per speculative step vs 16 B for the sync reply.
- At ~1.5 ms per iteration for copies + MPI alone, communication overhead is on the order of **650+ iterations/sec** if GPU compute were free. In practice, target/draft forward passes will be the bottleneck; this sets a baseline for the MPI + D2H/H2D budget.
- Draft MPI send (~0.19 ms) is faster than target MPI recv (~0.68 ms) because blocking send/recv times are measured on different ranks and include synchronization effects; both are sub-millisecond for this payload size on same-node GPUs.
- Target → draft sync is negligible (~microseconds).

---

## How to reproduce

```bash
# Default (30 trials, γ=4, qwen2.5-7b vocab)
bash test_mpi/run_mpi.sh --slurm-gpu --benchmark

# Custom trial count
bash test_mpi/run_mpi.sh --slurm-gpu --benchmark --trials 50 --warmup 5

# Skip target H2D after recv (MPI-only on target side)
bash test_mpi/run_mpi.sh --slurm-gpu --benchmark --no-h2d-after-recv
```

Manual (inside a 2-GPU allocation):

```bash
module load course/cme213/nvhpc/24.1 gnu12/12.3.0
cd /home/cme213/tobiascm/cme213-final-project
env PYTHONNOUSERSITE=1 mpirun --oversubscribe -np 2 python -m test_mpi.benchmark
```

---

## Related files

| File | Role |
|------|------|
| `test_mpi/benchmark.py` | Benchmark entry point |
| `test_mpi/protocol.py` | Wire format + `send_*` / `recv_*` helpers |
| `test_mpi/run_mpi.sh` | SLURM launcher (`--benchmark` flag) |
| `test_mpi/README.md` | Full MPI prototype docs + port checklist |
| `runtime/plan.md` | Phase 8c production MPI coordinator plan |
