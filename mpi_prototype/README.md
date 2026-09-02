# mpi_prototype — Mock MPI Speculative Decoding Prototype

Minimal **2-rank mpi4py** setup that mimics the speculative-decoding control loop without loading any models.

> ✅ **Ported (2026-06-07).** Phase 8c now lives in `runtime/speculative/mpi_protocol.py` +
> `mpi_coordinator.py` (real 7B target + 0.5B draft, launched by `slurm/run_speculative.sh`). This
> directory remains as a lightweight, model-free MPI smoke test and the design reference below.

**Related docs:** `docs/speculative_decoding.md`, `docs/mpi_benchmarks.md`, Phase 8a golden path in `runtime/speculative/target_step.py`.

---

## Purpose

| This prototype | Production (Phase 8c) |
|----------------|----------------------|
| `MockDraftModel` / `MockTargetModel` (deterministic fake logits) | `Qwen2Executor(kernel_set="draft")` + `Qwen2Executor(kernel_set="target")` |
| NumPy sampler in `mpi_prototype/sampler.py` | `runtime/speculative/sampler.py` (torch, CPU) |
| `mpi_prototype/protocol.py` wire helpers | Move to `runtime/speculative/mpi_protocol.py` or into `mpi_coordinator.py` |
| `mpi_prototype/main.py` rank loops | `runtime/speculative/mpi_coordinator.py` |

Phase 8a already implements target-side verify + host sampling **without MPI** (`target_speculative_step`). The MPI layer only replaces the in-process draft payload with messages between ranks.

---

## Directory layout

```
mpi_prototype/
├── README.md           # this file
├── main.py             # mpirun entry: rank 0 = target, rank 1 = draft
├── protocol.py         # wire format + Send/Recv helpers  ← port first
├── mock_models.py      # fake M_q / M_p (delete after integration)
├── sampler.py          # NumPy-only sampler (test-only; runtime uses torch)
├── gpu_check.py        # per-rank cuda:r binding + UUID distinctness check
├── run_mpi.sh          # local + SLURM launchers  ← template for slurm/run_speculative.sh
└── tests/
    └── test_mock_step.py   # protocol roundtrip + in-process sync tests
```

---

## Rank roles and one iteration

```
Rank 0 (TARGET_RANK, GPU 0 in production):
  1. Recv draft payload from rank 1
  2. [prod] take_pending_bonus() → verify_gamma(γ, leading_bonus=?) — single forward
  3. Host accept/reject + bonus sample (q from MPI, p from local D2H)
  4. [prod] rollback_cache + defer_bonus_token(bonus) — **MPI immediately**
  5. Send TargetResult to rank 1
  6. After final iteration: flush_pending_bonus()

Rank 1 (DRAFT_RANK, GPU 1 in production):
  1. [mock] draft_gamma(γ)  →  [prod] γ sequential decode_step on draft executor
  2. Send DraftPayload to rank 0
  3. Recv TargetResult
  4. Sync prefix: append draft_ids[:n_accepted] + bonus_token; draft decode on bonus
  5. Generate next γ drafts (overlaps with target idle until next payload)
```

`prefix_len` in `TargetResult` is the **target KV cursor at the start of the iteration** (before verify). It does **not** include a deferred bonus waiting to be bundled into the next verify. Draft sync rolls back to `prefix_len`, then appends `draft_ids[:n_accepted] + bonus_token` (draft applies bonus immediately for its γ loop).

---

## Wire protocol (`protocol.py`)

### Draft → target

| Field | Type | Shape |
|-------|------|-------|
| `draft_token_ids` | `int32` | `[γ]` |
| `draft_logits` | `float16` | `[γ+1, vocab]` |

MPI tags: `TAG_DRAFT_PAYLOAD` (ids), `TAG_DRAFT_PAYLOAD + 1` (logits).

Use `send_draft_payload()` / `recv_draft_payload()` — do not call `comm.Send` on float16 arrays directly (see quirks below).

### Target → draft

| Field | Type |
|-------|------|
| `n_accepted` | `int32` |
| `bonus_token` | `int32` |
| `prefix_len` | `int32` | Target KV cursor at iter start |
| `cache_pos_after` | `int32` | Target KV cursor after rollback (excludes new deferred bonus) |

Packed as one `int32[3]` vector. Tag: `TAG_TARGET_RESULT`.

**Target never sends `p` logits over MPI.** Only draft `q` crosses to target; target `p` stays local (D2H on target rank only).

---

## Cluster setup (Stanford HPCC)

### MPI stack

Same as **cme213-hw7** (`~/cme213-hw7`):

- Module: `course/cme213/nvhpc/24.1` (NVHPC Open MPI 4.1.x)
- C++ homework pattern: `mpic++ …` then `mpirun ./binary`
- Python: `mpi4py>=4.0.0` in `requirements.txt`, installed in conda env `cme213`

Always load NVHPC before running:

```bash
module load course/cme213/nvhpc/24.1 gnu12/12.3.0
```

### SLURM launch pattern

**Do:** one SLURM task, `mpirun` spawns MPI ranks inside it (hw7 style):

```bash
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
  bash -lc 'module load course/cme213/nvhpc/24.1; mpirun --oversubscribe -np 2 python -m mpi_prototype.main'
```

**Don't:** `--ntasks=2` with nested `mpirun -np 2` (double-launch).  
**Don't:** `srun --ntasks=2 python …` without `mpirun` unless `mpi4py` is rebuilt against the module `libmpi` and `LD_LIBRARY_PATH` is set — the prebuilt wheel expects NVHPC libs via the module.

GPU rehearsal (2 GPUs, 1 SLURM task):

```bash
srun --partition=gpu-turing --gres=gpu:2 --ntasks=1 --cpus-per-task=4 \
  mpirun --oversubscribe -np 2 python -m mpi_prototype.main
```

In production, bind rank 0 → `cuda:0`, rank 1 → `cuda:1` inside `mpi_coordinator.py`.

### Environment

| Variable / flag | Why |
|-----------------|-----|
| `PYTHONNOUSERSITE=1` | Avoid user-site torch shadowing conda env |
| `module load course/cme213/nvhpc/24.1` | Provides `mpirun` + `libmpi` for mpi4py |
| Direct python path on compute nodes | `~/.conda/envs/cme213/bin/python` — `conda run` can drop module `LD_LIBRARY_PATH` |

---

## Known quirks (save time when porting)

1. **float16 `Send`/`Recv` segfaults** on NVHPC Open MPI with mpi4py. Send logits as `logits.view(np.uint8)` and reshape on recv. Implemented in `send_draft_payload` / `recv_draft_payload`. Keep this in runtime.

2. **torch + mpirun on login node** can segfault even without CUDA. Production target rank will import torch (GPU); that is fine on compute nodes with GPUs. Mock uses NumPy-only `mpi_prototype/sampler.py` for login-node testing.

3. **`comm.send`/`recv` (pickle)** works for float16 but is slow at vocab≈152k. Use buffered `Send`/`Recv` with uint8 view for production payloads.

4. **Golden path for debugging:** single-process `target_speculative_step()` in `runtime/speculative/target_step.py` — keep tests passing; MPI coordinator should call the same logic on rank 0.

---

## How to run

```bash
# Unit tests (no MPI, login node)
PYTHONNOUSERSITE=1 python -m unittest mpi_prototype.tests.test_mock_step -v

# Local 2-rank (login node; NVHPC module loaded)
bash mpi_prototype/run_mpi.sh
bash mpi_prototype/run_mpi.sh --steps 20 --gamma 3

# SLURM CPU (validates cluster launch)
bash mpi_prototype/run_mpi.sh --slurm-cpu

# SLURM GPU partition (2 GPUs allocated, still mock CPU logic)
bash mpi_prototype/run_mpi.sh --slurm-gpu
```

Success output ends with rank 0 printing `OK: … final prefix len=…` after draft and target prefixes match.

### GPU verification (`--require-gpu`)

`bash mpi_prototype/run_mpi.sh --slurm-gpu` automatically passes `--require-gpu`. Each MPI rank:

1. Calls `torch.cuda.set_device(mpi_rank)` → rank 0 = `cuda:0`, rank 1 = `cuda:1`
2. Runs a small fp32 kernel on that device (unique `probe` value per rank)
3. `MPI.allgather` of device **UUID** (via `nvidia-smi`) — fails if both ranks share the same physical GPU

Manual:

```bash
bash mpi_prototype/run_mpi.sh --slurm-gpu --steps 3 --gamma 2
# or: mpirun -np 2 python -m mpi_prototype.main --require-gpu ...
```

Expected output (rank 0):

```
GPU binding OK — one physical device per rank:
  rank 0 (target): cuda:0 | Quadro RTX 6000 | id=GPU-5878… | mem=22.1GB | probe=2.0
  rank 1 (draft):  cuda:1 | Quadro RTX 6000 | id=GPU-ff5a… | mem=22.1GB | probe=5.0
OK: … (2-GPU verified)
```

CPU-only runs (`--slurm-cpu`, local login node) omit `--require-gpu` so they stay lightweight.

---

## Port checklist → `runtime/speculative/`

### 1. `runtime/speculative/mpi_protocol.py` (from `mpi_prototype/protocol.py`)

- Copy `DraftPayload`, `TargetResult`, rank constants, tags
- Copy `send_draft_payload`, `recv_draft_payload`, `pack_target_result`, `unpack_target_result`
- Optionally rename to match runtime naming (`SpeculativeStepResult` already exists in `types.py` — map fields or extend)

### 2. `runtime/speculative/mpi_coordinator.py` (from `mpi_prototype/main.py`)

```python
# Pseudocode for production loop on each rank:
if rank == TARGET_RANK:
    device = torch.device("cuda:0")
    # load target executor, prefill prompt once
    while not done:
        payload = recv_draft_payload(comm, source=DRAFT_RANK, gamma=γ, vocab=vocab)
        result = target_speculative_step(executor, payload.draft_token_ids, payload.draft_logits, rng)
        comm.Send(pack_target_result(...), dest=DRAFT_RANK, tag=TAG_TARGET_RESULT)

elif rank == DRAFT_RANK:
    device = torch.device("cuda:1")
    # load draft executor, prefill same prompt
    while not done:
        draft_ids, draft_logits = run_draft_gamma(draft_executor, γ)  # Phase 8d
        send_draft_payload(comm, DraftPayload(...), dest=TARGET_RANK)
        result = recv TargetResult; sync_draft_prefix(...)
```

Replace `MockTargetModel.verify_and_accept` with `target_speculative_step`. Replace `MockDraftModel.draft_gamma` with draft executor loop (Phase 8d).

### 3. `slurm/run_speculative.sh` (from `mpi_prototype/run_mpi.sh`)

- `--gres=gpu:2`, `--ntasks=1`, `mpirun --oversubscribe -np 2`
- `module load course/cme213/nvhpc/24.1 gnu12/12.3.0`
- `PYTHONNOUSERSITE=1`, project `setup.sh` or direct cme213 python
- Entry: `python -m runtime.speculative.mpi_coordinator` (or similar)

### 4. Tests to add in `runtime/tests/`

- Keep `TestNoMpiInPhase8a` — only `mpi_coordinator.py` (and slurm script) should import mpi4py
- GPU integration test: 2-rank job with mock draft logits first, then real draft model when Phase 8d lands
- Assert `prefix_len` sync semantics (copy from `test_mock_step.py`)

### 5. Do not port

- `mock_models.py` — discard after real executors wired
- `mpi_prototype/sampler.py` — runtime already has `runtime/speculative/sampler.py`

---

## Mapping to Phase 8 plan

| Phase | Status | mpi_prototype coverage |
|-------|--------|-------------------|
| 8a target verify + host sampler | done | logic mirrored; prod uses torch sampler |
| 8c MPI coordinator | pending | **this directory** |
| 8b CUDA graphs | pending | after 8c; never graph MPI/sampler |
| 8d draft kernels | pending | replace `MockDraftModel.draft_gamma` |

When 8c is done, this directory can remain as a lightweight MPI smoke test, or tests can move into `runtime/tests/test_speculative_mpi.py`.
