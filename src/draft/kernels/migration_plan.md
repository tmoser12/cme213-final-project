# Draft-kernel migration: Qwen2.5-7B dims → Qwen2.5-0.5B

`src/draft/kernels/` holds the inference kernels for the **Qwen2.5-0.5B draft model**.
They were copied verbatim from the 7B suite; this migration specializes them for 0.5B
dims, verifies correctness, and retunes launch configs. These kernels run **only** the
0.5B model, so dims are hardcoded for it (no general dimension guards).

## Dimension delta

| Field | 0.5B | 7B | Notes |
|---|---|---|---|
| hidden_size | 896 | 3584 | |
| intermediate_size | 4864 | 18944 | |
| num_attention_heads | 14 | 28 | |
| num_key_value_heads | 2 | 4 | |
| **head_dim** | **64** | 128 | structural for attention (was hardcoded D=128) |
| GQA ratio h_q/h_kv | 7 | 7 | unchanged; computed dynamically |
| vocab_size | 151936 | 152064 | |
| tie_word_embeddings | true | false | lm_head weight == embed weight (runtime wiring) |

## Build isolation (done)

All five draft extensions were renamed `custom_*_ops`/`qwen_embedding_kernel` →
**`draft_*_ops`** (in each `jit.py`) so the D=64 `.so` never clobbers the 7B D=128 `.so`
in the shared `~/.cache/torch_extensions/py311_cu121/` cache. All `run_benchmark.sh`
module paths and every Python import were repointed `src.target.kernels.*` →
`src.draft.kernels.*`, and cache-clear paths updated to `draft_<dir>_ops`.

## Status

| Kernel | Status | Change summary |
|---|---|---|
| build isolation | ✅ code done | rename `draft_*_ops`, fix `src.draft` paths everywhere |
| embedding | ✅ code done | dynamic on H; benchmark vocab 151936 / hidden 896; comments |
| residual_ops | ✅ code done | dynamic; benchmark hidden 896 / vocab 151936; tie note added |
| swiglu | ✅ code done | dynamic; benchmark + `_correctness_check` 896/4864; config-factory renamed |
| rmsnorm | ✅ code done | **retuned** thread count: round `hidden/8`=112 up to a full warp → 128 threads (112 was not warp-aligned, would break `__shfl_down_sync`'s full mask). benchmark + walkthrough updated |
| attention | ✅ code done | **D 128→64** in prefill + decode launchers; `TORCH_CHECK ==128 → ==64`; all benchmark_scripts dims → 14/2/64; comments |
| GPU verification | ⏳ pending | run each `run_benchmark.sh` on gpu-turing; confirm inline allclose + `draft_*_ops` build |

## Key correctness analysis

- **rmsnorm — the only non-attention risk.** `target_threads = 896/8 = 112`, NOT a warp
  multiple. The old guard (`% 32 == 0`) fell through to 256 threads (full warps, correct
  but 144 idle). Retuned to **round up to the next full warp → 128 threads** (`kernel.cu`
  launch heuristic). Full warps keep `blockReduceSum`'s `__shfl_down_sync(0xffffffff,…)`
  valid; the 16 padding threads contribute a zero partial sum.
- **attention prefill (`flash_attention_kernel`)** is fully templated on `D`. D=64 satisfies
  every constraint (64%16=0 for WMMA → `K_BLOCKS=O_BLOCKS=4`; 64%8=0 for int4 →
  `VECS_PER_ROW=8`). The hand-tuned warp fragment-index math (`s_a`/`s_b`, `my_cols`)
  depends on `KV_BLOCK`/`WMMA_N`, **not** on D — so only the `D` constant changed.
  Shared memory shrinks to ~17 KB. `Q_BLOCK=64`/`KV_BLOCK=32` kept.
- **attention decode (`decode_attn_kernel`)** is fully templated on `D`; only the `D`
  constant + the `head_dim==64` check changed. Shared mem ~12 KB at Q_TOKENS=8.
- **cuBLAS GEMMs** (qkv_proj, o_proj, lm_head, swiglu gate/up/down) read shapes
  dynamically — no code change; new N for qkv = 14·64 + 2·2·64 = 1152 flows through.

## Open tuning items (need GPU profiling — pick from data, not guessing)

1. **decode `NUM_THREADS` at D=64.** Kept at 128. In the PV/acc phase
   (`for d=tid; d<D; d+=NUM_THREADS`) only 64 of 128 threads are active at D=64. Dropping
   to 64 threads fills the acc phase but halves occupancy (decode is already block-starved
   at B=1: only h_q=14 blocks) and doubles iterations in the load/scores phases. Net effect
   unclear — **profile before changing.** Related: `decode_attention_plan.md` split-K +
   GQA-tile-sharing are the real B=1 occupancy fixes.
2. **decode `KV_TILE`** (32) — confirm vs the smaller D=64 footprint.
3. **prefill `Q_BLOCK`/`KV_BLOCK`** — retuning needs a rewrite of the fragment-index math;
   prefill is the secondary path for a draft model, so deferred unless profiling shows it
   matters.

## Verification

On a GPU node (each `benchmark.py` is its own inline `allclose` oracle vs the real HF
module built with a 0.5B `Qwen2Config`):
```bash
bash src/draft/kernels/rmsnorm/run_benchmark.sh        # fast — validates the draft_*_ops rename
bash src/draft/kernels/swiglu/run_benchmark.sh
bash src/draft/kernels/residual_ops/run_benchmark.sh
bash src/draft/kernels/attention/run_benchmark.sh benchmark_decode_attn   # PRIMARY draft path
bash src/draft/kernels/attention/run_benchmark.sh benchmark_rope
bash src/draft/kernels/attention/run_benchmark.sh benchmark_kv_write
bash src/draft/kernels/attention/run_benchmark.sh benchmark_qkv_proj
bash src/draft/kernels/attention/run_benchmark.sh benchmark_o_proj
bash src/draft/kernels/attention/run_benchmark.sh benchmark_fused_attn    # prefill
# embedding has no run_benchmark.sh: srun -m src.draft.kernels.embedding.benchmark (fix local import first if needed)
```
Confirm each prints PASS/✅ on its inline allclose, and that the build loads `draft_*_ops`
(not `custom_*_ops`).
