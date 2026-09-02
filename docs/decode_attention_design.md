# Decode-phase fused attention — design & test plan

**Scope:** Qwen2.5-**7B** (NH=28, NKV=4, D=128). Add a decode-optimized fused
attention path alongside the existing prefill kernel in `kernel.cu`, plus
correctness/benchmark coverage against a **populated** KV cache.

This is a **performance** addition, not a correctness gap: the existing
`flash_attention_kernel` already handles `cur_len > seq_len` (see
`kernel.cu:346`, `q_pos_offset = cur_len - seq_len`), so it is already a correct
— but slow — decode path. The new kernel exists because at `S=1, B=1` the WMMA
kernel launches only ~28 blocks (one per query head) on 72 SMs, wastes ~63/64 of
every Tensor-Core tile on masked rows, and is memory-bound on the KV stream.

## Two regimes, one kernel

| Regime | Query rows `S` | Where | Mask among query rows |
|---|---|---|---|
| **Decode** (`s=1`) | 1 | draft model, target greedy step | none (single row) |
| **Small-q verify** | `γ+1` (≈2–8) | spec-decode target verification | causal triangle |

The compute pattern is identical — load each KV tile once, dot it against the
few query rows, online-softmax-accumulate — so per the design decision this is
**one templated kernel** `decode_attn_kernel<Q_TOKENS, ...>` with `s=1` as the
`Q_TOKENS=1` instantiation. Two thin host launchers give the two named entry
points; both dispatch into the same kernel.

## Semantics (must match HF exactly)

- Query row `r` (0-indexed within the block) sits at **absolute position**
  `cur_len - S + r` in the sequence. Its K/V have already been written into the
  cache at that position (callers run `kv_write_forward` first).
- Row `r` attends to cache positions `j` where `j <= cur_len - S + r` (history is
  fully visible; the `S` new tokens form a causal triangle among themselves).
  For `S=1` this collapses to "attend to all `cur_len` positions".
- This is exactly `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with
  `q_len = S < k_len = cur_len` — PyTorch bottom-right-aligns the causal mask, so
  the existing benchmark's reference stays valid once the cache holds history.
- GQA: `kv_head = head / (h_q / h_kv) = head / 7` (matches `kernel.cu:304`).
- FP16 data, **FP32 accumulation** for scores and the online softmax (no WMMA —
  Tensor Cores are wasted on 1–8 query rows; `half2` scalar math is the right
  tool for a memory-bound GEMV-shaped op).

## Kernel design (Phase 1: single-pass, correctness-first)

`template<int Q_TOKENS, int D, int KV_TILE, int NUM_THREADS> decode_attn_kernel`

- **Grid:** `(1, h_q, B)` — one block per `(batch, query_head)`. (`blockIdx.x`
  reserved as the KV-split axis for the future Phase 3 split-K; it is `1` here.)
- **Block:** `NUM_THREADS = 128` (= D, convenient for the PV phase).
- **Shared memory** (Q_TOKENS=8, KV_TILE=32 worst case ≈ 23 KB, well under the
  48 KB Turing default):
  - `q_sh[Q_TOKENS][D]` (fp16) — query rows, loaded once.
  - `k_sh[KV_TILE][D]`, `v_sh[KV_TILE][D]` (fp16) — current KV tile.
  - `acc[Q_TOKENS][D]` (fp32) — running output accumulator.
  - `scores[Q_TOKENS][KV_TILE]` (fp32), `m[Q_TOKENS]`, `l[Q_TOKENS]` (fp32).
- **Flow** (flash attention, no WMMA):
  1. Cooperatively load `q_sh` (mask rows `>= S`; only matters if a caller pads).
  2. Loop KV tiles over `j0 = 0 .. cur_len` step `KV_TILE`:
     a. Load `k_sh`/`v_sh`, masking rows `>= cur_len`.
     b. **Scores:** map threads to `(row, key)` pairs — each thread computes a
        full `D`-length `half2` dot independently (no cross-thread reduction;
        ≤2 dots/thread at the worst shape). Apply `softmax_scale`, then set
        `-INF` where `j > cur_len - S + r` (causal) or `j >= cur_len` (tail).
     c. Online softmax per row over the tile: new row-max → rescale `l` and
        `acc[r][:]` by `exp(m_old - m_new)`, accumulate tile contribution.
     d. **PV:** map threads to `d` (thread `d` owns output channel `d`);
        `acc[r][d] += Σ_j p[r][j] * v_sh[j][d]`. No reduction.
  3. Normalize `acc[r][:] /= l[r]`, write to `o`.
- `__syncthreads()` separates the score phase (threads→pairs) from the PV phase
  (threads→channels) and guards tile reuse — the two distinct thread mappings
  are the one subtlety; everything else is straight-line.

Readability notes: `Q_TOKENS`, `KV_TILE`, `D`, `NUM_THREADS` are compile-time
template params with a single geometry comment block (mirror the existing
`flash_attention_kernel` header style). No unions / no fragment index magic —
this kernel is deliberately simpler than the WMMA one.

## Host launchers & bindings

Add to `kernel.cu`:

```cpp
// internal: switch(S) over the template instantiations we support
static torch::Tensor launch_decode_attn(q, cache_k, cache_v, cur_len, scale, int S);

torch::Tensor decode_attn_forward (q, cache_k, cache_v, cur_len, scale); // TORCH_CHECK(S == 1)
torch::Tensor small_q_attn_forward(q, cache_k, cache_v, cur_len, scale); // TORCH_CHECK(1 <= S <= MAX_VERIFY)
```

- Signature is **identical** to `fused_attn_forward` — `q:[B,h_q,S,D]`,
  `cache_k/v:[B,h_kv,max_seq,D]`, `cur_len`, `softmax_scale` — so callers and the
  future runtime op-ABI can dispatch on `S` with no shape juggling.
- `launch_decode_attn` does `switch (S) { case 1: ...<1>; case 2: ...<2>; ... }`
  up to `MAX_VERIFY` (start at 8; one line per case, easy to extend).
- Reuse the existing `TORCH_CHECK` block from `fused_attn_forward` (CUDA /
  contiguous / fp16 / dims / GQA divisibility / `cur_len` bounds).
- `bindings.cpp`: add the two forward decls + two `m.def` lines.

## Tests with a **populated** KV cache (the core ask)

The existing `benchmark_fused_attn.py` only ever sets `write_pos=0`,
`cur_len=S` (pure prefill, no history). Decode correctness lives in the
`cur_len > S` regime, so add `benchmark_scripts/benchmark_decode_attn.py`:

- **`make_inputs(B, S, history)`**: allocate `cache_k/v` of `max_seq`, fill
  `[0 : history+S]` with random fp16, set `cur_len = history + S`. `q` is the `S`
  new query rows (they correspond to cache positions `[history : history+S]`).
- **Dual oracle** — assert all three agree at `atol=rtol=1e-2`:
  1. HF: `F.scaled_dot_product_attention(q, repeat_kv(cache_k[:,:,:cur_len]),
     repeat_kv(cache_v[:,:,:cur_len]), is_causal=True, scale)`.
  2. Existing prefill kernel: `custom_ops.fused_attn_forward(q, cache_k, cache_v,
     cur_len, scale)` — independent custom path, catches mask/index bugs HF
     might share.
  3. New kernel: `decode_attn_forward` (S=1) / `small_q_attn_forward` (S>1).
- **Sweep** `S ∈ {1, 2, 4, 5, 8}` × `history ∈ {0, 127, 1024, 4096, 16383}`.
  Non-tile-aligned histories (127, 16383) exercise tail masking and the
  online-softmax combine across tile boundaries — the most bug-prone paths.
- Then time custom vs HF-eager vs `torch.compile`, write
  `benchmark_decode_attn_report.txt` — same structure as
  `benchmark_fused_attn.py` so the harness stays uniform (one file, both named
  ops, dispatched on `S`).

**`run_benchmark.sh`**: add `benchmark_decode_attn` to the header comment's
sub-benchmark list and a `KERNEL_REGEX="decode_attn_kernel"` case in the
`case "$TARGET"` switch (`run_benchmark.sh:72`). Run with
`bash kernel_dev/target/kernels/attention/run_benchmark.sh benchmark_decode_attn`.

## Phasing

1. **Kernel + launchers + bindings** — templated single-pass kernel, two host
   entry points, `bindings.cpp` wired.
2. **Tests** — `benchmark_decode_attn.py` with populated caches + dual oracle +
   `S × history` sweep; `run_benchmark.sh` case. *Gate: parity before any
   optimization.*
3. **Optimization (later, separate change)** — flash-decoding **split-K** over
   the `blockIdx.x` axis + a small reduction kernel (saturate the 72 SMs at
   `B=1`); GQA KV-tile sharing (one block per `(batch, kv_head)` serving its 7
   query heads to cut KV bandwidth). Re-run the Phase 2 oracle to prove the
   optimization is lossless.
4. **Runtime/wrapper dispatch (optional)** — teach `wrapper.py` / the future
   `runtime/` op-dispatch to route small `S` to these ops and large `S` to
   `fused_attn_forward` behind one `attention.run(...)` interface (runtime
   executor). The current `wrapper.py` raises `NotImplementedError` on
   `past_key_value`; this is where that gets filled in.

## Risks / watch-items

- **Two thread mappings** (scores→pairs, PV→channels) inside one kernel — the
  `__syncthreads()` between them and the tile-reuse barrier are the correctness
  pivots. Covered by the non-tile-aligned `history` cases.
- **`MAX_VERIFY`** caps the `switch`; pick `8` (γ≤7) and `TORCH_CHECK` it so an
  out-of-range `S` fails loudly instead of silently dispatching to the prefill
  kernel.
- **`history` near `max_seq`** — keep the `cur_len <= cache_k.size(2)` check;
  test `history = max_seq - S`.
- Split-K is explicitly **out of Phase 1** — do not let the scratch-buffer /
  reduction complexity leak into the first correct version.
