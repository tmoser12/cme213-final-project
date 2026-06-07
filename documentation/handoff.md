# CUDA Graph Implementation Handoff (Phase 8b)

Summary of findings from the first CUDA graph attempt. Use this when restarting graph work from scratch.

## Goal (Phase 8b)

CUDA-graph the **bundled verify** forward (`verify_gamma` with `leading_bonus`, fixed S = γ+1), so replay can update only device inputs (`input_ids`, `cache_position`, `cur_len`, RoPE tables) between iterations.

---

## What Worked

### 1. Device-side scalars (needed for dynamic `cache_pos`)

Host `int` args like `write_pos` and `cur_len` get **baked into the graph** at capture time. Fix:

- `rope_kv_write_forward_dev(..., write_pos_tensor)` — reads 0-d `int64` CUDA tensor
- `small_q_attn_forward_dev(..., cur_len_tensor)` — same

**Parity:** device-scalar bundled verify matches the eager int path (max diff **0.0**) when run **outside** a graph.

### 2. Graph input staging

Fixed-address inputs are fine:

- `verify_input_ids`, `verify_rope_cos/sin`, `verify_cur_len`, `cache_position`
- Fill these on the host/default stream before `replay()`

Cos/sin, input ids, and scalar values all matched eager before replay.

### 3. Capture (first run inside `torch.cuda.graph()`)

With this recipe, **capture’s inaugural run** can match eager:

- **Default CUDA stream** (side-stream warmup/capture caused **NaN**)
- **One warmup** forward before capture (without it: capture vs eager ≈ **20.9** — graph-pool memory not initialized)
- `torch.cuda.graph_pool_handle()`
- Copy logits into pre-allocated `buffers.logits` (stable output address)

---

## What Failed (The Blocker)

### Graph replay is broken — even with identical inputs

After a capture that matches eager (diff **0.0**):

| Check | Result |
|--------|--------|
| Immediate `replay()` vs capture output | **614** max logit diff |
| Immediate `replay()` vs eager | **614** |
| Second `replay()` | **3978** |
| Replay after reset + prefill + refill | **614** |

This is **not** specific to device scalars — the **int-scalar eager path** replay fails the same way.

So the bug is not “update device scalars before replay.” Replay itself re-executes incorrectly.

### Root cause (diagnosis)

Custom target kernels allocate fresh outputs every forward:

```
torch::empty / at::empty in: embedding, rmsnorm, qkv_proj, attn, o_proj,
residual_add, swiglu, lm_head
```

During capture those allocations come from the **graph memory pool**. Capture’s first run can still be correct, but **replay reuses pool memory without reliably re-running/writing all intermediates** → garbage logits.

Copying only the final logits into `buffers.logits` is **not enough**; **every** intermediate must come from **pre-allocated, fixed-address** buffers (outside or explicitly registered with the graph pool).

### Secondary graph footguns found

| Issue | Symptom |
|--------|---------|
| Side-stream capture/warmup | NaN logits |
| No warmup before capture | Capture vs eager ≈ 20.9 |
| `q.reshape().transpose().contiguous()` in `_run_attention` | Extra allocations even if kernels are fixed |
| SwiGLU internals | Three `linear_no_bias` + act — each allocates unless given scratch buffers |

---

## What Was Implemented (Partial — Safe to Stash/Review)

| Area | Status |
|------|--------|
| `rope_kv_write_forward_dev`, `small_q_attn_forward_dev` | Done, eager parity OK |
| Executor graph scaffolding (`use_cuda_graph`, graph buffers, capture/replay helpers) | Done, **not wired into production `verify_gamma`** (replay unsafe) |
| `test_cuda_graph_verify.py` | Device-scalar tests pass; replay test marked `@expectedFailure` |
| Static workspace plan (`qkv_flat`, `attn_out`, `mlp_up`, `*_forward_out`, split/merge kernels) | Started in shapes/memory only; kernels not finished |

---

## Recommended Restart Plan

1. **Static workspace first** — extend `RuntimeBuffers` with scratch (`qkv_flat`, `attn_out`, `mlp_up`; possibly reuse ping-pong `hidden_a/b`).
2. **Add `*_forward_out` to every op** — no `torch::empty` on the graph path.
3. **Add layout helpers** — `qkv_split_forward`, `merge_attn_heads_forward` to avoid Python `transpose().contiguous()`.
4. **Single static forward** — `_verify_forward_graph_static()` using only buffer views; no dynamic tensor creation in the captured region.
5. **Capture protocol** — default stream, one warmup, `graph_pool_handle()`, logits written directly into `buffers.logits` (no post-forward `copy_` if lm_head writes there).
6. **Tests** — (a) static forward vs eager, (b) capture vs eager, (c) replay vs eager at **multiple** `cache_pos` values, (d) keep existing speculative tests green.

---

## Other Incidental Changes (If Stashing Selectively)

- **`gpu_bind.py`**: lazy `mpi4py` import (for non-MPI unit tests)
- **`test_speculative_target.py`**: allowlist for 8c MPI modules in the “no mpi” scan
- **`runtime/plan.md`**: Phase 8b marked partial with replay blocker documented
- **Phase 8c MPI** (`mpi_coordinator`, etc.): incomplete; separate from graphs

---

## Tests at Time of Handoff

```bash
bash slurm/run_tests_gpu.sh runtime.tests.test_cuda_graph_verify   # device-scalar OK; replay expectedFailure
bash slurm/run_tests_gpu.sh runtime.tests.test_speculative_target  # 9/9 OK
```

---

## Bottom Line

Device-side `write_pos` / `cur_len` is solved and necessary. The remaining work is **making the entire verify forward allocation-free** so CUDAGraph replay has stable addresses for all intermediates — not just attention scalars or final logits.
