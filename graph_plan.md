# CUDA Graph Implementation Plan

Living planning doc for making the `runtime/` inference engine launch-bound → graph-bound.
Update the checkboxes and notes as work lands. Companion to:
- `documentation/cuda_graphs_explained.md` — concepts / mental model (read first)
- `documentation/cuda_graph_issues_and_concepts.md` — journal of concrete issues hit, each explained
- `runtime/plan.md` — overall engine plan
- `documentation/handoff.md` — previous attempt (advisory only; that code was reset away)

---

## Problem statement

Each full forward runs ~30 layers × ~5 ops ≈ **150+ kernel launches from Python**, each paying
~10–50 µs of pybind/interpreter overhead. At S=1 decode the GPU is mostly idle waiting on the
host. Goal: **capture the per-token forward into a CUDA graph and replay it as one launch**, so
the decode loop (and later the speculative `verify_gamma`) is bound by GPU work, not host launches.

---

## Key findings from repo audit (2026-06-06)

These reframe the work and **shrink it substantially** vs. the previous handoff.

1. **Kernels are already graph-pool safe.** Every op (`rmsnorm`, `embedding`, `attention`,
   `swiglu`, `residual_ops`) allocates outputs *only* via `torch::empty` / `at::empty` /
   `torch::empty_like` — the PyTorch caching allocator. Grep found **no raw `cudaMalloc`, no
   `cudaMemcpy`, no `.item()`, no `cuda*Synchronize`** in any `.cu`/`.cpp`.
   → During `torch.cuda.graph(...)` capture, those allocations come from the graph's private
     pool and get **deterministic, stable addresses on replay**. This is the supported PyTorch
     pattern (gpt-fast, vLLM). **We do NOT need allocation-free `_out` variants for every op.**
   → The handoff's stated root cause ("torch::empty breaks replay") is most likely **wrong**.
     Phase 0 validates this empirically before any kernel surgery.

2. **The real blocker is host-scalar position args.** `rope_kv_write_forward(..., int64_t
   write_pos)` and `decode_attn_forward(..., int64_t cur_len)` / `small_q_attn_forward(...)`
   take **host ints**, used as `static_cast<int>(...)` kernel launch params. These get **baked
   into the graph at capture time** → every replay writes/attends at the captured position →
   frozen, and produces growing garbage as the real sequence advances. Fix: device-scalar
   variants that read the position from a 0-d `int64` CUDA tensor.

3. **RoPE cos/sin are host-sliced by position.** `buffers.rope_embeddings(write_pos, S)` does
   `self.rope_cos[start:start+length]` — a host slice whose `data_ptr` is baked at capture.
   Fix: a small static `static_cos`/`static_sin` buffer refreshed per step *outside* the graph
   (one cheap copy), OR pass the full table + device index into the kernel. Start with the
   static-refresh approach (no kernel change for rope indexing).

4. **Inputs are freshly allocated each call.** `verify_gamma` does `torch.cat([bonus, drafts])`;
   `decode_step` does `token_id.unsqueeze(1)`. For graphs, the token id(s) must be written into
   a **static input buffer** that the captured embedding reads.

5. **Foundations already exist.** `buffers.cache_position` is a device 0-d int64 scalar, and
   `executor._advance_cache_pos` already maintains it via `fill_`/`add_` (no `.item()` in loop).
   Static logits buffer (`buffers.logits`) exists. KV cache and rope tables are persistent.

---

## Strategy

- **First graph target: DECODE (S=1).** Fixed shapes; only the token id and position change per
  step. Canonical CUDA-graph case, lowest complexity, directly speeds the autoregressive loop.
- **Second target: VERIFY (S=γ+1 fixed).** The speculative-decode hot path. Same machinery plus
  the `leading_bonus_valid` masking from `runtime/plan.md` §"Single CUDA VERIFY graph".
- **Prefill stays eager.** Variable length, runs once per prompt — low ROI, high complexity.
- **Capture whole forward** (embedding → layers → final norm → lm_head) per shape, relying on
  the graph memory pool for intermediates. **Additive, behind `use_cuda_graph` flag** — eager
  path stays the golden reference for parity at all times.
- **Never graph:** host sampling, MPI, `rollback_cache`, prefill.

---

## Conventions (from `.cursor/skills/gpu-cluster`)

- All GPU work via SLURM, never the login node. `source setup.sh` first (conda `cme213`,
  `gnu12`, model-path env). Target **sm_75 / FP16** (Turing, no BF16/TF32). 30-min job cap →
  run targeted test modules, load the model once per job.
- Run Python: `bash slurm/run_python.sh <script|-m module> [args]`.
- Run tests: `bash slurm/run_tests_gpu.sh runtime.tests.<module>`.
- After editing `.cu`/`bindings.cpp`: rebuild (`bash scripts/build_kernels.sh attention`) and
  clear JIT cache if needed (`rm -rf ~/.cache/torch_extensions/...`).
- Per `.cursor/skills/progress-logging`: append to `project_logs/YYYY-MM-DD.md` after each
  substantive step.
- CPU-only structure tests can run on the login node; anything touching CUDA goes through srun.

---

## Phases & TODOs

Status legend: ⬜ not started · 🟡 in progress · ✅ done · ❌ blocked

### Phase 0 — Baseline + validate the allocation hypothesis (cheap, decisive) ✅ DONE
Goal: prove the direction before any kernel changes. If graph-pool replay works for an
allocating forward, the whole "rewrite every op" path is unnecessary.

- ✅ 0.1 `runtime/benchmarks/phase0_graph_probe.py` — eager decode baseline:
  **33.8 tokens/sec (29.6 ms/token)** for 7B, batch=1 on Quadro RTX 6000. This is the number to beat.
- ✅ 0.2 Hypothesis probe + `runtime/benchmarks/phase0_graph_diag.py` (progressive capture: op →
  layer → stack → full). **Result: capture produces NaN/garbage — but NOT because of allocations.**
  rmsnorm alone "replayed" only because its graph was *empty*; a decoder layer's capture already
  diverges and **grows across replays**. Root cause found (see 0.3).
- ✅ 0.3 **CONCLUSION (see Findings log 2026-06-06):** the real blocker is that **every custom
  kernel launches on the default stream (stream 0), not the current stream**, so `torch.cuda.graph`
  never records them ("CUDA Graph is empty" warning). The allocation theory is dead (rmsnorm proves
  the caching allocator path is fine). New top priority: **Phase 1 — stream correctness**.

### Phase 1 — Kernel stream correctness (ROOT CAUSE) ✅ DONE (2026-06-07)
Goal: make capture actually record our kernels. Prerequisite for *any* graph work; also harmless
to eager mode (current stream == default stream when no graph/side-stream is active).
See `documentation/cuda_graph_issues_and_concepts.md` Issue #1 for the full concept writeup.

- ✅ 1.1 Added `, 0, at::cuda::getCurrentCUDAStream()` to all 7 launch sites: `embedding/kernel.cu:68`,
  `rmsnorm/kernel.cu:157`, `residual_ops/kernel.cu:93`, `attention/kernel.cu:201,594,830`,
  `swiglu/kernel.cu:137`. Added `#include <ATen/cuda/CUDAContext.h>` to embedding + rmsnorm (others
  already had it). cuBLAS ops (qkv/o_proj/lm_head/swiglu GEMMs) already use the current stream.
- ✅ 1.2 Rebuilt all exts (`bash scripts/build_kernels.sh all`).
- ✅ 1.3 Re-ran `phase0_graph_diag.py`: **no "empty graph" warning**; every stage (op → layer →
  stack → full) shows `replay == eager` exactly. Allocation hypothesis confirmed.
- ✅ 1.4 Re-ran `phase0_graph_probe.py` fixed-position probe → **PASS**, replay vs eager max|Δ| =
  **0.0000** ×3. (Pre-replay "capture vs eager" = 9.87 is the expected "capture records, doesn't
  execute" artifact — see Issue #1 follow-up.)
- ✅ 1.5 Eager parity unchanged: `runtime.tests.test_executor.TestExecutorGpu` — **4/4 OK** (prefill
  logits, decode step, greedy trajectory, HF argmax match). Stream fix is eager-safe.

### Phase 2 — Device-scalar position kernels (un-freeze the graph) ✅ DONE (2026-06-07)
Goal: positions live on device so the graph isn't frozen at the capture-time position. Additive —
host-int variants untouched for prefill/eager parity. Concept writeup: `cuda_graph_issues_and_concepts.md`
Concept #2. Implementation: one launcher/dispatch serves both paths via an optional `const int64_t*`
override (null → host int; non-null → read device scalar inside the kernel).

- ✅ 2.1 `rope_kv_write_forward_dev(..., Tensor write_pos)` — 0-d int64 CUDA scatter offset.
- ✅ 2.2 `decode_attn_forward_dev` + `small_q_attn_forward_dev(..., Tensor cur_len)` — device-scalar
  cur_len for loop bounds + causal mask + q_pos. (Grid independent of cur_len ✓; S stays a host
  shape selecting the template/grid — only the position varies.)
- ✅ 2.3 Exposed `_dev` ops in `attention/ops.py` + package `__init__.py`; bindings in `bindings.cpp`.
- ✅ 2.4 Rebuilt attention ext (on a **compute node** — login-node compile OOM-kills `cicc`).
- ✅ 2.5 **Parity (no graph):** `test_attention_dev_scalar` — 5/5, dev == host **bit-exact**
  (`torch.equal`) across positions for decode/small_q/rope_kv_write (±RoPE) + bad-scalar rejection.
  Regression: `test_attention` 8/8 OK (host-int refactor clean).

### Phase 3 — Static buffers + allocation-stable decode forward ✅ DONE (2026-06-07)
Goal: a decode forward whose only varying inputs are static device buffers. Concept: `cuda_graph_issues_and_concepts.md`
Concept #3 (prepare/forward split = the capture/replay boundary).

- ✅ 3.1 `RuntimeBuffers` gained `static_input_ids` ([batch,1] int64), `static_cur_len` (0-d int64),
  `static_cos`/`static_sin` ([batch,1,head_dim]). Kept OUT of `plan_memory`'s seq-scaling budget
  (sub-KB fixed scratch; folding in would break the linear-scaling test) — documented inline.
- ✅ 3.2 `RuntimeBuffers.refresh_decode_rope(pos)` — copies the rope row for `pos` into
  `static_cos/sin` in place.
- ✅ 3.3 Executor: `static_decode` branch in `_run_attention` (uses `_dev` ops + `cache_position`
  / `static_cur_len` device scalars + static cos/sin), threaded via `_run_decoder_layer` /
  `_forward_stack`. New `_prepare_decode_static` (per-replay update, outside graph),
  `_decode_forward_static` (the capturable region), `decode_step_static` (eager driver).
- ✅ 3.4 **Parity (no graph):** `test_decode_static` — `decode_step_static` == eager `decode_step`
  **bit-exact** (`torch.equal`) over a 6-step greedy trajectory. 2/2 OK.
  - ⚠️ Test-isolation note: `test_decode_static` holds a 7B resident (`setUpClass`); do NOT co-run it
    with `test_buffers` GPU tests (which load a *second* 7B) in one job → OOM. Run separately.

> **Future optimization (post-MVP) — fold the RoPE slice+copy into the graph.** Right now the
> per-step `static_cos/sin` refresh (3.2) is a host-side slice + `copy_` that runs *outside* the
> captured region — a tiny launch we re-issue every token. Eventually, pass the **full** rope table
> (`rope_cos/rope_sin`, already resident, `[max_seq, head_dim]`) into the `_dev` attention +
> rope-kv-write kernels and have them index it by the **device position scalar** (`cache_position`),
> so the RoPE lookup happens *inside* the graph. That removes the only remaining per-step host op on
> the decode path. Memory is a non-issue — the full table is already allocated; we'd just stop
> slicing it on the host. Requires extending the `_dev` kernels to take the table + an int row
> offset (device scalar) instead of a pre-sliced `[B, S, D]` cos/sin. Tracked in
> "Future optimizations" below.

### Phase 4 — Capture & replay the decode graph ✅ DONE (machinery) (2026-06-07)
Goal: one replay = one token. Concept: `cuda_graph_issues_and_concepts.md` Concept #4.

- ✅ 4.1 Executor graph state (`use_cuda_graph`, `_decode_graph`, `_decode_graph_logits`,
  `_graph_pool`), captured lazily.
- ✅ 4.2 `_capture_decode_graph`: side-stream warmup (3×) → `torch.cuda.graph(g, pool=pool)` over
  `_decode_forward_static`; stashes the live output handle.
- ✅ 4.3 `decode_step_graph(token_id)`: `_prepare_decode_static` (update in place) → lazy capture →
  `replay()` → advance → return live logits. No host sync, no `.item()`.
- ✅ 4.4 Lifecycle: one graph reused across prompts (positions are device scalars; KV/buffer
  addresses are stable, so reset+prefill needs no re-capture). Validated by `test_decode_graph`.

> 🔴 **KEY FINDING (Concept #4): 7B batch-1 decode is MEMORY-BANDWIDTH-BOUND, not launch-bound —
> graph speedup is 1.00×.** Graphed decode is bit-exact and *does* cut host work (CPU dispatch
> 28.1→20.6 ms/tok), but both are below the **29.7 ms/tok GPU floor** (≈15 GB weights / 672 GB/s ≈
> 22 ms hard floor), so wall-clock is unchanged. The original "super launch-bound" premise doesn't
> hold for 7B single-stream decode. The machinery is correct and its payoff is on the **0.5B draft**
> (Phase 8d, ~14× less weight traffic → genuinely launch-bound), larger batch, or quantized weights.
> A 7B VERIFY graph would also be ~1.00× (γ+1 rows share one weight read → even more compute-dense).

### Phase 5 — Validate graphed decode (correctness + perf) ✅ DONE (2026-06-07)
- ✅ 5.1 / 5.2 Parity: `test_decode_graph` — graphed decode == eager **bit-exact** over an 8-step
  trajectory; graph reused across prefills (no NaN). (HF parity already covered transitively via
  eager `test_executor`; static path == eager proven in Phase 3.)
- ✅ 5.3 Re-capture robustness: `test_graph_reused_across_prefills` (prefill→graph→reset→prefill→graph).
- ✅ 5.4 **Perf:** `runtime/benchmarks/phase4_graph_decode.py` — eager 33.6 vs graph 33.6 tok/s →
  **1.00× (memory-bound; see finding above).**
- ✅ 5.5 Wired `use_cuda_graph` through `greedy_extend` (defaults to the executor flag; picks
  `decode_step_graph` vs `decode_step`). The decode graph is now usable from the main generate loop.

### Phase 6 — VERIFY graph for the target ✅ DONE (2026-06-07)
Goal: a graph-capturable `verify_gamma`, available as an option on the 7B target.

> 🔵 **Perf note:** a 7B VERIFY graph is ~1.00× (even more compute-dense than decode — γ+1 query
> rows share one weight read). Implemented for **correctness/availability** and as the template the
> **0.5B draft** executor will reuse (where it actually pays off). Simpler than the masked
> single-graph design in `runtime/plan.md`: one graph **per query length S** (= γ or γ+1), no
> `leading_bonus_valid` masking needed — the bonus is just a real leading query token.

- ✅ 6.1 Per-S verify static buffers (lazy, contiguous `[batch,S]` ids + `[batch,S,head_dim]`
  cos/sin + `cur_len` scalar), keyed by S in `_verify_state`.
- ✅ 6.2 Unified static attention path: `_run_attention(static_attn=True)` reads `_static_ctx`
  and picks `decode_attn_forward_dev` (S=1) or `small_q_attn_forward_dev` (S>1).
  `_verify_forward_static` / `_prepare_verify_static` / `_capture_verify_graph`.
- ✅ 6.3 `verify_gamma_graph(draft_ids, leading_bonus=…)` — lazy capture per S, shared graph pool
  with the decode graph; one graph per distinct S reused across calls.
- ✅ 6.4 Parity: `test_verify_graph` — graphed == eager `verify_gamma` **bit-exact** for no-bonus
  (S=γ) and leading-bonus (S=γ+1); per-γ graphs cached. 3/3 OK. Decode/dev-scalar regressions green.
- ✅ 6.5 Perf: ~1.00× as expected (see benchmark note); not separately benchmarked (compute-bound).

### Phase 7 — Integration & docs ✅ (target side)
- ✅ 7.1 S=1 flush is covered by the existing decode graph (`decode_step_graph`); no separate graph.
- ✅ 7.2 Benchmark + finding written to `documentation/target_graph_benchmarks.md`;
  `runtime/plan.md` / `README` refresh tracked for the draft phase.
- ✅ 7.3 Speedups summarized: decode 1.00×, verify ~1.00× (memory/compute-bound 7B). See benchmark doc.

---

## ✅ Target-model graphs: COMPLETE (2026-06-07)

The 7B target can run decode and verify under CUDA graphs, behind `use_cuda_graph` /
`decode_step_graph` / `verify_gamma_graph` / `greedy_extend(use_cuda_graph=True)`. All bit-exact
vs eager. Speedup is ~1.00× because 7B single-stream decode is memory-bandwidth-bound (Concept #4),
so this is an **available option**, not a speedup, for the target.

**Next (separate effort): draft-model graphs.** The 0.5B draft is genuinely launch-bound (~14× less
weight traffic) — applying this exact pipeline there should give a real speedup. Draft kernels are
being ported in (`draft_model_files/`, synced from `origin/eli_dev`). Steps when ready:
1. Build draft kernels (`head_dim=64`), give them the same stream-correct launches + `_dev` device-
   scalar position variants (Phases 1–2 applied to `production_kernels/draft/`).
2. Draft `Qwen2Executor(kernel_set="draft")` mirroring the target API, including the static decode
   path + `decode_step_graph` / `verify_gamma_graph`.
3. Benchmark draft eager vs graph (expect a real speedup here) and record it.

---

## Risks & open questions

- **Default-stream kernel launches (RESOLVED diagnosis, Phase 1 fixes)** — all custom kernels
  launch on stream 0, so capture records nothing. This was THE blocker; fixed by launching on
  `getCurrentCUDAStream()`. Every other phase depends on this landing first.
- **Graph-pool replay correctness** — the original Phase 0 worry; effectively settled (rmsnorm
  captures/replays cleanly via the caching allocator). Re-confirmed by Phase 1.3 after the stream
  fix. If any single op still diverges then, fall back to an `_out` buffer for *that* op only.
- **Grid dims vs device scalar** — replay reuses capture-time grid/block. Safe only if grid is
  independent of `cur_len`/`write_pos`. Verified by reading kernels in Phase 1; the decode/verify
  kernels loop over `cur_len` internally, so grid depends on (batch, heads, S) — all fixed.
- **cuBLAS workspace under capture** — qkv/o_proj/lm_head use cuBLAS; ensure warmup initializes
  workspace before capture (≥3 warmup iters; set a fixed cuBLAS workspace if needed).
- **Re-capture cost** — capture once per (shape) and reuse; never per token.
- **Memory** — captured graph pins its pool; account for it in `buffer_fits_vram_budget` on 24 GB.
- **Sampling stays outside the graph** — argmax/softmax/accept-reject run eager on the output
  logits; only the forward is replayed.

---

## Future optimizations (post-decode-MVP)

Not blockers for a working graphed decode; revisit once the MVP lands and is benchmarked.

- ⬜ **Fold the RoPE slice+update into the graph.** Replace the per-step host-side `static_cos/sin`
  refresh with in-kernel indexing of the full `rope_cos/rope_sin` table by the device position
  scalar (`cache_position`). Removes the last per-step host launch on the decode path; the full
  table is already resident so there's no extra memory cost. Needs the `_dev` attention +
  rope-kv-write kernels to accept `(full_table, device_row_offset)` instead of pre-sliced cos/sin.
  (See Phase 3 note.)
- ⬜ **Persistent static logits output.** Optionally `copy_` the captured logits into
  `buffers.logits` (stable address) so callers never see a pool-internal tensor.
- ⬜ **Graph the prefill** (bucketed by length) if prefill launch overhead turns out to matter.

## Findings log

(Append dated notes as phases complete — e.g. Phase 0 conclusion, measured speedups, surprises.)

- 2026-06-06 — Plan created. Repo audit: kernels are caching-allocator-only (graph-pool safe);
  real blocker is host-scalar `write_pos`/`cur_len` + host-sliced rope. Targeting DECODE first,
  then VERIFY. Phase 0 will empirically confirm the allocation hypothesis before kernel surgery.

- 2026-06-06 — **Phase 0 done. Root cause found, plan reordered.**
  - Baseline: eager 7B decode = **33.8 tok/s (29.6 ms/token)**, batch=1, Quadro RTX 6000.
  - `phase0_graph_diag.py` (progressive capture) results: single rmsnorm "passed" but ONLY because
    its graph captured *nothing* (PyTorch warned "CUDA Graph is empty… wrong device or stream");
    a one-decoder-layer capture already diverged from eager and **grew across replays**
    (−2.4 → −1.6 → −34 → −47); full stack → NaN.
  - **Root cause: every custom kernel launches with bare `<<<grid,block>>>` = the default stream
    (stream 0), and NONE use `getCurrentCUDAStream()` (grep-confirmed).** `torch.cuda.graph`
    captures on a private non-default stream, so our kernels run for real on stream 0 during
    "capture" but are never *recorded*; replay re-runs only the few captured PyTorch view/copy ops
    over pool buffers nothing fills → garbage that grows as pool memory churns.
  - **The allocation hypothesis is therefore dead** — `torch::empty` via the caching allocator is
    fine (rmsnorm proves it); no need to rewrite ops allocation-free, contra `handoff.md`.
  - Plan reordered: **new Phase 1 = launch every kernel on the current stream** (small, mechanical,
    eager-safe), then re-run the Phase 0 diagnostics as the true validation, *then* device-scalar
    positions (now Phase 2) etc.
  - Note: also moved `benchmarks/` → `runtime/benchmarks/` so all engine code lives under `runtime/`.

- 2026-06-07 — **Phase 1 done (stream fix) + validated.**
  - Added `, 0, at::cuda::getCurrentCUDAStream()` to all 7 hand-written kernel launches; rebuilt.
  - Re-ran the diagnostic: "empty graph" warning gone; `replay == eager` exactly at every stage
    (op → layer → stack → full forward). Fixed-position probe: replay vs eager max|Δ| = **0.0000**.
  - **Allocation hypothesis confirmed dead-end** — caching-allocator intermediates replay bit-exact.
  - New concept learned & documented: once kernels launch on the capture stream, **capture records
    but does not execute** — the output buffer is zeros until the first `replay()` (pre-replay
    "capture vs eager" = 9.87 is expected, not a failure). See `cuda_graph_issues_and_concepts.md`.
  - Started `documentation/cuda_graph_issues_and_concepts.md` (Issue #1 = the stream bug).
  - Next: Phase 2 — device-scalar `write_pos`/`cur_len` so the graph isn't frozen at the
    capture-time position (needed before a graph can drive *real* multi-step decoding).

- 2026-06-07 — **Phase 2 done (device-scalar positions) + validated.**
  - Added `_dev` variants reading `write_pos`/`cur_len` from a 0-d int64 CUDA scalar:
    `rope_kv_write_forward_dev`, `decode_attn_forward_dev`, `small_q_attn_forward_dev`. One
    launcher/dispatch serves both paths via an optional `const int64_t*` override (null → host int).
  - `test_attention_dev_scalar` 5/5 — dev == host **bit-exact** across positions; `test_attention`
    8/8 regression-clean.
  - Concept documented (Concept #2): captured scalars are frozen, so per-step positions must be
    device memory; host-side value-range checks can't be kept on the `_dev` path (no sync allowed),
    so they move to the caller's contract. S (query len) stays a host shape (fixed per graph).
  - Build gotcha: attention `kernel.cu` OOM-kills `cicc` on the login node → build via `srun`.
  - Next: Phase 3 — static buffers (`static_input_ids`, `cur_len` scalar, `static_cos/sin`) + an
    allocation-stable `_decode_forward_static()` that uses the `_dev` ops, then capture (Phase 4).

- 2026-06-07 — **Phase 3 done (static decode forward) + validated.**
  - `RuntimeBuffers`: added `static_input_ids` / `static_cur_len` / `static_cos` / `static_sin`
    (+ `refresh_decode_rope`). Kept out of the seq-scaling memory plan (sub-KB fixed scratch).
  - Executor: `static_decode` branch (uses `_dev` ops + `cache_position`/`static_cur_len` device
    scalars + static cos/sin); `_prepare_decode_static` (per-replay update, outside graph) /
    `_decode_forward_static` (the capturable region) / `decode_step_static` (eager driver).
  - `test_decode_static`: `decode_step_static` == eager `decode_step` **bit-exact** over a 6-step
    greedy trajectory. 2/2 OK.
  - Concept #3 documented: the prepare/forward split *is* the capture/replay boundary (update fixed
    buffers in place eagerly; capture+replay the read-only forward).
  - User-requested future opt added to plan: fold the host-side RoPE slice/copy into the graph via
    in-kernel indexing of the full table by the device position scalar (memory-free; removes the
    last per-step host launch). Tracked under "Future optimizations".
  - Next: Phase 4 — capture `_decode_forward_static` into a `torch.cuda.CUDAGraph` (side-stream
    warmup, shared pool) and replay per token via `decode_step_graph` (prepare → replay → read logits).

- 2026-06-07 — **Phase 4+5 done (graph capture/replay) — correct, but the headline finding is that
  7B decode is memory-bound, so speedup = 1.00×.**
  - `decode_step_graph` + `_capture_decode_graph` in the executor; lazy one-time capture, one graph
    reused across prompts. `test_decode_graph`: graphed == eager **bit-exact** over 8 steps + reuse
    across prefills. 2/2 OK.
  - `phase4_graph_decode.py`: eager **33.6** vs graph **33.6 tok/s** → **1.00×**. Host dispatch does
    drop (28.1 → 20.6 ms/tok) but stays under the 29.7 ms/tok GPU floor (~15 GB weights / 672 GB/s ≈
    22 ms hard floor, measured 29.7 ≈ 74% BW). **Decode is memory-bandwidth-bound, not launch-bound.**
  - The "super launch-bound" premise is false for 7B batch-1 decode. Graph machinery is correct &
    reusable; its real payoff is the **0.5B draft** (Phase 8d, ~14× less weight traffic), larger
    batch, or quantized weights. A 7B VERIFY graph would also be ~1.00×.
  - Concept #4 written up (launch-bound vs memory-bound; measure before optimizing). Phase 5.5 /
    Phase 6 (7B verify) deprioritized accordingly.
  - **Decision point for the user:** where to point the (working) graph machinery next.

- 2026-06-07 — **Target-model graphs COMPLETE (user chose to finish them for availability).**
  - Phase 5.5: `greedy_extend(use_cuda_graph=…)` routes to `decode_step_graph`.
  - Phase 6: `verify_gamma_graph` — one graph per query length S (= γ or γ+1), bit-exact vs eager
    (`test_verify_graph` 3/3). Unified the static attention path (`static_attn` flag + `_static_ctx`;
    S=1→decode kernel, S>1→small_q kernel); decode/dev-scalar regressions stay green.
  - Benchmark data recorded in `documentation/target_graph_benchmarks.md` (decode 1.00×, verify
    ~1.00×, with the memory-bandwidth analysis). Target graphs are an **available option**, not a
    7B speedup.
  - Next effort (separate): apply the same pipeline to the **0.5B draft** (launch-bound → real win).
    Draft kernels being ported into `draft_model_files/` (synced from `origin/eli_dev`).
