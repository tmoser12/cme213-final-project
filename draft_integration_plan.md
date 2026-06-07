# Draft-Model Integration Plan (0.5B draft + CUDA graphs)

Living plan for integrating the Qwen2.5-**0.5B draft** model into the speculative-decoding
pipeline and CUDA-graphing it — where graphs *actually* pay off (the draft is launch-bound, unlike
the memory-bound 7B target; see `documentation/target_graph_benchmarks.md` / Concept #4).

Companion to:
- `graph_plan.md` — the (complete) target-model graph effort; this reuses its machinery.
- `documentation/cuda_graph_issues_and_concepts.md` — concepts (streams, device scalars, etc.).
- `documentation/speculative_decoding.md` — the accept/reject algorithm.
- `runtime/plan.md` — overall engine plan (Phase 8 = speculative decoding).

Conventions (gpu-cluster skill): SLURM only, `source setup.sh`, sm_75/FP16, 30-min jobs, build
attention on a compute node (login-node `cicc` OOM), `PYTHONNOUSERSITE=1`. Log to `project_logs/`.

---

## Two deliverables (user)

1. **Draft API** — generate γ drafts for the target to verify; apply the target's feedback
   (n_accepted + bonus token) and continue.
2. **Draft CUDA graphs** — modify the draft kernels (stream-correct launches + device-scalar
   positions) exactly as we did for the target, then graph the draft decode loop. This is the
   one with a real speedup.

A single-process (one-GPU) draft+target loop validates both end-to-end (no MPI — that's a separate
future phase). 7B (~14 GB) + 0.5B (~1 GB) fit together on the 24 GB card.

> ⚠️ **`draft_model_files/` is a scratch sync of an OLD `runtime/` (from `origin/eli_dev`) — do NOT
> port its runtime code.** It contains a stale executor, buffers, tests, configs, and even a `target/`
> tree that would clobber the graph work in our live `runtime/`. **Take ONLY the draft kernel sources**
> (`production_kernels/draft/<op>/{kernel.cu, bindings.cpp, ops.py, __init__.py}`) and nothing else.
> Everything else the draft needs (executor, buffers, configs, spec API) we build/extend in the live
> `runtime/`. The `0.5b` YAML already exists in `runtime/core/configs/`.

---

## Architecture decisions

- **Reuse `Qwen2Executor`** with `kernel_set="draft"` (head_dim=64, draft ops). No new executor
  class — the decode/prefill/graph machinery is identical; only the kernel set + dims differ.
- **Draft only needs decode** (prefill + S=1 decode). It never runs VERIFY (the target does). So
  the draft graph = the existing `decode_step_graph`; the draft `_dev` kernels needed are just
  `rope_kv_write_forward_dev` + `decode_attn_forward_dev` (`small_q_dev` optional, for symmetry).
- **Draft orchestration in `runtime/speculative/draft_runner.py`** (a `DraftRunner` wrapping a draft
  executor + rng + `_last_logits`), mirroring how `target_step.py` wraps the target executor. Keeps
  the generic executor clean.
- **Sampling:** the draft samples xᵢ ~ softmax(qᵢ) (stochastic, the reference algorithm) on the GPU
  to avoid per-step D2H in the γ-loop; it reports the full qᵢ logits for the target's ratio test.
  Sampling stays **outside** any graph (host/GPU decision between replays).
- **Vocab alignment (critical):** draft vocab 151936 ≠ target vocab 152064. The accept/reject math
  needs p and q over the same index space. Decision: **compare on the shared `[0, 151936)` range** —
  slice the target's p logits to the draft vocab before the ratio/`p−q`. The target still *samples*
  the bonus over its full 152064, but a bonus id ≥ 151936 can't be embedded by the draft → clamp/skip
  (rare; those are reserved/near-unused Qwen2.5 ids). Handle explicitly and test.

---

## Phases & TODOs

Status: ⬜ todo · 🟡 wip · ✅ done · 🔵 deprioritized

### Phase D0 — Bring draft kernels into the tree + build system ✅ DONE
- ✅ Copied draft kernel sources into `runtime/production_kernels/draft/`; generalized `setup.py`
  (`BUILD_ROLE`/`BUILD_KERNEL`, with the `residual_ops` → `<role>_residual_ops` naming quirk handled)
  and `scripts/build_kernels.sh` (`[role] [op]`, back-compat). Built all 5 draft exts on a compute
  node; all import. ⚠️ build from project root (shell CWD can drift).

<details><summary>original D0 tasks</summary>

- ⬜ D0.1 Copy ONLY the draft **kernel sources** from
  `draft_model_files/runtime/production_kernels/draft/<op>/` → `runtime/production_kernels/draft/<op>/`
  for the 5 ops (attention, embedding, rmsnorm, residual_ops, swiglu): `kernel.cu`, `bindings.cpp`,
  `ops.py`, `__init__.py` (+ `production_kernels/draft/__init__.py`). Skip `benchmark_scripts/` and
  `kernel_walkthrough.md`. **Do not copy anything else from `draft_model_files/`.** These become tracked.
- ⬜ D0.2 Add 5 draft `CUDAExtension`s to `setup.py` (`draft_<op>_ops`, package prefix
  `runtime.production_kernels.draft`, `-arch=sm_75 -O3`); extend `BUILD_KERNEL` selection.
- ⬜ D0.3 Extend `scripts/build_kernels.sh` to accept draft targets (e.g. `draft_attention` or a
  `draft` group). Build all draft kernels **on a compute node**.
- ⬜ D0.4 Smoke: import each `runtime.production_kernels.draft.<op>` ext; confirm `.so` colocated.

</details>

### Phase D1 — Draft kernel stream correctness (mirror target Phase 1) ✅ DONE
- ✅ Added `, 0, at::cuda::getCurrentCUDAStream()` to all 7 draft launch sites + `ATen/cuda/CUDAContext.h`
  include to embedding/rmsnorm. Validated via D4 decode-graph capture (no "empty graph" warning).

### Phase D2 — Draft device-scalar attention (mirror target Phase 2) ✅ DONE
- ✅ Added `rope_kv_write_forward_dev`, `decode_attn_forward_dev`, `small_q_attn_forward_dev` to the
  draft attention kernel/bindings/ops/__init__ (same optional-`const int64_t*` override pattern as
  target). `test_draft_attention_dev_scalar` — `_dev` == host **bit-exact** (D=64), 4/4 OK.

### Phase D3 — Generalize the executor for the draft (Phase 3c: kernel_set) ✅ DONE
- ✅ Added `kernel_set` to `RuntimeConfig` (default `target`) + both YAMLs (0.5b→draft, 7b→target).
  `_import_kernels` imports the role's ops dynamically; `_ATTN_HEAD_DIM` is now `{target:128, draft:64}`
  and the gate checks per-role; executor defaults `kernel_set` from `cfg.kernel_set`. Fixed the now-obsolete
  `test_head_dim_gate_for_05b` → `test_kernel_set_head_dim_gate`.
- ✅ `test_draft_executor` — draft executor == HF Qwen2.5-0.5B for prefill, decode, and the greedy
  trajectory; uses draft kernels. 4/4 OK (+2 structure).

### Phase D4 — Draft CUDA graphs (the payoff) ✅ DONE
- ✅ `decode_step_graph` works for the draft unchanged. `test_draft_decode_graph` — graphed draft
  decode == eager **bit-exact** over an 8-step trajectory + reused across prefills. 2/2 OK.
- ✅ **Benchmark** `runtime/benchmarks/draft_graph_decode.py` (0.5B, batch=1, RTX 6000):
  eager **165 tok/s (6.06 ms/tok)** → graph **270 tok/s (3.71 ms/tok)** = **1.64× speedup**,
  bit-exact. Host dispatch 5.42 → 2.18 ms/tok. **The draft IS launch-bound — graphs pay off here**
  (vs 1.00× on the 7B target). 🎯

### Phase D5 — Draft API (generate γ + apply feedback) ✅ DONE
- ✅ `runtime/speculative/draft_runner.py` — `DraftRunner` with `prefill`, `generate_drafts(γ, greedy=)`
  → `(draft_ids[1,γ], q_logits[γ+1, vocab])` (uses `decode_step_graph` when `use_cuda_graph`; clones
  each logit row since the graph buffer is reused), and `apply_target_feedback(n, bonus)` (rollback +
  commit bonus + refresh q₁ + bonus-vocab guard).
- ✅ `test_draft_runner` — shapes, KV cursor (+γ then rollback to prefix+n+1), multi-iteration
  positions, greedy drafts == `greedy_extend`, bonus-vocab guard. 5/5 OK.

### Phase D6 — Single-process spec-decode integration (no MPI) ✅ DONE
- ✅ `runtime/speculative/spec_decode.py` — `speculative_generate(target, draft, prompt, n, γ, greedy=)`:
  loops generate_drafts → verify+sample → apply_target_feedback → flush. Greedy path uses argmax
  acceptance (`_greedy_target_step`); stochastic path uses `target_speculative_step` (vocab-aligned).
- ✅ D6.2 **Strong gate:** `test_spec_decode.test_greedy_matches_target_greedy` — greedy spec decode
  == target `greedy_extend` **exactly** (both models on one GPU). Stochastic path runs end-to-end.
- ✅ D6.3 **End-to-end benchmark** `spec_decode_bench.py` (γ=4, greedy, synthetic prompt): target
  greedy **33.9**, spec draft-graph-OFF **32.0 (0.94×)**, spec draft-graph-ON **40.0 (1.18×)**; accept
  rate 1.09/4. **The draft graph is what makes spec a net win** (0.94→1.18×; +1.25× within spec).

### Phase D7 — Docs & wrap-up ✅ DONE
- ✅ `documentation/draft_graph_benchmarks.md` (draft 1.64×, end-to-end spec numbers + accept-rate
  caveat). Concept #5 (vocab alignment) in the issues journal. `runtime/plan.md` Phase 8d updated.
- 🔵 D7.3 MPI 2-rank (Phase 8c) noted as the separate next step for cross-GPU (would overlap draft +
  target instead of serializing them on one card).

---

## Risks & open questions

- **Vocab mismatch (151936 vs 152064)** — handled by comparing on the shared range + bonus-id guard
  (see Architecture). Needs an explicit test; the cleanest fix lives in `sampler`/`target_step`.
- **VRAM** — 7B+0.5B ≈ 15 GB + buffers/KV; should fit in 24 GB but watch two model copies + two sets
  of buffers. Use `buffer_fits_vram_budget`; load both once.
- **Draft graph capture writes KV during warmup** — same idempotent-write reasoning as the target;
  capture after a prefill at a valid position.
- **Sampling RNG determinism** — draft GPU sampling vs the target host `random.Random`; seed both for
  reproducible tests. Greedy-standardized mode sidesteps RNG entirely (D6.2).
- **`head_dim=64` kernels** — partner code; treat their eager correctness as the baseline and only
  add stream + `_dev` (don't refactor their math). Confirm D=64 paths in `decode_attn`/`small_q`.
- **Two `graph_pool_handle`s / two executors** — each executor has its own graph + pool; fine, they
  don't share buffers. Keep draft and target graph state independent.

## Validation gates (per phase)

| Phase | Gate |
|-------|------|
| D0 | draft exts import |
| D1 | draft capture non-empty |
| D2 | draft `_dev` == host bit-exact (D=64) |
| D3 | draft executor == HF 0.5B (eager) |
| D4 | graphed draft decode == eager bit-exact; **graph speedup measured** |
| D5 | draft API shapes/positions correct |
| D6 | greedy-standardized spec decode == target greedy; speedup measured |

## ✅ Draft integration COMPLETE (2026-06-07) — D0–D7 all done

The 0.5B draft runs under the same engine (`Qwen2Executor(kernel_set="draft")`), is CUDA-graphed,
and drives speculative decoding against the 7B target end-to-end (one GPU, no MPI). Headline numbers:
**draft decode 1.64× graphed**; **greedy spec decode == target greedy exactly**; **end-to-end 1.18×**
(and the draft graph is what makes spec a net win vs 0.94× without it). All paths bit-exact / parity-
checked. Tests: `test_draft_attention_dev_scalar`, `test_draft_executor`, `test_draft_decode_graph`,
`test_draft_runner`, `test_spec_decode` — all green.

**Next (optional, separate):** MPI 2-rank (Phase 8c) to overlap draft + target across two GPUs;
raise accept rate (realistic prompts / tuned γ); quantize the target to lift its memory-bandwidth floor.

## Findings log
- 2026-06-07 — Plan created. Draft kernels (D=64) synced into `draft_model_files/`, currently
  pre-stream-fix. Reusing the target graph pipeline; the draft is launch-bound so graphs should give
  a real win here (unlike the 7B target). Vocab mismatch + single-process two-model loop are the
  main new wrinkles.
- 2026-06-07 — **D0–D7 done in one pass.** Draft kernels copied in + stream-fixed + `_dev`-ified
  (bit-exact D=64); executor generalized via `kernel_set` (draft==HF 0.5B); draft decode graph
  **1.64×** (launch-bound, as predicted); `DraftRunner` API + single-process spec loop; greedy spec
  == target greedy exactly; end-to-end **1.18×** at 1.09/4 accept (draft graph flips 0.94→1.18×).
  Vocab mismatch handled by −∞ padding of draft q + bonus guard (Concept #5).
