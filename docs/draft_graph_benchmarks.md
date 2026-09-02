# Draft-Model CUDA Graph + Speculative Decoding Benchmarks

Results for the 0.5B **draft** graphs and the single-process draft+target speculative
decoding loop. Companion to `docs/target_graph_benchmarks.md` (the 7B target, where
graphs are 1.00× because it's memory-bound).

- **Hardware:** NVIDIA Quadro RTX 6000 (Turing, sm_75), 24 GB, ~672 GB/s
- **Models:** Qwen2.5-7B target + Qwen2.5-0.5B draft, fp16, batch = 1
- **Date:** 2026-06-07

## 1. Draft decode: graphs pay off (unlike the target)

`runtime/benchmarks/draft_graph_decode.py` (0.5B, batch=1):

| Path | ms/token | tokens/sec | speedup |
|------|----------|-----------:|--------:|
| eager (`decode_step`)            | 6.06 | 165.0 | — |
| CUDA graph (`decode_step_graph`) | 3.71 | 269.9 | **1.64×** |

Host-side dispatch (CPU time to queue one step): eager **5.42 ms** → graph **2.18 ms**. Here the
draft IS launch-bound: eager host dispatch (5.42 ms) ≈ the GPU time (6.06 ms), so the host is the
bottleneck and the graph removes most of it. Graph output is **bit-exact** vs eager.

**Why the draft and not the 7B target?** ~14× less weight memory (~1 GB vs ~15 GB) → ~1.5 ms of HBM
traffic per token instead of ~22 ms, so the ~150 host launches dominate. (Target: memory-bound,
1.00×. Draft: launch-bound, 1.64×. See Concept #4.)

## 2. End-to-end speculative decoding (one GPU, greedy)

`runtime/benchmarks/spec_decode_bench.py` — 7B target + 0.5B draft, γ=4, greedy standardization,
synthetic 10-token prompt, 96 new tokens:

| Path | ms/token | tokens/sec | vs target |
|------|----------|-----------:|----------:|
| target-only greedy             | 29.54 | 33.9 | 1.00× |
| spec-decode, draft graph **OFF** | 31.29 | 32.0 | **0.94×** |
| spec-decode, draft graph **ON**  | 25.02 | 40.0 | **1.18×** |

- **Accept rate: 1.09 / 4** drafts per iteration (46 iters). Low because the prompt is short and
  synthetic, so the 0.5B and 7B diverge; a real, longer prompt gives a higher accept rate and a
  larger spec-decode win.
- **The draft graph is essential here:** at this accept rate, spec decode *without* the draft graph
  is **slower than plain target greedy (0.94×)**; the draft graph turns it into a net win (1.18×).
  Draft-graph contribution within spec decode: **1.25×**.

Speculative-decode throughput scales with accept rate. The infrastructure win (the draft graph) is
fixed (~1.6× on draft decode); the end-to-end win is `f(accept_rate)` on top of it.

## 3. Correctness

- **Greedy spec decode reproduces the target's own greedy sequence exactly**
  (`test_spec_decode.test_greedy_matches_target_greedy`) — the strong correctness gate for the whole
  draft+target+verify+sampler+rollback pipeline.
- Stochastic accept/reject runs end-to-end with **vocab alignment** (draft 151936 → target 152064
  padded with −∞; see Concept #5).
- Draft graph decode is bit-exact vs eager; draft executor matches HF Qwen2.5-0.5B.

## 4. Takeaways

1. CUDA graphs are the right tool for the **launch-bound draft** (1.64×), not the memory-bound 7B
   target (1.00×) — measure the bottleneck.
2. At low accept rates the **draft graph is what makes spec decode worthwhile** at all (0.94× → 1.18×).
3. To push end-to-end speed further: raise the accept rate (better/realistic prompts, tuned γ) and/or
   quantize/batch the target (its memory bandwidth is the hard floor). Cross-GPU MPI (Phase 8c) would
   overlap draft and target instead of serializing them on one card.
