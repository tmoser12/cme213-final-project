# Target-Model CUDA Graph Benchmarks (Qwen2.5-7B)

Standalone record of the CUDA-graph results for the **7B target** model. See
`cuda_graph_issues_and_concepts.md` Concept #4 for the full analysis and
`cuda_graph_issues_and_concepts.md` for the implementation journal.

- **Hardware:** NVIDIA Quadro RTX 6000 (Turing, sm_75), 24 GB GDDR6, ~672 GB/s
- **Model:** Qwen2.5-7B-Instruct, fp16, batch = 1
- **Date:** 2026-06-07
- **Repro:** `bash slurm/run_python.sh runtime/benchmarks/phase4_graph_decode.py`

## Headline: decode tokens/sec (batch=1)

| Path | ms/token | tokens/sec | speedup |
|------|----------|-----------:|--------:|
| eager (`decode_step`)        | 29.73 | 33.6 | — |
| CUDA graph (`decode_step_graph`) | 29.74 | 33.6 | **1.00×** |

**The graph gives no wall-clock speedup on 7B decode** — and that is expected, not a
bug (the graph output is bit-exact, see below).

## Why 1.00×: 7B decode is memory-bandwidth-bound, not launch-bound

Every decode token must stream all ~15 GB of fp16 weights from HBM:

```
floor    = 15 GB / 672 GB/s  ≈ 22 ms/token   (pure weight read)
measured = 29.73 ms/token                     ≈ 74% of peak bandwidth
```

The graph removes host launch overhead, but that overhead was already hidden behind GPU work.
Host-side dispatch time (CPU time to *queue* one step, no GPU wait):

| Path | host dispatch ms/token |
|------|------------------------:|
| eager | 28.1 |
| graph | 20.6 |

Both are **below** the 29.7 ms GPU time, so the CPU never becomes the bottleneck → wall-clock
is unchanged. (At batch=1 the GEMMs are bandwidth-starved: each weight is used once per token, so
arithmetic intensity is ~1.)

## Correctness (the graph is exact)

- `decode_step_graph` reproduces eager `decode_step` **bit-exactly** (`torch.equal`) over an
  8-step greedy trajectory — `runtime/tests/test_decode_graph.py`.
- `verify_gamma_graph` reproduces eager `verify_gamma` **bit-exactly** for both the no-bonus
  (S=γ) and leading-bonus (S=γ+1) paths — `runtime/tests/test_verify_graph.py`.
- One captured graph is reused across prompts (positions are device scalars; KV/buffer addresses
  are stable).

## VERIFY graph (speculative-decode forward)

The 7B `verify_gamma_graph` (S = γ or γ+1) is also **~1.00×** and for the same reason — in fact
more so, since γ+1 query rows share a *single* weight read, making it even more compute-dense than
decode. It is implemented for correctness/availability, not speed.

## Takeaways

1. **Measure the bottleneck before optimizing.** "Many small kernels ⇒ launch-bound" is a
   hypothesis; here it's false for single-stream 7B decode.
2. **The graph machinery is correct and reusable** (stream-correct kernels → device-scalar
   positions → static buffers → capture/replay), exposed via `use_cuda_graph` /
   `decode_step_graph` / `verify_gamma_graph`. It's available whenever wanted; it just doesn't move
   7B wall-clock.
3. **Where it *will* pay off:** the 0.5B **draft** model (~14× less weight traffic ⇒ genuinely
   launch-bound), larger batch, or quantized (int8/int4) weights. Draft kernels are being ported in
   (`draft_model_files/`); the draft executor is the next place to apply this same pipeline.
4. **To actually speed up 7B decode**, the lever is *weight traffic*, not launches: quantization or
   batching — a separate effort from CUDA graphs.
