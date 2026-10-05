# Custom CUDA inference demo

**Eli Wandless** — CME 213, Stanford, Spring 2026.

This branch runs **Qwen2.5-7B-Instruct** with hand-written CUDA kernels and compares ordinary greedy decoding with **Qwen2.5-0.5B-Instruct** speculative decoding. Both models stay on one GPU. The web UI streams target-only output first, then speculative output alongside it.

The forward pass uses `runtime/production_kernels/target` and `runtime/production_kernels/draft`. Embedding, RMSNorm, RoPE/cache writes, attention, SwiGLU, and residual operations use the production extensions; matrix projections use cuBLAS. Hugging Face supplies the weights and tokenizer, and serves as the test reference. It does not run the model in the demo server.

## Run the demo

Inside a CUDA development environment with GPU-enabled torch and `nvcc`:

```bash
bash demo/setup.sh
.venv/bin/python -m demo.server
```

Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)**. For a remote GPU, forward that port over SSH. [Setup, connection, and validation instructions](demo/README.md) include the current Runpod session.

## Walk through the code

For the full request lifecycle, read [demo/WALKTHROUGH.md](demo/WALKTHROUGH.md). It follows compilation, model loading, every decoder stage, speculative verification/cache rollback, and streamed text, with matching source comments and line references.


| Order | File                                                                       | What to explain                                                                           |
| ----- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 1     | [demo/index.html](demo/index.html)                                         | One prompt, two streaming panels, timing and token agreement.                             |
| 2     | [demo/server.py](demo/server.py)                                           | Load both models, format the shared prompt, serialize the two runs, stream results.       |
| 3     | [runtime/executor.py](runtime/executor.py)                                 | Compose kernels into prefill, single-token decode, and multi-token verification.          |
| 4     | [runtime/production_kernels](runtime/production_kernels)                   | `ops.py` → C++ bindings → CUDA implementation, with separate target and draft dimensions. |
| 5     | [demo/decoding.py](demo/decoding.py)                                       | The speculative loop, output limits, EOS, and cancellation cleanup.                       |
| 6     | [runtime/speculative/draft_runner.py](runtime/speculative/draft_runner.py) | Draft γ tokens, then roll back rejected tokens and commit the target bonus.               |
| 7     | [runtime/speculative/target_step.py](runtime/speculative/target_step.py)   | Verify the proposals, accept the matching prefix, and choose the target bonus.            |
| 8     | [runtime/executor_graph.py](runtime/executor_graph.py)                     | Replay the draft decode forward with fixed buffers and device-side positions.             |
| 9     | [runtime/core](runtime/core) and [runtime/buffers.py](runtime/buffers.py)  | Model dimensions, safetensors loading, buffer sizes, KV cache, and RoPE tables.           |


Kernel reading guides: [embedding](runtime/production_kernels/target/embedding/kernel_walkthrough.md), [RMSNorm](runtime/production_kernels/target/rmsnorm/kernel_walkthrough.md), [attention](runtime/production_kernels/target/attention/kernel_walkthrough.md), [SwiGLU](runtime/production_kernels/target/swiglu/kernel_walkthrough.md), and [residual/LM head](runtime/production_kernels/target/residual_ops/kernel_walkthrough.md).

## What the comparison means

Both paths use **greedy argmax**, the same prompt, and the same output budget. The draft proposes up to γ tokens. The target evaluates those positions together, accepts the prefix that matches its own choices, and emits its own token at the first mismatch (or an extra token if all drafts match). Only verified tokens reach the UI.

The target defers that bonus token's cache write until the next verification pass. The draft rewinds its cache to the accepted prefix and commits the bonus immediately. This keeps both models on the same logical sequence while avoiding a separate target forward for the bonus.

The UI compares actual token IDs. Batched verification and single-token decoding can differ slightly numerically; a changed argmax is reported as an output mismatch. Speedup is target elapsed time divided by speculative elapsed time, including prefill and streaming. It varies with acceptance rate and prompt length. Speculation may improve total time while delaying the first visible chunk.

## Repository layout

```text
 demo/                       Web UI, server, speculative streaming, setup, serving tests
 runtime/
   core/                     Model configurations, shapes, buffer sizing, weight loading
   production_kernels/       Target and draft CUDA extensions + kernel walkthroughs
   speculative/              Greedy draft runner, target verifier, forward-stage types
   tests/                    Kernel, model-reference, and draft graph correctness tests
   buffers.py                KV cache, RoPE tables, and graph inputs
   executor.py               Full model forward and cache management
   executor_graph.py         Draft decode graph capture and replay
 setup.py                    Build the production CUDA extensions
```

The historical cluster jobs, MPI implementation, stochastic sampler, development kernels, benchmarks, and report are retained in Git on `main`. This branch contains the single-GPU web demo and its relevant tests.

## Performance evidence

The [benchmarking suite](benchmarking/README.md) measures the production target and draft kernels against explicit roofline bounds and compares target-only Qwen2.5-7B-Instruct decoding with vLLM on the same GPU. The [measured report](benchmarking/results/REPORT.md) includes latency tables, correctness checks, raw samples, and limitations.

The [attention optimization report](benchmarking/optimizations/attention_v1/REPORT.md) records paired before/after measurements and rollback instructions. The [decoder-layer profile](benchmarks/README.md) includes a small native Nsight Systems report and [interpretation notes](benchmarks/PROFILE_NOTES.md).

For interview preparation, use the [spoken walkthrough](demo/INTERVIEW_WALKTHROUGH.md).
