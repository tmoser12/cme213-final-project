# Decoder-layer Nsight profile

Open **[target_layer0_decode_ctx512.nsys-rep](target_layer0_decode_ctx512.nsys-rep)** in NVIDIA Nsight Systems. This is a native report, captured with **2026.2.1**, matching the installed local viewer. The report and supporting files have been copied back from the GPU.

## What was captured

- Qwen2.5-7B-Instruct **target layer 0**, using the current optimized production kernels and real checkpoint weights.
- RTX 4090, FP16, batch **1**, **one decode query**, **512 previously cached tokens**. Attention sees **513 tokens**, including the current query's newly written key/value. Cache capacity is 2,048.
- Ten eager layer invocations after 20 warmup invocations. Each repeats the same input and cache position; these are not ten newly generated tokens.
- The prefix cache was populated by processing real benchmark-prompt embeddings through layer 0. The decode input is the next token's real embedding, taken from the fixed prompt continuation, rather than sampled from a whole-model generation.
- Only the decoder layer is inside capture. Loading, prefix preparation, warmup, embedding lookup, the other 27 layers, final model normalization, and the vocabulary head are excluded.

The script calls the production executor's `_run_decoder_layer`; NVTX wrappers label its existing operations. Outputs before and after annotation, and after capture, matched bitwise. Loading only one layer allowed the existing demo server to remain running. It was idle during collection.

## Where to look in the viewer

Expand the profiled Python process's **NVTX**, **CUDA API**, and GPU **CUDA kernels/stream** rows. Find the outer `decoder_layer_decode` range and its `decode_iteration_000` through `decode_iteration_009` children. Zoom into one iteration's correlated GPU work.

The labels identify this sequence:

1. `rmsnorm_forward`: input normalization.
2. `attention_block`: QKV projection, `rope_kv_write_forward`, `decode_attn_forward`, and output projection.
3. `residual_add_forward`: attention residual.
4. `rmsnorm_forward`: post-attention normalization.
5. `swiglu_forward`: gate/up projections, custom gated activation, and down projection.
6. `residual_add_forward`: MLP residual.

`decode_attn_forward` contains the split attention and combine kernels. cuBLAS selects **GEMV kernels** for these single-row projections even though the runtime invokes its matrix-multiplication interface. The cuBLAS NVTX ranges and CUDA launch correlations help distinguish the five projections.

NVTX ranges measure **CPU submission scopes**, not necessarily completion of their GPU work. There is no synchronization between individual operations or iterations; the final device synchronization is inside the outer capture range. Use correlated GPU kernel intervals for execution time, rather than reading a Python NVTX duration as GPU latency.

## Trace validation and scope

The exported trace contains **130 kernel executions**, exactly **13 per layer invocation**:

| Kernel group | Count across 10 calls |
|---|---:|
| cuBLAS projections | 50 |
| RMSNorm | 20 |
| Residual add | 20 |
| RoPE/KV write | 10 |
| Split decode attention | 10 |
| Attention combine | 10 |
| SwiGLU activation | 10 |

See [kernel summary](target_layer0_decode_ctx512_cuda_gpu_kern_sum.csv), [NVTX summary](target_layer0_decode_ctx512_nvtx_pushpop_sum.csv), and [CUDA API summary](target_layer0_decode_ctx512_cuda_api_sum.csv). The report was successfully exported to SQLite, and the downloaded report's SHA-256 matches the GPU copy in [report.sha256](report.sha256).

This is an **isolated layer profile**, not a full-model latency benchmark. Repeated layer execution changes cache residency relative to interleaving all 28 layers. Python NVTX labels and profiler instrumentation add overhead. The demo remained resident in GPU memory, but it was not generating concurrently. CPU stack sampling and context-switch sampling were disabled; CUDA, NVTX, and cuBLAS tracing were enabled.

## Nsight Compute attempt

Nsight Compute **2025.1.1** was also run with the full metric set for one invocation. The host denied access to GPU counters with `ERR_NVGPUCTRPERM`, so **there is no usable `.ncu-rep`**. The precise failure is retained in [ncu.log](ncu.log). Changing the local viewer does not resolve the GPU host's counter-access restriction. See NVIDIA's [counter-permission diagnostic](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Reproduce on the GPU

From the repository root, run:

```sh
bash benchmarks/capture_decode_layer.sh
```

The script uses `nsys` on PATH. Set `NSYS_BIN` to the collector executable or `PROFILE_PYTHON` to another suitable environment if needed. This capture used the Linux collector bundled with the local Nsight Systems 2026.2.1 app; the copied tool directory was removed from the pod during cleanup. To repeat that setup, copy the app's `Contents/target-linux-x64` directory to the pod and point `NSYS_BIN` at its `nsys` executable. The script overwrites the named report; save it under another name before rerunning if preserving this particular capture matters.

To retry Nsight Compute on a host with counter access:

```sh
/usr/local/cuda/bin/ncu --set full --profile-from-start off \
  --clock-control none --force-overwrite \
  --export benchmarks/target_layer0_decode_ctx512 \
  .venv/bin/python -m benchmarks.profile_decode_layer \
  --iterations 1 --metadata benchmarks/ncu_metadata.json
```

Compute's default cache flushing and kernel replay affect timings; use its metrics to diagnose kernels, not as a substitute for the Systems timeline. `--clock-control none` avoids changing GPU clock settings.

[nsys_metadata.json](nsys_metadata.json) records the exact input IDs, workload, memory allocation, environment, and hashes of the profiling script, executor, CUDA sources, and compiled target extensions. [profile_decode_layer.py](profile_decode_layer.py) is the reusable capture harness. Production runtime files were not modified for this profile.
