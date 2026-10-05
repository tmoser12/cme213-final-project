# Interpreting the decode-layer profile

These notes describe `target_layer0_decode_ctx512.nsys-rep`: target layer 0, one query, batch one, 512 cached prefix tokens, RTX 4090. They summarize the discussion after inspecting the native report and its exported SQLite data.

## The screenshot's gaps are CPU submission gaps

Iteration 9's CPU `attention_block` range spans **26.513230–26.568994 ms**. Its QKV wrapper ends at **26.526575 ms**, while its RoPE wrapper begins at **26.540681 ms**. That roughly 14 µs interval contains host tensor/view preparation and dispatch overhead; it is not a 14 µs GPU stall.

CUDA correlation IDs associate that iteration with GPU work much later in the timeline:

| Operation | GPU start (ms) | GPU duration (µs) | Gap after previous kernel (µs) |
|---|---:|---:|---:|
| QKV projection | 30.270461 | 37.664 | — |
| RoPE/cache write | 30.308829 | 1.280 | 0.704 |
| Split attention | 30.310717 | 15.264 | 0.608 |
| Combine | 30.326589 | 2.080 | 0.608 |
| Output projection | 30.329373 | 29.217 | 0.704 |

CPU NVTX ranges bracket submission scopes. CUDA API rows show host driver/runtime calls; GPU stream rows show actual execution. GPU-projected NVTX ranges are a separate view associating device work with its originating CPU scope. Use correlation rather than assuming events vertically aligned at the same timestamp belong to the same iteration.

The harness queues ten repetitions without intervening synchronization, allowing the CPU to run ahead. This hides submission overhead behind preceding GPU work and differs from serial whole-model generation with token selection between steps. At batch one and sequence length one, Q/K/V layout views require no layout-copy kernel in this trace.

## Concrete attention weaknesses worth testing

- **Serial score and softmax work:** for one query and a 32-key tile, only 32 of the 128 threads compute scores, each looping over 128 channels. One thread then computes the tile maximum, exponentials, and denominator. Warp-cooperative dots and parallel reductions are a candidate improvement; the previous shared-memory padding did not change this decomposition.
- **Inherited split heuristic:** this shape launches 3 splits × 28 heads = 84 blocks on 128 SMs. The heuristic targets the original 72-SM GPU. More splits may improve parallelism but increase partial-result and combine overhead; benchmark a sweep.
- **Repeated GQA loads:** seven query heads share each KV head, but separate blocks reload the same KV data. Sharing that work within blocks could reduce redundant loads, at the cost of more per-block state. Repeated block loads do not establish repeated DRAM reads because L2 can serve them.
- **Bias initialization:** QKV projection copies the 9 KiB bias into its output before calling cuBLAS with beta=1. The trace contains ten such device-to-device copies, averaging about 1 µs each. A supported cuBLASLt bias epilogue is a smaller experiment than replacing the projection implementation.

Across this capture, projections account for about **95% of GPU kernel time**; RoPE, split attention, and combine together account for about **3.6%**. These are isolated-layer proportions, not universal proportions for prefill, longer contexts, or speculative verification.

## QKV materialization and fusion

The current packed projection materializes QKV in global memory. A separate kernel reads K/V, rotates K, and writes the persistent cache. Stock cuBLAS does not expose an arbitrary RoPE/cache-scatter epilogue. Concatenating weights is not itself the restriction: a custom projection could retain packed weights and implement the paired rotation and cache stores, with substantial implementation and tuning work.

An intermediate global-memory store/load does not necessarily mean a physical DRAM round trip. The packed decode output is only 9 KiB and may be served from L2. This GPU uses GDDR6X, not HBM. Hardware counter access was denied, so the trace does not establish actual DRAM traffic or bank-conflict counts.

References: [Nsight Systems analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), [cuBLAS 12.8 epilogues](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasltepilogue-t), [production executor](../runtime/executor.py), [target attention](../runtime/production_kernels/target/attention/kernel.cu).
