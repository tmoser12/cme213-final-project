# Measured results — RTX 4090, October 5, 2026

**Historical baseline:** These measurements precede the attention optimizations. See [the paired before/after report](../optimizations/attention_v1/REPORT.md) for the active kernels.

117 operation/shape cases passed numerical checks. These results describe the original production kernels before the saved attention optimizations.

Measured 128 MiB copy bandwidth: **921.5 GB/s**, versus 1008 GB/s specified peak. GPU: NVIDIA GeForce RTX 4090, 128 SMs; torch 2.8.0+cu128.

## Target-only engine comparison

Batch 1, FP16, Qwen2.5-7B-Instruct, 128 generated tokens, EOS ignored, identical prompt IDs, context capacity 2048. Two warmups per prompt length and five measured requests. TPOT = time from first to last output / 127; tokens/s = 127 / that interval. Tokenization, loading, compilation, HTTP, browser rendering, and detokenization are excluded. CPU scheduling, input copies, greedy selection, and GPU synchronization needed to return IDs are included. TTFT includes prefill and cache reset in the custom path. The draft model is not loaded.

| Engine | Prompt | Decode tokens/s (median; min–max) | TPOT ms | TTFT ms |
|---|---:|---:|---:|---:|
| custom | 128 | 62.24 (62.18–62.25) | 16.068 | 20.7 |
| custom | 512 | 61.94 (61.92–61.97) | 16.144 | 49.7 |
| custom | 1742 | 59.50 (59.49–59.52) | 16.806 | 180.5 |
| vllm | 128 | 62.80 (62.76–62.85) | 15.923 | 20.7 |
| vllm | 512 | 62.76 (62.72–62.78) | 15.934 | 46.6 |
| vllm | 1742 | 62.36 (62.27–62.43) | 16.035 | 158.5 |
| custom-graph | 128 | 64.27 (64.25–64.27) | 15.559 | 20.8 |
| custom-graph | 512 | 63.60 (63.58–63.63) | 15.723 | 50.0 |
| custom-graph | 1742 | 60.30 (60.30–60.31) | 16.583 | 180.4 |

The main comparison is **custom** (the demo’s eager target path) versus **vllm** (vLLM V1 with its default compilation/graph optimizations). `custom-graph` and `vllm-eager`, if present, are diagnostic configurations, not the demo default.

vLLM version: 0.10.2. Both environments use torch 2.8.0+cu128; model revision is `a09a35458c702b33eeacc393d103063234e8bc28`.

| Prompt | vLLM / custom decode throughput | Matching output prefix | Equal token IDs |
|---|---:|---:|---|
| 128 | 1.01× | 128/128 | True (all repeats) |
| 512 | 1.01× | 128/128 | True (all repeats) |
| 1742 | 1.05× | 128/128 | True (all repeats) |

Token agreement is checked, not assumed. FP16 kernel and reduction differences can change an argmax; divergent sequences still have equal lengths and computational shapes, but they are not evidence of exact model parity. All repeats and complete generated IDs are retained in JSON.

Existing target graph mode also matches the eager target tokens across all measured requests: **True**. It is an optional diagnostic, not a change to the demo default.

## Whole-target bandwidth reference

The target reads approximately **14.141 GB of FP16 weights per decode step**: packed QKV, attention output, three MLP projections, normalization scales, LM head, and one embedding row. At 1008 GB/s, streaming those weights alone takes **14.03 ms**, or **71.28 tokens/s**. This excludes KV traffic, activations, scratch, compute, and CPU/device scheduling. It assumes weights are streamed once; cache residency can change actual DRAM bytes. It is a model-level bandwidth reference, not a measured counter or a promise of attainable latency.

- Custom at context 128: 62.24 tokens/s, or 87.3% of this weight-streaming throughput reference.
- Custom at context 512: 61.94 tokens/s, or 86.9% of this weight-streaming throughput reference.
- Custom at context 1742: 59.50 tokens/s, or 83.5% of this weight-streaming throughput reference.

## Representative custom-kernel measurements

Times below are CUDA graph event medians with Python dispatch excluded. Cold means a 128 MiB write precedes the operation, outside the timed region. Warm replays reuse inputs and may be served by L2. “Roofline %” is an optimistic logical-DRAM lower bound divided by cold measured time; it is not a measured hardware utilization counter. Values near or above 100% require cache/traffic analysis, not a claim of exceeding peak hardware throughput.

| Role | Operation | Query / context | Warm µs | Cold µs | DRAM lower bound µs | Roofline % |
|---|---|---:|---:|---:|---:|---:|
| target | embedding | 1 / 1 | 3.891 | 10.224 | 0.0142 | 0.1% |
| target | rmsnorm | 1 / 1 | 1.469 | 3.840 | 0.0213 | 0.6% |
| target | residual | 1 / 1 | 0.893 | 3.248 | 0.0213 | 0.7% |
| target | swiglu_activation | 1 / 1 | 1.010 | 3.584 | 0.1128 | 3.1% |
| target | rope_kv_write | 1 / 1 | 1.053 | 3.488 | 0.0043 | 0.1% |
| target | embedding | 1742 / 1742 | 10.898 | 48.528 | 24.7889 | 51.1% |
| target | rmsnorm | 1742 / 1742 | 8.120 | 31.104 | 24.7822 | 79.7% |
| target | residual | 1742 / 1742 | 8.327 | 47.712 | 37.1627 | 77.9% |
| target | swiglu_activation | 1742 / 1742 | 212.462 | 221.488 | 196.4312 | 88.7% |
| target | rope_kv_write | 1742 / 1742 | 86.236 | 167.600 | 7.5210 | 4.5% |
| target | decode_attention | 1 / 1742 | 31.577 | 57.152 | 3.5538 | 6.2% |
| target | verify_attention | 9 / 1742 | 118.274 | 141.264 | 3.6696 | 2.6% |
| target | prefill_attention | 1742 / 1742 | 541.035 | 562.832 | 131.8582 | 23.4% |
| draft | embedding | 1 / 1 | 1.589 | 4.864 | 0.0036 | 0.1% |
| draft | rmsnorm | 1 / 1 | 1.279 | 3.856 | 0.0053 | 0.1% |
| draft | residual | 1 / 1 | 0.816 | 2.992 | 0.0053 | 0.2% |
| draft | swiglu_activation | 1 / 1 | 0.980 | 3.264 | 0.0290 | 0.9% |
| draft | rope_kv_write_dev | 1 / 1 | 1.118 | 3.808 | 0.0011 | 0.0% |
| draft | embedding | 1742 / 1742 | 3.513 | 14.224 | 6.2076 | 43.6% |
| draft | rmsnorm | 1742 / 1742 | 3.385 | 10.832 | 6.1956 | 57.2% |
| draft | residual | 1742 / 1742 | 3.004 | 14.608 | 9.2907 | 63.6% |
| draft | swiglu_activation | 1742 / 1742 | 11.391 | 61.792 | 50.4350 | 81.6% |
| draft | rope_kv_write | 1742 / 1742 | 46.156 | 87.904 | 1.9909 | 2.3% |
| draft | decode_attention_dev | 1 / 1742 | 14.587 | 25.408 | 0.8885 | 3.5% |
| draft | prefill_attention | 1742 / 1742 | 200.168 | 196.032 | 32.9645 | 16.8% |

## Claims the data supports

- Cite measured latency at an explicit role, query length, and context. Avoid calling every kernel “near roofline.” The full table includes unfavorable cases.
- Small kernels can be limited by launch/device scheduling, insufficient blocks, reductions, and special functions well before the full-chip memory roofline. The target RoPE/cache-write kernel launches only four blocks; draft launches two, versus 128 SMs on this GPU.
- The split-KV heuristic targets 144 blocks based on an older 72-SM assumption. It is unchanged here. Split and combine are recorded individually in `individual_kernels.csv`, including their FP32 scratch traffic.
- Prefill attention uses FP16 tensor-core products with FP32 accumulation; decode and verification use scalar arithmetic. Comparing scalar decode to advertised sparse tensor TOPS would be invalid.
- Fusing RoPE with the cache write and attention avoids separate passes, and vectorized elementwise kernels reduce instruction count. This experiment measures their latency; it does not isolate the speedup of fusion versus an unfused implementation.
- End-to-end engine performance also depends on cuBLAS projections, launch orchestration, cache layout, and sampling. Fast individual custom kernels do not establish superiority to vLLM.

## Artifacts

- `operations.csv`: every tested wrapper/shape, warm/cold latency, numerical result, logical bytes/FLOPs and roofline.
- `individual_kernels.csv`: every observed custom CUDA kernel, including split and combine, with profiler timing and per-kernel cost model.
- `kernels.json`: all event samples, profiler duration samples, hardware metadata and runtime source hashes.
- Engine JSON files: every request, each decode step, all output IDs, exact prompt hash and environment.
- See `../README.md` for cost equations, scope, limitations and reproduction commands.
