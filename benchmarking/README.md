# Kernel rooflines and target-only decoding

**Version note:** `results/` preserves the original pre-optimization measurements. The active attention kernels now include K-tile padding and tiled RoPE writes; see [the paired optimization report](optimizations/attention_v1/REPORT.md) and its saved originals/rollback instructions.

This directory measures the production demo in `runtime/`, on the migrated RTX 4090. Start with [the measured report](results/REPORT.md), [kernel latency chart](results/kernel_bounds.png), [roofline chart](results/roofline.png), and [engine chart](results/engine_decode.png). Raw measurements and source hashes are retained beside them. No inference kernel or demo behavior is changed by these scripts.

## What is covered

Both roles use embedding, RMSNorm, residual addition, fused RoPE/key/value cache writes, SwiGLU activation, and prefill attention. Target decode/verification exercises the unsplit attention kernel and split-KV plus combine kernels. Draft graph decode exercises device-scalar position/length inputs, whose split geometry is sized from the 2048-token capacity. Actual CUDA kernel names are extracted from a GPU trace, including separate split and combine durations.

Shapes use the real configurations: target H=3584, I=18944, 28 query heads, 4 KV heads, D=128; draft H=896, I=4864, 14 query heads, 2 KV heads, D=64. Batch is one. Elementwise operations cover S=1,2,4,5,8,9,128,512,1742. Attention contexts are 128,512,1742; target verification covers S=2,4,5,8,9, including a prepended bonus. These are representative shapes, not an exhaustive context sweep.

The MLP’s three matrix projections, QKV projection, output projection, and LM head are cuBLAS calls, not hand-written CUDA kernels. PyTorch’s bias, transpose/copy, position updates, and argmax operations are also not custom kernels. They are included in engine timing, but do not appear as custom-kernel results. The existing standalone `swiglu_act_forward` launches the same activation kernel used inside the MLP; the benchmark output is separate whereas production overwrites the gate buffer. Both have two input reads and one output write. There is no speculative accept/reject CUDA kernel in this demo branch.

## Meaning of the roofline

For useful arithmetic F, optimistic logical traffic B, bandwidth BW, and appropriate arithmetic peak P:

```
arithmetic intensity = F / B
ideal time = max(B / BW, F / P)
roofline fraction = ideal time / measured time
```

Use decimal GB/s and TFLOP/s. RTX 4090 peaks are 1008 GB/s DRAM, 82.6 TFLOP/s scalar FP32, and **165.2 TFLOP/s dense FP16 tensor operations with FP32 accumulation**. The last rate applies only to prefill WMMA. Sparse, FP8, and FP16-accumulate headline rates do not apply. Source: [NVIDIA Ada architecture whitepaper, Appendix A, pp. 29–31](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf). A 128 MiB device-copy microbenchmark supplies a measured bandwidth reference as well.

Let S be query rows, T valid context, H hidden width, I intermediate width, A query heads, K KV heads, D head width, and N splits. FP16 values occupy two bytes; scratch values are FP32. Cost models are deliberately **optimistic compulsory-traffic bounds**, not hardware-counter measurements:

| Operation | Logical bytes | Useful FLOPs |
|---|---:|---:|
| Embedding | 4SH + 8S | 0 (gather/copy) |
| RMSNorm | 4SH + 2H | 4SH + 2S |
| Residual | 6SH | SH |
| SwiGLU activation | 6SI | 4SI |
| RoPE/KV write | 8KSD + 2SD | 3KSD |
| Attention, complete operation | 4ASD + 4KTD + 2SD | 4AD·[ST − S(S−1)/2] + 3ASD |
| Split attention | Attention bytes − 2ASD + 4ANS(D+2) | Attention FLOPs + 3ASD(N−1) |
| Combine | 4ANS(D+2) + 2ASD | AS·[3ND + 4N + D] |

The RoPE tables have duplicated halves; these kernels read only half of each table, hence 2SD bytes for cos+sin. Attention counts useful causal QK and PV work, with multiply-add counted as two FLOPs. It excludes masked/padded tensor-core work, softmax exponentials, max comparisons, address arithmetic, synchronization, and conversion costs. SwiGLU counts arithmetic around its exponential; RMSNorm excludes reduction overhead and reciprocal-square-root throughput. Combine FLOPs are an approximate arithmetic count around the exponential/max reductions. Special functions have their own throughput limits, so these arithmetic bounds cannot predict optimal latency by themselves.

RMSNorm logically reuses its input and weights; attention logically reuses KV across grouped query heads. Actual instructions reload data and tiled prefill reads KV repeatedly. Cache reuse can prevent those loads from becoming DRAM transactions. The operation-level attention bound assumes no externally visible scratch; the per-kernel split/combine table includes scratch, which will ordinarily stay in cache. In particular, a DRAM ratio for combine is a reference only: its inputs were just produced by split attention. A rigorous hardware roofline would additionally measure DRAM/L2 bytes, instruction counts, achieved clocks and occupancy with Nsight Compute; no such counters are claimed here.

## Measurement protocol

- Deterministic random FP16 inputs; embedding, normalization, activation, cache writes, and attention are checked against PyTorch references before timing. Attention uses a nonzero RoPE angle, causal mask, and GQA expansion. Prefill tolerance is explicitly saved in the script; finite-precision fused attention need not be bitwise equal.
- Warm up each operation. Capture 50 invocations in one CUDA graph for warm timing. Use CUDA events *inside* the graph to exclude Python launch overhead; retain event objects throughout trace replay. Report 20 event samples with median and p10/p90.
- Cold timing captures a 128 MiB device write, a start event, one invocation, and an end event. The flush is outside the interval and exceeds the GPU’s 72 MiB L2. This is a controlled cold-input proxy, not proof that every accessed line came from DRAM. Scratch produced within an operation stays warm.
- Trace five replays separately for actual CUDA kernel durations. Profiler timing is labeled separately from unprofiled event timing and can include instrumentation effects. Graph timing still includes device node scheduling and gaps between kernels.
- Run GPU workloads sequentially. Record GPU, driver, CUDA, PyTorch, SM count, memory, and SHA-256 of runtime Python/CUDA sources. Clocks and power limits are not locked. Raw sample spread describes these repeated runs, not a confidence interval over hosts or days.

## Engine comparison

The primary comparison is the **existing eager target-only executor** against **vLLM 0.10.2 V1**, with its default graph/compilation optimizations. This version matches the demo’s PyTorch 2.8.0/CUDA 12.8 stack; it is a pinned baseline, not a claim about the newest vLLM release. Both engines load the same local Qwen2.5-7B-Instruct checkpoint (revision `a09a35458c702b33eeacc393d103063234e8bc28`), with FP16 weights/KV cache, tensor parallel size one, no quantization or speculation, and identical saved token IDs.

Each prompt has 128, 512, or 1742 tokens, with 128 generated tokens and EOS ignored so generation lengths match. Prompts are deterministic text prefixes for controlled length, not a representative quality dataset or a chat-template benchmark. Two complete warmups precede five measured requests per shape. Prefix caching is disabled. vLLM V1 forces its chunked-prefill scheduler on, with an 8192-token per-step budget; every prompt here fits in one prefill step, so none is split. This behavior was checked against the installed scheduler source and startup log. There is no concurrent batching. The 2048-token capacity matches the demo; vLLM’s physical cache reservation and custom buffer allocations differ.

The first output comes from prefill. Decode throughput therefore uses **127 steps**, from the first output to the 128th output. TPOT = decode interval / 127. Every engine emission is checked to contain exactly one newly generated token. CPU scheduling, greedy selection, and synchronization to return token IDs are included; HTTP, tokenizer, detokenization, UI streaming, model loading and compilation are excluded. This is an in-process engine comparison, not GPU-forward-only latency or a concurrent serving benchmark. The custom path follows the demo’s CPU argmax/item and per-token tensor allocation. It loads only the target; the live demo normally keeps the draft resident too.

Additional `custom-graph` results isolate the existing executor’s graph mode. The script also supports an optional `vllm-eager` diagnostic. The former uses the existing executor graph support without changing the demo’s default. See [vLLM’s versioned LLM engine API](https://docs.vllm.ai/en/v0.10.2/api/vllm/engine/llm_engine.html) and the saved startup log for the actual selected V1 backend and graph settings. Per-request full token IDs expose any numerical divergence.

## Reproduce on the pod

From `/workspace/cme213-demo`, with extensions already built by `demo/setup.sh`, and no model server or other GPU workload running:

```bash
.venv/bin/python -m benchmarking.kernels
.venv/bin/python -m benchmarking.decode --engine custom --output benchmarking/results/custom.json
/workspace/cme213-benchmark-vllm/bin/python -m benchmarking.decode --engine vllm --output benchmarking/results/vllm.json
.venv/bin/python -m benchmarking.decode --engine custom-graph --output benchmarking/results/custom-graph.json
.venv/bin/python -m benchmarking.report
```

Create vLLM’s separate environment if needed (do not install it into the demo environment):

```bash
uv venv --system-site-packages /workspace/cme213-benchmark-vllm --python /usr/bin/python3
uv pip install --python /workspace/cme213-benchmark-vllm/bin/python 'vllm==0.10.2' 'transformers==4.55.2'
```

`results/vllm-freeze.txt` records the resolved environment for this run. Plot generation additionally needs matplotlib. The scripts refuse to apply 4090 roofline constants to another GPU; update the documented peaks first. Preserve previous results before rerunning because output paths are overwritten.

## Verify saved evidence

Run `python3 -m benchmarking.audit --require-engines --source-snapshot benchmarking/optimizations/attention_v1/baseline` from the repository root. This uses only the Python standard library and checks runtime hashes, custom-kernel coverage, numerical pass flags, sample statistics, token counts, shared prompt hashes, and throughput arithmetic.
