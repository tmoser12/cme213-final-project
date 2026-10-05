# Attention optimization results

Both changes are retained in the target and draft production attention extensions. Measurements below compare the saved original binaries with rebuilt optimized binaries on the same RTX 4090.

## Changes

1. Pad the scalar decode/verification shared K tile from `[KV_TILE][D]` to `[KV_TILE][D + 2]`. The added two FP16 elements make the row stride odd in 32-bit bank words, removing the bank-address aliasing of column reads. Applied to both ordinary and split attention. Arithmetic and reduction order are unchanged.
2. Give each RoPE/KV-write block at most 16 sequence rows. At batch one and 1,742 target prompt tokens, this increases the launch from 4 blocks to 436. Short sequences still use one block per KV head. Host and device cache positions, partial tiles, and empty writes remain supported.

The WMMA prefill attention implementation is unchanged. The operation-level comparisons isolate the changes: RoPE measurements exercise sequence tiling; scalar attention measurements exercise shared-memory padding. No hardware bank-conflict counters were collected, so the timing improvement is measured while the bank-conflict explanation comes from inspecting the access pattern.

## Kernel results

Batch one; microseconds per operation. Each value is the median of three round medians. Warm timing reuses resident inputs; cold timing evicts cache before the operation. Speedup is original time divided by optimized time.

| Role / operation | Query / context | Warm original → optimized | Speedup | Cold original → optimized | Speedup |
|---|---:|---:|---:|---:|---:|
| target rope | 1 / 1 | 1.11 → 1.13 | 0.98× | 3.07 → 3.07 | 1.00× |
| target rope | 9 / 9 | 1.54 → 1.52 | 1.01× | 4.10 → 4.10 | 1.00× |
| target rope | 128 / 128 | 7.82 → 1.80 | 4.35× | 15.36 → 5.12 | 3.00× |
| target rope | 1742 / 1742 | 86.53 → 3.24 | 26.74× | 168.10 → 16.38 | 10.26× |
| target decode | 1 / 128 | 12.04 → 9.91 | 1.21× | 20.77 → 18.94 | 1.10× |
| target verify | 9 / 128 | 29.63 → 17.71 | 1.67× | 39.65 → 27.65 | 1.43× |
| target decode | 1 / 512 | 24.00 → 19.90 | 1.21× | 40.96 → 36.86 | 1.11× |
| target verify | 9 / 512 | 63.18 → 39.59 | 1.60× | 80.26 → 57.34 | 1.40× |
| target decode | 1 / 1742 | 31.64 → 27.61 | 1.15× | 56.24 → 50.69 | 1.11× |
| target verify | 9 / 1742 | 118.38 → 70.11 | 1.69× | 141.31 → 92.16 | 1.53× |
| draft rope | 1 / 1 | 1.13 → 1.13 | 1.00× | 3.07 → 3.07 | 1.00× |
| draft rope | 9 / 9 | 1.20 → 1.21 | 0.99× | 3.07 → 3.12 | 0.98× |
| draft rope | 128 / 128 | 4.10 → 1.27 | 3.23× | 9.22 → 4.10 | 2.25× |
| draft rope | 1742 / 1742 | 45.09 → 1.62 | 27.87× | 87.04 → 7.17 | 12.14× |
| draft decode | 1 / 128 | 4.34 → 4.08 | 1.07× | 7.17 → 7.17 | 1.00× |
| draft decode | 1 / 512 | 6.00 → 5.47 | 1.10× | 10.24 → 10.24 | 1.00× |
| draft decode | 1 / 1742 | 13.60 → 11.71 | 1.16× | 23.55 → 21.50 | 1.10× |

All 63 cases, including batches 1, 2, and 4, are in [kernel_summary.csv](kernel_summary.csv). Raw event samples and source/binary hashes are in [ab.json](ab.json). Tiny short-sequence differences should not be interpreted as reliable wins; event timing is coarse at a few microseconds.

## Full target-only model

Same loaded Qwen2.5-7B-Instruct weights and executor, swapping only the attention extension. Eager execution, batch one, 128 generated tokens, two warmups per variant and prompt, five paired repetitions with randomized variant order. Decode throughput covers the 127 steps after the first token. TTFT includes prefill and the first token. Values below are medians.

| Prompt tokens | Original tokens/s | Optimized tokens/s | Throughput change | TTFT original → optimized (ms) |
|---:|---:|---:|---:|---:|
| 128 | 62.24 | 62.63 | +0.62% | 20.72 → 20.49 |
| 512 | 61.94 | 62.30 | +0.58% | 49.62 → 48.75 |
| 1742 | 59.48 | 60.00 | +0.87% | 181.42 → 177.84 |

Generated token IDs matched for every original/optimized pair. The end-to-end gains are much smaller than the isolated kernel gains because these operations occupy only part of each model step. Nine-token verification is relevant to speculative decoding; its speedup is not a measurement of complete speculative generation. Full-model batching and optimized graph-mode generation were not benchmarked here. The historical vLLM comparison remains under `benchmarking/results/`; it is not a fresh concurrent comparison against these modified kernels.

## Validation and measurement limits

- 63 paired operation/shape comparisons required bitwise-identical outputs before timing.
- 45 runtime tests and 16 demo tests passed on the GPU; see [runtime-tests.log](runtime-tests.log) and [demo-tests.log](demo-tests.log).
- Added tests cover both model dimensions, batches 1 and 2, empty/partial/exact RoPE tiles, untouched cache regions, nonzero positions, graph replay with changing device positions, query sizes 1–9, and contexts around the split boundary.
- CUDA events are graph nodes around the operation, excluding Python launch overhead. Each of three paired rounds uses 20 samples; warm samples average 50 calls, cold samples time one call following a 128 MiB eviction outside the timed region.
- This is one GPU and one session, not a confidence interval across machines. Cold-cache eviction is an approximation of a large working set, not a simulation of the entire inference memory hierarchy.

## Saved originals and rollback

`baseline/` contains the original target/draft attention CUDA sources, bindings, wrappers, associated documentation, and original compiled extensions. SHA-256 manifests protect the source and binary snapshots. Copies are saved both locally and on the GPU. The binary backups are ignored by Git and should be retained alongside the source snapshots if moving this directory.

Preview without changes:

```sh
python3 benchmarking/optimizations/attention_v1/rollback.py
```

Restore and rebuild on the GPU, from the repository root:

```sh
.venv/bin/python benchmarking/optimizations/attention_v1/rollback.py --apply --rebuild
```

The restore first saves the displaced optimized sources and binaries to a timestamped directory. Restart any model process after rebuilding. The restore was exercised in a disposable copy, including exact comparison of restored originals and preserved displaced files; production remains optimized.

## Reproduce

```sh
PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 BUILD_KERNEL=attention .venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m benchmarking.attention_ab
.venv/bin/python -m benchmarking.attention_model_ab
.venv/bin/python -m benchmarking.attention_report
```

Run GPU measurements sequentially on an idle device. The original binaries require the saved Python/PyTorch/CUDA ABI; source backups are the portable rollback path.
