#!/usr/bin/env bash
# Run on the GPU from any directory. Loading and warmup are outside capture.
set -euo pipefail
cd "$(dirname "$0")/.."
profile_python="${PROFILE_PYTHON:-.venv/bin/python}"
profile_nsys="${NSYS_BIN:-nsys}"
"$profile_nsys" profile \
  --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output=benchmarks/target_layer0_decode_ctx512 \
  "$profile_python" -m benchmarks.profile_decode_layer \
  --context 512 --iterations 10 --metadata benchmarks/nsys_metadata.json
"$profile_nsys" stats \
  --force-export=true --force-overwrite=true \
  --report cuda_gpu_kern_sum,nvtx_pushpop_sum,cuda_api_sum \
  --format csv --output benchmarks/target_layer0_decode_ctx512 \
  benchmarks/target_layer0_decode_ctx512.nsys-rep
