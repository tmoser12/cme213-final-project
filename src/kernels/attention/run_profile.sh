#!/bin/bash
#
# Capture an Nsight Compute (.ncu-rep) report for one of the attention sub-op kernels.
# Designed to be scp'd back to a workstation and opened in the Nsight Compute GUI.
#
# Usage:
#   bash src/kernels/attention/run_profile.sh [KERNEL_SYMBOL_REGEX] [TARGET]
#
# Defaults:
#   KERNEL_SYMBOL_REGEX = fused_attn_kernel   (the headline kernel)
#   TARGET              = benchmark_fused_attn
#
# Examples:
#   bash run_profile.sh rope_kernel        benchmark_rope
#   bash run_profile.sh kv_write_kernel    benchmark_kv_write
#   bash run_profile.sh fused_attn_kernel  benchmark_fused_attn
#
# Output:
#   <project_root>/results/profiles/attention_<TARGET>_<timestamp>.ncu-rep

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
_KERNEL_PARENT="$(cd "$(dirname "$_SCRIPT")" && pwd)"
cd "$_KERNEL_PARENT/../../.."

source setup.sh

KERNEL_DIR=$(basename "$_KERNEL_PARENT")        # "attention"
KERNEL_SYMBOL="${1:-fused_attn_kernel}"          # which __global__ symbol to capture
TARGET="${2:-benchmark_fused_attn}"              # which sub-benchmark to run
OUT_DIR="$PROJECT_ROOT/results/profiles"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$OUT_DIR/${KERNEL_DIR}_${TARGET}_${STAMP}"

# Dispatch: umbrella "benchmark" lives in the kernel dir; per-sub-op
# benchmarks live in `benchmark_scripts/`.
if [ "$TARGET" = "benchmark" ]; then
    MODULE="src.kernels.${KERNEL_DIR}.${TARGET}"
else
    MODULE="src.kernels.${KERNEL_DIR}.benchmark_scripts.${TARGET}"
fi

echo "=========================================================="
echo "    Nsight Compute profile: $KERNEL_DIR"
echo "    Kernel symbol: $KERNEL_SYMBOL"
echo "    Target:        $TARGET"
echo "    Output:        ${OUT_BASE}.ncu-rep"
echo "=========================================================="

# ncu creates /tmp/nsight-compute-lock by default; on the shared compute nodes
# that file already exists and isn't writable by us. Redirect ncu's tempdir to
# a per-user location. (`srun` forwards exported env vars by default.)
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

echo "[1/2] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

echo "[2/2] Submitting to Slurm (gpu-turing) under ncu..."
srun --partition=gpu-turing --gres=gpu:1 \
    ncu \
        --set full \
        --kernel-name "regex:${KERNEL_SYMBOL}" \
        --launch-skip 5 \
        --launch-count 1 \
        --target-processes all \
        --import-source on \
        -f -o "$OUT_BASE" \
        python -m "$MODULE" --profile

echo ""
echo "✅ Done. Report at: ${OUT_BASE}.ncu-rep"
echo "   scp it to your Mac, e.g.:"
echo "     scp <cluster>:${OUT_BASE}.ncu-rep ~/Downloads/"
