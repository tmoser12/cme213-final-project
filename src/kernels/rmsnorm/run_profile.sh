#!/bin/bash
#
# Capture an Nsight Compute (.ncu-rep) report for the custom RMSNorm kernel.
# Designed to be scp'd back to a workstation and opened in the Nsight Compute GUI.
#
# Usage:
#   bash src/kernels/rmsnorm/run_profile.sh
#
# Output:
#   <project_root>/results/profiles/rmsnorm_<timestamp>.ncu-rep

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
_KERNEL_PARENT="$(cd "$(dirname "$_SCRIPT")" && pwd)"
cd "$_KERNEL_PARENT/../../.."

# Activates the cme213 conda env, loads gnu12, exports PROJECT_ROOT, etc.
source setup.sh

KERNEL_DIR=$(basename "$_KERNEL_PARENT")            # "rmsnorm"
KERNEL_SYMBOL="rmsnorm_forward_kernel_vectorized"   # matches kernel.cu
OUT_DIR="$PROJECT_ROOT/results/profiles"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$OUT_DIR/${KERNEL_DIR}_${STAMP}"          # ncu appends .ncu-rep

echo "=========================================================="
echo "    Nsight Compute profile: $KERNEL_DIR"
echo "    Kernel symbol: $KERNEL_SYMBOL"
echo "    Output:        ${OUT_BASE}.ncu-rep"
echo "=========================================================="

# ncu creates /tmp/nsight-compute-lock by default; on the shared compute nodes
# that file already exists and isn't writable by us. Redirect ncu's tempdir to
# a per-user location. (`srun` forwards exported env vars by default.)
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Force JIT recompile so the captured kernel matches the current source.
echo "[1/2] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

# --set full              : capture every metric section (slowest, richest report)
# --kernel-name regex:... : only profile our kernel (skip any incidental CUDA work)
# --launch-skip 5         : skip the 5 warmup launches in profile_main()
# --launch-count 1        : profile exactly one launch (we only need one shape)
# --target-processes all  : be safe if torch spawns workers
# --import-source on      : embed kernel.cu in the report so source view works after scp
# -f                      : overwrite existing report
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
        python -m src.kernels.${KERNEL_DIR}.benchmark --profile

echo ""
echo "✅ Done. Report at: ${OUT_BASE}.ncu-rep"
echo "   scp it to your Mac, e.g.:"
echo "     scp <cluster>:${OUT_BASE}.ncu-rep ~/Downloads/"
