#!/bin/bash
#
# Unified benchmark + profile driver for the rmsnorm extension.
#
# Usage:
#   bash run_benchmark.sh             # benchmark.py, no profile
#   bash run_benchmark.sh --profile   # benchmark.py + nsys timeline + ncu metrics
#
# Profile outputs (timestamped, never overwrite):
#   <project_root>/results/profiles/rmsnorm_<stamp>.{nsys-rep,ncu-rep}

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
_KERNEL_PARENT="$(cd "$(dirname "$_SCRIPT")" && pwd)"
cd "$_KERNEL_PARENT/../../../.."
source setup.sh

KERNEL_DIR=$(basename "$_KERNEL_PARENT")            # "rmsnorm"
KERNEL_REGEX="rmsnorm_forward_kernel_vectorized"    # matches kernel.cu
MODULE="src.target.kernels.${KERNEL_DIR}.benchmark"

PROFILE=false
for arg in "$@"; do
    case "$arg" in
        --profile) PROFILE=true ;;
        *)         echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "=========================================================="
echo "    $KERNEL_DIR   (profile=$PROFILE)"
echo "=========================================================="
echo "[setup] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

if [ "$PROFILE" = false ]; then
    echo "[run] srun python -m $MODULE"
    srun --partition=gpu-turing --gres=gpu:1 python -m "$MODULE"
    exit 0
fi

# ---- Profile path ----------------------------------------------------------
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
OUT_DIR="$PROJECT_ROOT/results/profiles"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$OUT_DIR/${KERNEL_DIR}_${STAMP}"

# nsys: full timeline. profile_main brackets with cudaProfilerStart/Stop, so
# warmup launches stay out of the capture.
echo ""
echo "[nsys] Capturing full timeline..."
srun --partition=gpu-turing --gres=gpu:1 \
    nsys profile \
        --trace=cuda,nvtx,cublas,osrt \
        --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        -o "$OUT_BASE" \
        python -m "$MODULE" --profile
echo "  Wrote: ${OUT_BASE}.nsys-rep"

# ncu: per-kernel metrics. profile_main uses cudaProfilerStart/Stop, so the
# bracket scopes ncu via --profile-from-start off; 1 launch inside.
echo ""
echo "[ncu] Capturing per-kernel metrics (regex: $KERNEL_REGEX)..."
srun --partition=gpu-turing --gres=gpu:1 \
    ncu \
        --set full \
        --kernel-name "regex:${KERNEL_REGEX}" \
        --profile-from-start off \
        --launch-count 1 \
        --target-processes all \
        --import-source on \
        -f -o "$OUT_BASE" \
        python -m "$MODULE" --profile
echo "  Wrote: ${OUT_BASE}.ncu-rep"

echo ""
echo "Done. scp results to a workstation, e.g.:"
echo "  scp <cluster>:${OUT_BASE}.* ~/Downloads/"
