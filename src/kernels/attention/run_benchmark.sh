#!/bin/bash
#
# Unified benchmark + profile driver for the attention extension.
#
# Usage:
#   bash run_benchmark.sh                           # umbrella benchmark.py, no profile
#   bash run_benchmark.sh --profile                 # umbrella + nsys timeline + ncu metrics
#   bash run_benchmark.sh <sub-benchmark>           # one sub-op benchmark, no profile
#   bash run_benchmark.sh <sub-benchmark> --profile # one sub-op + ncu metrics
#
# Sub-benchmark names (files under benchmark_scripts/):
#   benchmark_qkv_proj      benchmark_o_proj       (cuBLAS — ncu skipped)
#   benchmark_rope          benchmark_kv_write     benchmark_fused_attn
#
# Profile outputs (timestamped, never overwrite):
#   <project_root>/results/profiles/attention_<TARGET>_<stamp>.{nsys-rep,ncu-rep}
#
# nsys is only produced for the umbrella benchmark; sub-ops get ncu only.
# scp the .rep files back to a workstation and open them in the Nsight GUIs.

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
_KERNEL_PARENT="$(cd "$(dirname "$_SCRIPT")" && pwd)"
cd "$_KERNEL_PARENT/../../.."
source setup.sh

KERNEL_DIR=$(basename "$_KERNEL_PARENT")            # "attention"

# ---- Arg parsing: positional [TARGET], optional --profile -----------------
PROFILE=false
TARGET="benchmark"
for arg in "$@"; do
    case "$arg" in
        --profile) PROFILE=true ;;
        --*)       echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)         TARGET="$arg" ;;
    esac
done

# Umbrella lives in the kernel dir; sub-ops live in benchmark_scripts/.
if [ "$TARGET" = "benchmark" ]; then
    MODULE="src.kernels.${KERNEL_DIR}.${TARGET}"
else
    MODULE="src.kernels.${KERNEL_DIR}.benchmark_scripts.${TARGET}"
fi

echo "=========================================================="
echo "    $KERNEL_DIR / $TARGET   (profile=$PROFILE)"
echo "    Module: $MODULE"
echo "=========================================================="
echo "[setup] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

# ---- No-profile path: just run the benchmark -------------------------------
if [ "$PROFILE" = false ]; then
    echo "[run] srun python -m $MODULE"
    srun --partition=gpu-turing --gres=gpu:1 python -m "$MODULE"
    exit 0
fi

# ---- Profile path ----------------------------------------------------------
# ncu/nsys write lock files to /tmp by default; on the shared compute nodes
# that collides with other users. Redirect to a per-user dir.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
OUT_DIR="$PROJECT_ROOT/results/profiles"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$OUT_DIR/${KERNEL_DIR}_${TARGET}_${STAMP}"

# Per-target ncu kernel filter. Empty means "skip ncu" — qkv_proj/o_proj are
# pure cuBLAS calls with no custom __global__ to profile.
case "$TARGET" in
    benchmark)            KERNEL_REGEX="rope_kernel|kv_write_kernel|flash_attention_kernel" ;;
    benchmark_rope)       KERNEL_REGEX="rope_kernel" ;;
    benchmark_kv_write)   KERNEL_REGEX="kv_write_kernel" ;;
    benchmark_fused_attn) KERNEL_REGEX="flash_attention_kernel" ;;
    benchmark_qkv_proj|benchmark_o_proj) KERNEL_REGEX="" ;;
    *)                    KERNEL_REGEX=".*" ;;
esac

# nsys is only meaningful for the umbrella benchmark (multi-kernel timeline).
# benchmark.py:profile_main brackets the captured forward with
# cudaProfilerStart/Stop, so warmup stays out of the timeline.
if [ "$TARGET" = "benchmark" ]; then
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
fi

# ncu per-kernel metrics.
if [ -z "$KERNEL_REGEX" ]; then
    echo ""
    echo "[ncu] Skipped: '$TARGET' has no custom __global__ kernel (cuBLAS only)."
else
    # Umbrella uses cudaProfilerStart/Stop → ncu honors it via
    # --profile-from-start off, then captures all 3 custom kernels in the
    # bracketed forward (rope, kv_write, flash_attn). Sub-ops have no
    # bracket; they do 5 warmup + 1 NVTX-tagged launch, so skip-5-take-1.
    if [ "$TARGET" = "benchmark" ]; then
        NCU_SCOPE=(--profile-from-start off --launch-count 3)
    else
        NCU_SCOPE=(--launch-skip 5 --launch-count 1)
    fi
    echo ""
    echo "[ncu] Capturing per-kernel metrics (regex: $KERNEL_REGEX)..."
    srun --partition=gpu-turing --gres=gpu:1 \
        ncu \
            --set full \
            --kernel-name "regex:${KERNEL_REGEX}" \
            "${NCU_SCOPE[@]}" \
            --target-processes all \
            --import-source on \
            -f -o "$OUT_BASE" \
            python -m "$MODULE" --profile
    echo "  Wrote: ${OUT_BASE}.ncu-rep"
fi

echo ""
echo "Done. scp results to a workstation, e.g.:"
echo "  scp <cluster>:${OUT_BASE}.* ~/Downloads/"
