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
#   benchmark_fused_attn
#   benchmark_rope_kv_write (RoPE-fused KV write — the pipeline's K path)
#   benchmark_decode_attn   (decode/small-q attention, populated KV cache)
#
# Profile outputs (timestamped, never overwrite):
#   results/profiles/<model>/attention/<sub-op>_<stamp>.{nsys-rep,ncu-rep}
#
# nsys is only produced for the umbrella benchmark; sub-ops get ncu only.
# scp the .rep files back to a workstation and open them in the Nsight GUIs.

set -euo pipefail

_SCRIPT="${BASH_SOURCE[0]}"
_KERNEL_PARENT="$(cd "$(dirname "$_SCRIPT")" && pwd)"
cd "$_KERNEL_PARENT/../../../.."
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
    MODULE="src.target.kernels.${KERNEL_DIR}.${TARGET}"
else
    MODULE="src.target.kernels.${KERNEL_DIR}.benchmark_scripts.${TARGET}"
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
# Route profiles into results/profiles/<model>/<group>/ (model from path, group from kernel).
case "$_KERNEL_PARENT" in
    */src/draft/*)  MODEL=draft ;;
    */src/target/*) MODEL=target ;;
    *)              MODEL=unknown ;;
esac
case "$KERNEL_DIR" in
    swiglu)    GROUP=swiglu ;;
    attention) GROUP=attention ;;
    *)         GROUP=misc ;;
esac
OUT_DIR="$PROJECT_ROOT/results/profiles/$MODEL/$GROUP"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LABEL="${TARGET#benchmark_}"; [ "$TARGET" = "benchmark" ] && LABEL="attention"
OUT_BASE="$OUT_DIR/${LABEL}_${STAMP}"

case "$TARGET" in
    benchmark)            KERNEL_REGEX="rope_kv_write_kernel|flash_attention_kernel" ;;
    benchmark_rope_kv_write) KERNEL_REGEX="rope_kv_write_kernel" ;;
    benchmark_fused_attn) KERNEL_REGEX="flash_attention_kernel" ;;
    benchmark_decode_attn) KERNEL_REGEX="decode_attn_split_kernel|decode_attn_combine_kernel|decode_attn_kernel" ;;
    benchmark_qkv_proj|benchmark_o_proj) KERNEL_REGEX="" ;;
    *)                    KERNEL_REGEX=".*" ;;
esac

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
    if [ "$TARGET" = "benchmark" ]; then
        NCU_SCOPE=(--profile-from-start off --launch-count 3)
    elif [ "$TARGET" = "benchmark_decode_attn" ]; then
        # Decode emits TWO kernels per call (split + combine) on the split-KV path,
        # so skip the 5 warmup iters (10 kernels) and capture both kernels of the
        # measured launch. Short-context configs fall back to decode_attn_kernel.
        NCU_SCOPE=(--launch-skip 10 --launch-count 2)
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
