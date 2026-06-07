#!/bin/bash
#
# scripts/profile_forward.sh
#
# nsys timeline of a full native forward pass (prefill + decode) for one model
# at one prompt length, built from the AOT kernel .so's via runtime/executor.py.
#
# Per-op / per-layer / per-phase attribution comes from the executor's NVTX
# ranges (RUNTIME_NVTX=1, set here). The driver brackets the prefill+decode
# region with cudaProfilerStart/Stop, so warmup + weight-load stay out of the
# capture (nsys --capture-range=cudaProfilerApi).
#
# Usage (from project root):
#   bash scripts/profile_forward.sh --model target --seq-len 512
#   bash scripts/profile_forward.sh --model draft  --seq-len 2048 --decode-steps 32
#
# Output (timestamped, never overwrites):
#   results/profiles/<model>/full/forward_s<seq>_<stamp>.nsys-rep
#
# Each run is a single short job, well under the 30-min cluster cap.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source setup.sh

MODEL=""
SEQ_LEN=512
DECODE_STEPS=32
while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --seq-len)      SEQ_LEN="$2";      shift 2 ;;
        --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
        -h|--help)      sed -n '2,22p' "$0"; exit 0 ;;
        *)              echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [ "$MODEL" != "target" ] && [ "$MODEL" != "draft" ]; then
    echo "Error: --model must be 'target' or 'draft'" >&2
    exit 1
fi

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
OUT_DIR="$PROJECT_ROOT/results/profiles/$MODEL/full"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="$OUT_DIR/forward_s${SEQ_LEN}_${STAMP}"

echo "=========================================================="
echo "    forward profile: model=$MODEL seq_len=$SEQ_LEN decode_steps=$DECODE_STEPS"
echo "=========================================================="
echo "[nsys] Capturing prefill + decode timeline..."
srun --partition=gpu-turing --gres=gpu:1 \
    env RUNTIME_NVTX=1 nsys profile \
        --trace=cuda,nvtx,cublas,osrt \
        --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        -o "$OUT_BASE" \
        python -m runtime.tools.profile_forward \
            --model "$MODEL" --seq-len "$SEQ_LEN" --decode-steps "$DECODE_STEPS"

echo "  Wrote: ${OUT_BASE}.nsys-rep"
echo ""
echo "Per-op / per-phase breakdown:"
echo "  nsys stats --report cuda_gpu_kern_sum,nvtx_sum,cuda_gpu_mem_time_sum ${OUT_BASE}.nsys-rep"
