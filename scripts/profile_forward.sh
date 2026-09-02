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
# --graph profiles the CUDA-graph replay path instead (adds --cuda-graph-trace=node
# so nsys expands the graph into its kernel nodes). NVTX ranges do NOT fire under
# replay, so read the graph runs via cuda_gpu_kern_sum, not nvtx_sum. --forward
# {decode,verify} picks the S=1 decode graph or the S=γ verify graph.
#
# Usage (from project root):
#   bash scripts/profile_forward.sh --model target --seq-len 512
#   bash scripts/profile_forward.sh --model draft  --seq-len 2048 --decode-steps 32
#   bash scripts/profile_forward.sh --model draft  --graph                # graph-replay decode
#   bash scripts/profile_forward.sh --model target --graph --forward verify --gamma 4
#
# Output (timestamped, never overwrites):
#   results/profiles/<model>/full/forward_<forward>_<path>_s<seq>_<stamp>.nsys-rep
#
# Each run is a single short job, well under the 30-min cluster cap.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source setup.sh

MODEL=""
SEQ_LEN=512
DECODE_STEPS=32
FORWARD="decode"
GAMMA=4
GRAPH=""
PATH_TAG="eager"
GRAPH_TRACE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --seq-len)      SEQ_LEN="$2";      shift 2 ;;
        --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
        --forward)      FORWARD="$2";      shift 2 ;;
        --gamma)        GAMMA="$2";        shift 2 ;;
        --graph)        GRAPH="--graph"; PATH_TAG="graph"; GRAPH_TRACE="--cuda-graph-trace=node"; shift ;;
        -h|--help)      sed -n '2,27p' "$0"; exit 0 ;;
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
OUT_BASE="$OUT_DIR/forward_${FORWARD}_${PATH_TAG}_s${SEQ_LEN}_${STAMP}"

echo "=========================================================="
echo "    forward profile: model=$MODEL forward=$FORWARD path=$PATH_TAG seq_len=$SEQ_LEN decode_steps=$DECODE_STEPS"
echo "=========================================================="
echo "[nsys] Capturing prefill + $FORWARD ($PATH_TAG) timeline..."
srun --partition=gpu-turing --gres=gpu:1 \
    env RUNTIME_NVTX=1 nsys profile \
        --trace=cuda,nvtx,cublas,osrt \
        --cuda-memory-usage=true \
        $GRAPH_TRACE \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --force-overwrite=true \
        -o "$OUT_BASE" \
        python -m runtime.tools.profile_forward \
            --model "$MODEL" --seq-len "$SEQ_LEN" --decode-steps "$DECODE_STEPS" \
            --forward "$FORWARD" --gamma "$GAMMA" $GRAPH

echo "  Wrote: ${OUT_BASE}.nsys-rep"
echo ""
echo "Per-kernel GPU time (graph + eager):"
echo "  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum ${OUT_BASE}.nsys-rep"
echo "Per-op / per-phase NVTX breakdown (eager only — NVTX is silent under graph replay):"
echo "  nsys stats --report nvtx_sum ${OUT_BASE}.nsys-rep"
