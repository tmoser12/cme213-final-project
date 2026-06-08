#!/bin/bash
#
# scripts/benchmark_forward.sh
#
# Timed prefill/decode latency + tok/s sweep (NO nsys) for the native Qwen2
# runtime, across a few prompt lengths at batch=1. Default times the eager
# (launch-bound) decode; --graph routes decode through the CUDA-graph replay path
# (decode_step_graph) so the table reflects the captured forward. Prefill is
# always eager (never graphed).
#
# NVTX stays OFF (clean wall-clock); decode is timed with a per-token device sync
# so each sample is one token's true latency and the p90/max columns surface the
# host-side jitter spikes. profile_forward.py writes the table itself to:
#   results/profiles/<model>/full/forward_sweep_<stamp>.txt
#
# Usage (from project root):
#   bash scripts/benchmark_forward.sh                          # both models, default lengths
#   bash scripts/benchmark_forward.sh --model draft            # one model (repeat to add)
#   bash scripts/benchmark_forward.sh --seq-lens 128,512,1024  # override lengths
#   bash scripts/benchmark_forward.sh --decode-steps 64 --reps 5
#   bash scripts/benchmark_forward.sh --graph                  # graph-replay decode timing
#
# Continues past a failed model and exits non-zero if any failed.

set -uo pipefail   # NOT -e: one failed model must not abort the sweep

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source setup.sh

MODELS=()
SEQ_LENS="128,512,1024,2048"
DECODE_STEPS=32
REPS=3
GRAPH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODELS+=("$2");    shift 2 ;;
        --seq-lens)     SEQ_LENS="$2";     shift 2 ;;
        --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
        --reps)         REPS="$2";         shift 2 ;;
        --graph)        GRAPH="--graph";   shift   ;;
        -h|--help)      sed -n '2,23p' "$0"; exit 0 ;;
        *)              echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=(draft target)

export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"

PASS=(); FAIL=()
for model in "${MODELS[@]}"; do
    echo
    echo "=========================================================="
    echo "    forward latency sweep: model=$model  seq_lens=$SEQ_LENS  decode_steps=$DECODE_STEPS"
    echo "=========================================================="
    if srun --partition=gpu-turing --gres=gpu:1 \
            python -m runtime.tools.profile_forward \
                --sweep --model "$model" $GRAPH \
                --seq-lens "$SEQ_LENS" --decode-steps "$DECODE_STEPS" --reps "$REPS"; then
        PASS+=("$model")
    else
        FAIL+=("$model (exit $?)")
        echo "[WARN] $model sweep failed -- continuing." >&2
    fi
done

echo
echo "=================== FORWARD SWEEP SUMMARY ==================="
echo "PASS: ${#PASS[@]}"
[ ${#PASS[@]} -gt 0 ] && printf '  done %s\n' "${PASS[@]}"
if [ ${#FAIL[@]} -gt 0 ]; then
    echo "FAIL: ${#FAIL[@]}"
    printf '  FAIL %s\n' "${FAIL[@]}"
    exit 1
fi
echo "Tables under: results/profiles/<model>/full/forward_sweep_<path>_<stamp>.txt"
