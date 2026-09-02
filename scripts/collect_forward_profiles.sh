#!/bin/bash
#
# scripts/collect_forward_profiles.sh
#
# Sweep full-forward nsys profiles across BOTH models and the small length sweep
# (seq_len = 512, 2048). Each (model, seq_len) point is one short srun job via
# scripts/profile_forward.sh; jobs run sequentially and you can Ctrl-C between
# them. Every job stays well under the 30-min cluster cap.
#
# Outputs land under results/profiles/<model>/full/ (placed by
# profile_forward.sh):
#   forward_<forward>_<path>_s<seq>_<stamp>.nsys-rep
#
# --graph / --forward / --gamma are passed straight through to profile_forward.sh.
#
# Usage:
#   bash scripts/collect_forward_profiles.sh                 # both models, 512 + 2048
#   bash scripts/collect_forward_profiles.sh --model target  # one model (repeat to add)
#   bash scripts/collect_forward_profiles.sh --seq-len 1024  # override lengths (repeat to add)
#   bash scripts/collect_forward_profiles.sh --model draft --graph  # graph-replay nsys
#
# Continues past a failed job; prints a PASS/FAIL summary and exits non-zero on
# any failure.

set -uo pipefail   # NOT -e: one failed job must not abort the sweep

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODELS=()
SEQ_LENS=()
DECODE_STEPS=32
EXTRA=""            # --graph / --forward / --gamma, passed through to profile_forward.sh
while [ $# -gt 0 ]; do
    case "$1" in
        --model)        MODELS+=("$2");   shift 2 ;;
        --seq-len)      SEQ_LENS+=("$2");  shift 2 ;;
        --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
        --graph)        EXTRA="$EXTRA --graph";        shift   ;;
        --forward)      EXTRA="$EXTRA --forward $2";   shift 2 ;;
        --gamma)        EXTRA="$EXTRA --gamma $2";     shift 2 ;;
        -h|--help)      sed -n '2,23p' "$0"; exit 0 ;;
        *)              echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[ ${#MODELS[@]}   -eq 0 ] && MODELS=(draft target)
[ ${#SEQ_LENS[@]} -eq 0 ] && SEQ_LENS=(512 2048)

PASS=(); FAIL=()
run() {
    local desc="$1"; shift
    echo
    echo "############################################################"
    echo "# $desc"
    echo "#  \$ $*"
    echo "############################################################"
    if "$@"; then
        PASS+=("$desc")
    else
        local rc=$?
        FAIL+=("$desc (exit $rc)")
        echo "[WARN] $desc failed (exit $rc) -- continuing." >&2
    fi
}

for model in "${MODELS[@]}"; do
    for s in "${SEQ_LENS[@]}"; do
        run "$model/forward s=$s" \
            bash scripts/profile_forward.sh \
                --model "$model" --seq-len "$s" --decode-steps "$DECODE_STEPS" $EXTRA
    done
done

echo
echo "=================== FORWARD PROFILE SUMMARY ==================="
echo "PASS: ${#PASS[@]}"
[ ${#PASS[@]} -gt 0 ] && printf '  done %s\n' "${PASS[@]}"
if [ ${#FAIL[@]} -gt 0 ]; then
    echo "FAIL: ${#FAIL[@]}"
    printf '  FAIL %s\n' "${FAIL[@]}"
    echo "Succeeded outputs are under: results/profiles/<model>/full/"
    exit 1
fi
echo "All ${#PASS[@]} jobs completed. Outputs under: results/profiles/<model>/full/"
