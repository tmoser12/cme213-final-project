#!/bin/bash
#
# scripts/collect_profiles.sh
#
# Collect timings (no --profile) AND profiles (--profile: nsys/ncu) for every
# kernel of BOTH models, as DISTINCT srun jobs.
#
# The cluster caps a single job at 30 min. This script does NOT run everything in
# one job: it reuses the per-kernel run_benchmark.sh drivers, each of which issues
# its own short srun(s) (one kernel; --profile adds an nsys + an ncu launch). So
# every job is a single kernel and stays well under the 30-min cap. Jobs run
# sequentially from this (login-node) script; you can Ctrl-C between jobs.
#
# Outputs land under results/profiles/<model>/<group>/ (placed by run_benchmark.sh
# for profiles and by the benchmark.py modules for reports):
#   timings  -> <label>_<stamp>.txt
#   profiles -> <label>_<stamp>.{nsys-rep,ncu-rep,sqlite}
#
# Usage:
#   bash scripts/collect_profiles.sh                  # both models, timings + profiles
#   bash scripts/collect_profiles.sh --model draft    # one model (repeat --model to add)
#   bash scripts/collect_profiles.sh --timings        # timings only (skip --profile passes)
#   bash scripts/collect_profiles.sh --profiles       # profiles only (skip timing passes)
#
# Notes:
#   * embedding is intentionally excluded.
#   * attention's qkv_proj / o_proj are cuBLAS-only; their --profile pass self-skips
#     ncu (no custom __global__ kernel). That is expected, not a failure.
#   * Runs continue past a failed kernel; a PASS/FAIL summary prints at the end and
#     the script exits non-zero if anything failed.

set -uo pipefail   # deliberately NOT -e: one failed kernel must not abort the sweep

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODELS=()
DO_TIMINGS=true
DO_PROFILES=true
while [ $# -gt 0 ]; do
    case "$1" in
        --model)    MODELS+=("$2"); shift 2 ;;
        --timings)  DO_PROFILES=false; shift ;;
        --profiles) DO_TIMINGS=false; shift ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=(draft target)

# Single-op kernels with a run_benchmark.sh + a --profile path.
SIMPLE_KERNELS=(swiglu rmsnorm residual_ops)
# attention umbrella + sub-ops (the positional TARGET its run_benchmark.sh takes).
ATTN_TARGETS=(benchmark benchmark_qkv_proj benchmark_o_proj \
              benchmark_fused_attn benchmark_rope_kv_write benchmark_decode_attn)

PASS=(); FAIL=()
run() {                       # run "<description>" <command...>
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
    for k in "${SIMPLE_KERNELS[@]}"; do
        sh="src/$model/kernels/$k/run_benchmark.sh"
        $DO_TIMINGS  && run "$model/$k [timings]" bash "$sh"
        $DO_PROFILES && run "$model/$k [profile]" bash "$sh" --profile
    done
    attn="src/$model/kernels/attention/run_benchmark.sh"
    for t in "${ATTN_TARGETS[@]}"; do
        $DO_TIMINGS  && run "$model/attention/$t [timings]" bash "$attn" "$t"
        $DO_PROFILES && run "$model/attention/$t [profile]" bash "$attn" "$t" --profile
    done
done

echo
echo "=================== COLLECTION SUMMARY ==================="
echo "PASS: ${#PASS[@]}"
[ ${#PASS[@]} -gt 0 ] && printf '  done %s\n' "${PASS[@]}"
if [ ${#FAIL[@]} -gt 0 ]; then
    echo "FAIL: ${#FAIL[@]}"
    printf '  FAIL %s\n' "${FAIL[@]}"
    echo "Succeeded outputs are under: results/profiles/<model>/<group>/"
    exit 1
fi
echo "All ${#PASS[@]} jobs completed. Outputs under: results/profiles/<model>/<group>/"
