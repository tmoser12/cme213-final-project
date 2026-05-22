#!/bin/bash
#
# Capture both Nsight Systems (.nsys-rep, timeline) and Nsight Compute
# (.ncu-rep, per-kernel metrics) profiles for the flash_attention_kernel
# in src/kernels/attention/kernel.cu.
#
# To change the profiled workload, edit BATCH_SIZE / SEQ_LEN at the top of
#     src/kernels/attention/benchmark_scripts/profile_flash_attn.py
# This script reads those constants so the output filenames match.
#
# Output (timestamped, never overwrites):
#     <repo>/results/profiles/flash_attn_B{B}_S{S}_{stamp}.nsys-rep
#     <repo>/results/profiles/flash_attn_B{B}_S{S}_{stamp}.ncu-rep
#
# scp the .rep files back to a workstation and open them in the Nsight
# Systems / Nsight Compute GUIs.
#
# Usage:
#     bash src/kernels/attention/benchmark_scripts/run_profile_flash_attn.sh
#     bash src/kernels/attention/benchmark_scripts/run_profile_flash_attn.sh nsys-only
#     bash src/kernels/attention/benchmark_scripts/run_profile_flash_attn.sh ncu-only

set -euo pipefail

# This script lives at src/kernels/attention/benchmark_scripts/ -- four
# levels below the repo root.
_SCRIPT="${BASH_SOURCE[0]}"
_BENCH_DIR="$(cd "$(dirname "$_SCRIPT")" && pwd)"   # .../attention/benchmark_scripts
_KERNEL_PARENT="$(cd "$_BENCH_DIR/.." && pwd)"      # .../attention
cd "$_KERNEL_PARENT/../../.."                       # repo root

source setup.sh

MODE="${1:-both}"  # both | nsys-only | ncu-only

KERNEL_DIR=$(basename "$_KERNEL_PARENT")          # "attention"
OUT_DIR="$PROJECT_ROOT/results/profiles"
mkdir -p "$OUT_DIR"

# Read BATCH_SIZE / SEQ_LEN from the driver so output filenames match what
# was actually profiled. Importing the module only pulls in the constants;
# load_attention_ops() is deferred to main() so this does NOT trigger a JIT
# compile.
read B S <<EOF
$(python -c 'from src.kernels.attention.benchmark_scripts.profile_flash_attn import BATCH_SIZE, SEQ_LEN; print(BATCH_SIZE, SEQ_LEN)')
EOF

STAMP=$(date +%Y%m%d_%H%M%S)
BASENAME="flash_attn_B${B}_S${S}_${STAMP}"
NSYS_OUT="$OUT_DIR/$BASENAME"
NCU_OUT="$OUT_DIR/$BASENAME"

# ncu/nsys write temp files; the default /tmp/nsight-compute-lock is owned
# by other users on the shared compute nodes. Redirect to a per-user dir.
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

echo "=========================================================="
echo "    Profiling flash_attention_kernel"
echo "    Workload: B=$B  S=$S   (edit profile_flash_attn.py to change)"
echo "    Mode:     $MODE"
echo "    Out dir:  $OUT_DIR"
echo "=========================================================="

echo "[step] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

# ---- Nsight Systems: full timeline ----------------------------------------
# --capture-range=cudaProfilerApi    : start/stop bracketed by cudaProfiler*
# --trace=cuda,nvtx,cublas,osrt      : CUDA API + kernels + NVTX + cuBLAS + libc
# --cuda-memory-usage=true           : track allocs in timeline
# --stats=true                       : print kernel/API summary after capture
if [[ "$MODE" == "both" || "$MODE" == "nsys-only" ]]; then
    echo "[step] nsys (timeline) ..."
    srun --partition=gpu-turing --gres=gpu:1 \
        nsys profile \
            --trace=cuda,nvtx,cublas,osrt \
            --cuda-memory-usage=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --stats=true \
            --force-overwrite=true \
            -o "$NSYS_OUT" \
            python -m src.kernels.attention.benchmark_scripts.profile_flash_attn
fi

# ---- Nsight Compute: per-launch metrics -----------------------------------
# --set full                         : full metric set (occupancy, memory, etc.)
# --kernel-name regex:...            : match the templated flash attn symbol
# --profile-from-start off           : honor cudaProfilerStart in the driver
# --launch-skip 0 --launch-count 1   : capture first matching launch inside it
# --nvtx                             : record NVTX ranges in the report
# --import-source on                 : embed .cu source so the GUI can show it
# --target-processes all             : follow into the python child
if [[ "$MODE" == "both" || "$MODE" == "ncu-only" ]]; then
    echo "[step] ncu (per-kernel metrics) ..."
    srun --partition=gpu-turing --gres=gpu:1 \
        ncu \
            --set full \
            --kernel-name "regex:flash_attention_kernel" \
            --profile-from-start off \
            --launch-skip 0 \
            --launch-count 1 \
            --nvtx \
            --target-processes all \
            --import-source on \
            -f -o "$NCU_OUT" \
            python -m src.kernels.attention.benchmark_scripts.profile_flash_attn
fi

echo ""
echo "Done."
if [[ "$MODE" == "both" || "$MODE" == "nsys-only" ]]; then
    echo "   Nsight Systems:  ${NSYS_OUT}.nsys-rep"
fi
if [[ "$MODE" == "both" || "$MODE" == "ncu-only" ]]; then
    echo "   Nsight Compute:  ${NCU_OUT}.ncu-rep"
fi
echo ""
echo "   scp back to a workstation, e.g.:"
if [[ "$MODE" == "both" || "$MODE" == "nsys-only" ]]; then
    echo "     scp <cluster>:${NSYS_OUT}.nsys-rep ~/Downloads/"
fi
if [[ "$MODE" == "both" || "$MODE" == "ncu-only" ]]; then
    echo "     scp <cluster>:${NCU_OUT}.ncu-rep   ~/Downloads/"
fi
