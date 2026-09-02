#!/bin/bash
# slurm/run_python.sh
# Run any Python script or module on a GPU node via SLURM.
#
# Usage (from project root):
#   bash slurm/run_python.sh scripts/verify_env.py
#   bash slurm/run_python.sh -m kernel_dev.target.kernels.rmsnorm.benchmark
#   bash slurm/run_python.sh scripts/my_script.py --flag value
#   bash slurm/run_python.sh --gpus 2 -m package.module --arg value
#
# Options (must appear before the Python invocation):
#   --gpus N       Number of GPUs (default: 1)
#   --time HH:MM:SS  Wall time, max 00:30:00 (default: 00:30:00)
#   --mem SIZE     Memory limit (default: 32G)
#   --cpus N       CPUs per task (default: 4)

set -e

NUM_GPUS=1
TIME="00:30:00"
MEM="32G"
CPUS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)   NUM_GPUS="$2"; shift 2 ;;
        --time)   TIME="$2";     shift 2 ;;
        --mem)    MEM="$2";      shift 2 ;;
        --cpus)   CPUS="$2";     shift 2 ;;
        --)       shift; break ;;
        -*)       break ;;  # start of python args (e.g. -m)
        *)        break ;;  # start of python args (e.g. script.py)
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: bash slurm/run_python.sh [options] <python-args...>"
    echo ""
    echo "Examples:"
    echo "  bash slurm/run_python.sh scripts/verify_env.py"
    echo "  bash slurm/run_python.sh -m kernel_dev.target.kernels.rmsnorm.benchmark"
    echo "  bash slurm/run_python.sh --gpus 2 scripts/my_script.py --flag value"
    exit 1
fi

cd "$(dirname "$0")/.." || exit
source setup.sh

echo "Submitting to gpu-turing (${NUM_GPUS} GPU(s), ${TIME}, ${MEM})..."
echo "Command: python $*"
echo ""

srun \
    --partition=gpu-turing \
    --gres=gpu:${NUM_GPUS} \
    --ntasks=${NUM_GPUS} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --time=${TIME} \
    python "$@"
