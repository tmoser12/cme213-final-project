#!/bin/bash
# slurm/run_tests_gpu.sh
# Run GPU-dependent tests on a Turing node.
#
# Usage (from project root):
#   bash slurm/run_tests_gpu.sh
#   bash slurm/run_tests_gpu.sh runtime.tests.test_weights.TestGpuLoad7B
#   bash slurm/run_tests_gpu.sh runtime.tests.test_rmsnorm

set -e
cd "$(dirname "$0")/.." || exit

# Default: canonical 7B-only GPU load + VRAM budget (single model per GPU)
TARGET="${1:-runtime.tests.test_weights.TestGpuLoad7B}"
CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
export PROJECT_ROOT="$(pwd)"

echo "Running GPU tests: $TARGET"
PYTHONNOUSERSITE=1 srun \
    --partition=gpu-turing \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    bash -lc "
        module load gnu12/12.3.0
        export PROJECT_ROOT='$PROJECT_ROOT'
        export PYTHONNOUSERSITE=1
        '$CONDA_PATH/bin/conda' run -n cme213 python -m unittest '$TARGET' -v
    "
