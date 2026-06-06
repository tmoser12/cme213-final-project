#!/bin/bash
# slurm/run_tests_gpu.sh
# Run GPU-dependent tests on a Turing node.
#
# Usage (from project root):
#   bash slurm/run_tests_gpu.sh
#   bash slurm/run_tests_gpu.sh runtime.tests.test_weights.TestGpuLoad7B
#   bash slurm/run_tests_gpu.sh runtime.tests.test_decoder_layer   # Phase 6 layer parity
#   bash slurm/run_tests_gpu.sh runtime.tests.test_parity_greedy  # Phase 6 greedy trajectory

set -e
cd "$(dirname "$0")/.." || exit

# Default: CPU setup tests (also runnable on GPU nodes)
TARGET="${1:-}"
CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
export PROJECT_ROOT="$(pwd)"

if [ -n "$TARGET" ]; then
  TEST_CMD="'$CONDA_PATH/bin/conda' run -n cme213 python -m unittest '$TARGET' -v"
  echo "Running GPU tests: $TARGET"
else
  TEST_CMD="'$CONDA_PATH/bin/conda' run -n cme213 python -m unittest runtime.tests.test_config runtime.tests.test_memory runtime.tests.test_engine_setup -v"
  echo "Running setup tests: runtime/tests (config, memory, engine_setup)"
fi

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
        $TEST_CMD
    "
