#!/bin/bash
# Build AOT CUDA extensions for runtime/production_kernels/target/.
#
# Usage (from project root):
#   bash scripts/build_kernels.sh           # all target kernels
#   bash scripts/build_kernels.sh rmsnorm   # one kernel

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

KERNEL="${1:-all}"
CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
export PROJECT_ROOT="$(pwd)"
export BUILD_KERNEL="$KERNEL"

echo "Building target kernel(s): $BUILD_KERNEL"
module load gnu12/12.3.0
module load cuda/12.2 2>/dev/null || true
export CC=gcc
export CXX=g++
export PYTHONNOUSERSITE=1

"$CONDA_PATH/bin/conda" run -n cme213 env BUILD_KERNEL="$BUILD_KERNEL" CC=gcc CXX=g++ python setup.py build_ext --inplace

echo "Done. Built extension(s) for: $BUILD_KERNEL"
