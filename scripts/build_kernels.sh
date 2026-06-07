#!/bin/bash
# Build AOT CUDA extensions for runtime/production_kernels/{target,draft}/.
#
# Usage (from project root):
#   bash scripts/build_kernels.sh                  # all roles, all kernels
#   bash scripts/build_kernels.sh rmsnorm          # all roles, rmsnorm (back-compat)
#   bash scripts/build_kernels.sh draft            # draft role, all kernels
#   bash scripts/build_kernels.sh draft attention  # draft attention only
#   bash scripts/build_kernels.sh target attention # target attention only

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

ARG1="${1:-all}"
ARG2="${2:-all}"
if [[ "$ARG1" == "target" || "$ARG1" == "draft" || "$ARG1" == "all" ]]; then
    ROLE="$ARG1"; KERNEL="$ARG2"
else
    ROLE="all"; KERNEL="$ARG1"   # back-compat: `build_kernels.sh attention`
fi

CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
export PROJECT_ROOT="$(pwd)"
export BUILD_ROLE="$ROLE"
export BUILD_KERNEL="$KERNEL"

echo "Building role=$ROLE kernel(s)=$KERNEL"
module load gnu12/12.3.0
module load cuda/12.2 2>/dev/null || true
export CC=gcc
export CXX=g++
export PYTHONNOUSERSITE=1

"$CONDA_PATH/bin/conda" run -n cme213 env BUILD_ROLE="$BUILD_ROLE" BUILD_KERNEL="$BUILD_KERNEL" CC=gcc CXX=g++ python setup.py build_ext --inplace

echo "Done. Built role=$ROLE kernel(s)=$KERNEL"
