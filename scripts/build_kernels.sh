#!/bin/bash
# Build AOT CUDA extensions for runtime/production_kernels/{target,draft}/.
#
# Usage (from project root):
#   bash scripts/build_kernels.sh                  # both roles, all ops
#   bash scripts/build_kernels.sh draft            # all draft ops
#   bash scripts/build_kernels.sh target rmsnorm   # one op, one role
#   bash scripts/build_kernels.sh draft attention  # one op, one role
#   bash scripts/build_kernels.sh rmsnorm          # one op, both roles (back-compat)

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# First arg may be a role (target|draft|all) or, for back-compat, an op name.
ARG1="${1:-all}"
ARG2="${2:-all}"
case "$ARG1" in
  all|target|draft) BUILD_ROLE="$ARG1"; BUILD_KERNEL="$ARG2" ;;
  *)                 BUILD_ROLE="all";  BUILD_KERNEL="$ARG1" ;;  # arg1 is an op
esac

CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
export PROJECT_ROOT="$(pwd)"
export BUILD_ROLE BUILD_KERNEL

echo "Building role=$BUILD_ROLE kernel=$BUILD_KERNEL"
module load gnu12/12.3.0
module load cuda/12.2 2>/dev/null || true
export CC=gcc
export CXX=g++
export PYTHONNOUSERSITE=1

"$CONDA_PATH/bin/conda" run -n cme213 env \
    BUILD_ROLE="$BUILD_ROLE" BUILD_KERNEL="$BUILD_KERNEL" CC=gcc CXX=g++ \
    python setup.py build_ext --inplace

echo "Done. Built role=$BUILD_ROLE kernel=$BUILD_KERNEL"
