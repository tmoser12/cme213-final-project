#!/bin/bash

# Use BASH_SOURCE so this works when executed OR sourced ($0 is e.g. "-bash" when sourced).
_SCRIPT="${BASH_SOURCE[0]}"
_KERNEL_PARENT="$(cd "$(dirname "$_SCRIPT")" && pwd)" || exit

# Navigate to the project root directory so python paths work correctly
cd "$_KERNEL_PARENT/../../.." || exit

# Source the required modules and conda environment
source setup.sh

# Get the name of the kernel folder (e.g. "rmsnorm", "swiglu", "attention")
KERNEL_DIR=$(basename "$_KERNEL_PARENT")

# Optional second arg: which benchmark target to run inside this kernel dir
# (e.g. "benchmark", "benchmark_rope", "benchmark_fused_attn"). Defaults to
# the umbrella "benchmark" target so the no-arg invocation matches rmsnorm.
TARGET="${1:-benchmark}"

# Dispatch: the umbrella "benchmark" lives in the kernel dir; per-sub-op
# benchmarks live in `benchmark_scripts/` (created when those files were
# moved out of the top-level kernel dir).
if [ "$TARGET" = "benchmark" ]; then
    MODULE="src.kernels.${KERNEL_DIR}.${TARGET}"
else
    MODULE="src.kernels.${KERNEL_DIR}.benchmark_scripts.${TARGET}"
fi

echo "=========================================================="
echo "    Running Benchmark for: $KERNEL_DIR / $TARGET"
echo "    Module:                $MODULE"
echo "=========================================================="

# Clear the JIT cache specifically for this kernel to force recompilation of the C++ extension
echo "[1/2] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

# Run the benchmark on the Turing GPU
echo "[2/2] Submitting to Slurm (gpu-turing)..."
srun --partition=gpu-turing --gres=gpu:1 python -m "$MODULE"
