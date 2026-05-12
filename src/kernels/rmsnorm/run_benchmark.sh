#!/bin/bash

# Navigate to the project root directory so python paths work correctly
cd "$(dirname "$0")/../../.." || exit

# Source the required modules and conda environment
source setup.sh

# Get the name of the kernel folder (e.g. "rmsnorm", "swiglu")
KERNEL_DIR=$(basename "$(dirname "$0")")

echo "=========================================================="
echo "    Running Benchmark for: $KERNEL_DIR"
echo "=========================================================="

# Clear the JIT cache specifically for this kernel to force recompilation of the C++ extension
echo "[1/2] Clearing PyTorch JIT cache for custom_${KERNEL_DIR}_ops..."
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_${KERNEL_DIR}_ops

# Run the benchmark on the Turing GPU
echo "[2/2] Submitting to Slurm (gpu-turing)..."
srun --partition=gpu-turing --gres=gpu:1 python -m src.kernels.${KERNEL_DIR}.benchmark
