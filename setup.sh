#!/bin/bash
# setup.sh -- SOURCING this file activates the environment and sets model paths.
# Usage: source setup.sh

# Initialize Conda hook if conda command is not already available
# CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
# if ! command -v conda &> /dev/null; then
#     eval "$($CONDA_PATH/bin/conda shell.bash hook)"
# fi

# Activate the project environment
conda activate cme213

# Load modern GCC for PyTorch C++ extensions
module load gnu12/12.3.0

# Set environment variables for the flattened model paths
export PROJECT_ROOT="/home/cme213/tobiascm/cme213-final-project"
export QWEN_7B_PATH="$PROJECT_ROOT/models/Qwen2.5-7B-Instruct"
export QWEN_05B_PATH="$PROJECT_ROOT/models/Qwen2.5-0.5B-Instruct"

# Optional: Set HF_HOME to the project's models directory to prevent accidental downloads to ~/.cache
export HF_HOME="$PROJECT_ROOT/models"

echo "--------------------------------------------------------"
echo "CME213 Final Project Environment Loaded"
echo "--------------------------------------------------------"
echo "Conda Env: cme213 (Active)"
echo "Models Path Variables:"
echo "  \$QWEN_7B_PATH  -> $QWEN_7B_PATH"
echo "  \$QWEN_05B_PATH -> $QWEN_05B_PATH"
echo "--------------------------------------------------------"
