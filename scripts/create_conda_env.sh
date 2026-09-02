#!/bin/bash
# setup.sh -- run once on the login node
# Installs PyTorch + Hugging Face dependencies into conda env `cme213`.

set -e

# Initialize conda for this script session
CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
eval "$($CONDA_PATH/bin/conda shell.bash hook)"

# Check if environment exists, if not create it (optional but good for robustness)
if ! conda info --envs | grep -q "cme213"; then
    echo "Environment 'cme213' not found. Creating it..."
    conda create --name cme213 python=3.11 -y
fi

conda activate cme213

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with: conda activate cme213"
echo "Note: You may need to run '$CONDA_PATH/bin/conda init bash' and restart your shell first."
echo "Then run: python scripts/verify_env.py  (on a GPU node)"
