#!/bin/bash
# setup.sh -- run once on the login node
# Installs PyTorch + Hugging Face dependencies into conda env `cme213`.
# The nvhpc module provides CUDA 12.x headers/libs that pip torch will use.

set -e

source activate cme213

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with: source activate cme213"
echo "Then run: python scripts/verify_env.py  (on a GPU node)"

# First-time conda on the cluster (if `conda` / `activate` are missing):
#   eval "$(/opt/ohpc/pub/compiler/anaconda3/2023.09-0/bin/conda shell.bash hook)"
#   conda init
#   source ~/.bashrc
#   conda create --name cme213 python=3.11
#   source activate cme213
#   pip install -r requirements.txt
