#!/bin/bash
# setup.sh -- run once on the login node
# Creates a Python venv with PyTorch + HuggingFace dependencies.
# The nvhpc module provides CUDA 12.x headers/libs that pip torch will use.

set -e

module load course/cme213/nvhpc/24.1

# Create venv in project root (excluded from git via .gitignore)
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Then run: python scripts/verify_env.py  (on a GPU node)"
