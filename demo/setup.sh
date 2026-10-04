#!/usr/bin/env bash
# Run inside a CUDA development image with CUDA-enabled torch already installed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/local/cuda/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv-cache}"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export MAX_JOBS="${MAX_JOBS:-2}"
command -v nvcc >/dev/null
if [[ ! -x .venv/bin/python ]]; then
  uv venv --system-site-packages .venv --python /usr/bin/python3
fi
# pip honors already-installed system-site packages here; an unconstrained uv
# resolution can pull a newer torch/CUDA stack despite the inherited torch.
.venv/bin/python -m pip install -r demo/requirements.txt
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(.venv/bin/python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')}"
mkdir -p logs
BUILD_ROLE=target .venv/bin/python setup.py build_ext --inplace
BUILD_ROLE=draft .venv/bin/python setup.py build_ext --inplace
.venv/bin/hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --include '*.safetensors' '*.json' '*.txt' '*.jinja' \
  --local-dir models/Qwen2.5-7B-Instruct
.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --include '*.safetensors' '*.json' '*.txt' '*.jinja' \
  --local-dir models/Qwen2.5-0.5B-Instruct
echo 'Ready to validate, then run: .venv/bin/python -m demo.server'
