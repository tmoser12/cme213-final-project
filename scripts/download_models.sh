#!/bin/bash
# scripts/download_models.sh
# Run on the LOGIN NODE (has internet). GPU nodes may not.
# Downloads both Qwen2.5 models into HuggingFace's default cache:
#   ~/.cache/huggingface/hub/
# which is on your home filesystem and therefore accessible from GPU nodes.
#
# Quota note: 7B FP16 ~= 14 GB, 0.5B FP16 ~= 1 GB. Total ~15 GB of your 200 GB quota.

set -e

source activate cme213

echo "Downloading Qwen2.5-7B-Instruct (~14 GB)..."
hf download Qwen/Qwen2.5-7B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" \
    --repo-type model

echo ""
echo "Downloading Qwen2.5-0.5B-Instruct (~1 GB)..."
hf download Qwen/Qwen2.5-0.5B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" \
    --repo-type model

echo ""
echo "Downloads complete. Verify with:"
echo "  ls ~/.cache/huggingface/hub/"
