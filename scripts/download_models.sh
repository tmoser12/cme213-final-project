#!/bin/bash
# scripts/download_models.sh
# Run on the LOGIN NODE (has internet). GPU nodes may not.
#
# Downloads both Qwen2.5 models into this repo's models/ directory (by default):
#   <repo>/models/Qwen2.5-7B-Instruct/
#   <repo>/models/Qwen2.5-0.5B-Instruct/
#
# Override destination with: MODELS_DIR=/path/to/folder bash scripts/download_models.sh
#
# Each folder is a full HuggingFace snapshot (config + tokenizer + *.safetensors).
# Load in Python with AutoModel.from_pretrained("/abs/path/to/models/Qwen2.5-7B-Instruct")
# or a path relative to your cwd — same strings work wherever you copy these dirs.
#
# Quota note: 7B FP16 ~= 14 GB, 0.5B FP16 ~= 1 GB. Total ~15 GB of your 200 GB quota.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$PROJECT_ROOT/models}"

mkdir -p "$MODELS_DIR"

conda activate cme213

echo "Downloading Qwen2.5-7B-Instruct (~14 GB) into:"
echo "  $MODELS_DIR/Qwen2.5-7B-Instruct"
hf download Qwen/Qwen2.5-7B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" \
    --repo-type model \
    --local-dir "$MODELS_DIR/Qwen2.5-7B-Instruct"

echo ""
echo "Downloading Qwen2.5-0.5B-Instruct (~1 GB) into:"
echo "  $MODELS_DIR/Qwen2.5-0.5B-Instruct"
hf download Qwen/Qwen2.5-0.5B-Instruct \
    --include "*.safetensors" "*.json" "*.txt" \
    --repo-type model \
    --local-dir "$MODELS_DIR/Qwen2.5-0.5B-Instruct"

echo ""
echo "Downloads complete."
echo "Weights live under: $MODELS_DIR"
echo "Verify:"
echo "  ls \"$MODELS_DIR/Qwen2.5-7B-Instruct\""
echo "  ls \"$MODELS_DIR/Qwen2.5-0.5B-Instruct\""
echo ""
echo "Point Transformers at these folders (absolute paths recommended), e.g.:"
echo "  $MODELS_DIR/Qwen2.5-7B-Instruct"
