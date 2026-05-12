"""
src/kernels/embedding/test.py
Isolated correctness test for the custom CUDA embedding kernel.

Setup:
  - Tokenize one prompt from benchmarks/prompts/mt_bench_subset.jsonl using
    Qwen2.5-7B's tokenizer.
  - Build an nn.Embedding sized from Qwen2Config (V=152064, H=3584) and
    initialize its weight from the real Qwen2.5-7B safetensors shard.
  - Run the prompt through both BaselineEmbedding (F.embedding reference)
    and CustomEmbedding (our CUDA kernel).
  - Assert torch.equal -- a pure gather is bit-exact, so any deviation is a
    real bug.

Run from the repo root with the cme213 conda env active and a GPU node:
    python -m src.kernels.embedding.test
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow `python src/kernels/embedding/test.py` in addition to `-m` form.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer, Qwen2Config

from benchmarks.correctness import (
    EMBEDDING_WEIGHT_NAME,
    load_prompt,
    load_weight,
)
from src.kernels.embedding.baseline import BaselineEmbedding
from src.kernels.embedding.wrapper import CustomEmbedding


MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
PROMPTS_PATH = REPO_ROOT / "benchmarks" / "prompts" / "mt_bench_subset.jsonl"


def main() -> None:
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device. Run on a GPU node.", file=sys.stderr)
        sys.exit(1)

    device = "cuda:0"

    # 1. Tokenize a real prompt.
    prompt = load_prompt(PROMPTS_PATH, prompt_index=0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    input_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to(device)

    # 2. Size an nn.Embedding from Qwen2Config; fill it with the real weight.
    config = Qwen2Config.from_pretrained(MODEL_PATH)
    weight = load_weight(MODEL_PATH, EMBEDDING_WEIGHT_NAME, device=device)
    assert weight.shape == (config.vocab_size, config.hidden_size), (
        f"weight {tuple(weight.shape)} mismatches config "
        f"({config.vocab_size}, {config.hidden_size})"
    )

    host = nn.Embedding(config.vocab_size, config.hidden_size).to(
        device=device, dtype=weight.dtype
    )
    with torch.no_grad():
        host.weight.copy_(weight)

    baseline = BaselineEmbedding(host).to(device).eval()
    custom = CustomEmbedding(host).to(device).eval()

    # 3. Forward both. Warm up the kernel first so launch errors surface
    # before the measured call.
    with torch.no_grad():
        _ = custom(input_ids)
        torch.cuda.synchronize(device)

        out_ref = baseline(input_ids)
        out_cuda = custom(input_ids)
        torch.cuda.synchronize(device)

    # 4. Compare. Pure gather -> must be bit-exact.
    exact = torch.equal(out_ref, out_cuda)
    max_abs_diff = (out_ref.float() - out_cuda.float()).abs().max().item()

    print(f"Prompt        : {prompt['id']} ({prompt.get('category', 'unknown')})")
    print(f"Device        : {device}")
    print(f"Input ids     : {tuple(input_ids.shape)}  {input_ids.dtype}")
    print(f"Weight        : {tuple(weight.shape)}  {weight.dtype}")
    print(f"Output        : {tuple(out_cuda.shape)}  {out_cuda.dtype}")
    print(f"max_abs_diff  : {max_abs_diff}")
    print(f"torch.equal   : {exact}")
    print("PASS" if exact else "FAIL")
    if not exact:
        sys.exit(1)


if __name__ == "__main__":
    main()
