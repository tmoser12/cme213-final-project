"""Hand-driven sanity check: type a prompt, watch the executor generate text.

Not a unittest — just an interactive eyeball test. Run on a GPU node:

    srun --partition=gpu-turing --gres=gpu:1 \
         python -m runtime.tests.eyeball_check

Edit ``PROMPT`` / ``N_NEW_TOKENS`` below (or pass them on the CLI) and read the
decoded output to confirm we're getting sane text out of the native runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Allow running directly (``python runtime/tests/eyeball_check.py``) as well as
# via ``-m``: ensure the project root is importable so ``runtime`` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime.tests.parity_support import default_7b_cfg, default_05b_cfg, load_hf_and_executor

PROMPT = "The capital of France is"
N_NEW_TOKENS = 32


def eyeball_check(
    prompt: str = PROMPT,
    n_new_tokens: int = N_NEW_TOKENS,
) -> str:
    """Tokenize ``prompt`` with HF, generate with our executor, decode with HF."""
    cfg = default_05b_cfg()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    _, executor = load_hf_and_executor(cfg, max_seq_len=len(prompt) + n_new_tokens + 64)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"\nprompt     : {prompt!r}")
    print(f"input_ids  : {input_ids[0].tolist()}")

    with torch.no_grad():
        full_ids = executor.greedy_extend(input_ids, n_new_tokens)

    new_ids = full_ids[0, input_ids.shape[1]:].tolist()
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(full_ids[0].tolist(), skip_special_tokens=True)

    print(f"new_ids    : {new_ids}")
    print(f"completion : {completion!r}")
    print(f"\nfull text  :\n{full_text}\n")
    return full_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=PROMPT, help="prompt string to feed the model")
    parser.add_argument("--n", type=int, default=N_NEW_TOKENS, help="number of new tokens")
    args = parser.parse_args()
    eyeball_check(args.prompt, args.n)


if __name__ == "__main__":
    main()
