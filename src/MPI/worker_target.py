from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MPIBaselineConfig
from .utils import now, rank_log


def run_target_worker(
    config: MPIBaselineConfig,
    rank: int,
    device: str,
    prompt: str,
) -> dict[str, Any]:
    rank_log(rank, f"Loading target model from {config.target_model_path} on {device}")
    load_start = now()
    tokenizer = AutoTokenizer.from_pretrained(
        config.target_model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.target_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    load_s = now() - load_start

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    gen_start = now()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_s = now() - gen_start

    total_tokens = int(outputs.shape[1])
    new_tokens = max(total_tokens - prompt_tokens, 0)
    tok_per_s = float(new_tokens / gen_s) if gen_s > 0 else 0.0

    return {
        "rank": rank,
        "role": "target",
        "device": device,
        "model_path": config.target_model_path,
        "load_s": round(load_s, 4),
        "gen_s": round(gen_s, 4),
        "prompt_tokens": prompt_tokens,
        "new_tokens": new_tokens,
        "tok_per_s": round(tok_per_s, 4),
    }

