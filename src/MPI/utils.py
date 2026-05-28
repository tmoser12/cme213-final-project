from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def load_prompt(prompt_text: str | None, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8").strip()
    if prompt_text is None:
        raise ValueError("Prompt cannot be empty.")
    return prompt_text.strip()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def rank_log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def now() -> float:
    return time.perf_counter()


def summarize_result(result: dict[str, Any]) -> str:
    keys = [
        "rank",
        "role",
        "device",
        "model_path",
        "load_s",
        "gen_s",
        "prompt_tokens",
        "new_tokens",
        "tok_per_s",
    ]
    compact = {k: result.get(k) for k in keys}
    return json.dumps(compact, sort_keys=True)

