"""
src/models/loading.py
Shared helpers for loading Qwen2.5 weights from disk and prompts from the
benchmark JSONL files.
"""

import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
PROMPTS_PATH = REPO_ROOT / "benchmarks" / "prompts" / "mt_bench_subset.jsonl"

WeightName = str

def load_prompt(path: Path = PROMPTS_PATH, prompt_index: int = 0) -> dict:
    """Load one prompt record from a JSONL prompt file."""
    with path.open() as prompt_file:
        for index, line in enumerate(prompt_file):
            line = line.strip()
            if not line:
                continue
            if index == prompt_index:
                return json.loads(line)


def _as_weight_list(weight_names: WeightName | Iterable[WeightName]) -> list[WeightName]:
    if isinstance(weight_names, str):
        return [weight_names]
    return list(weight_names)


def load_weights(
    model_path: Path = MODEL_PATH,
    weight_names: WeightName | Iterable[WeightName] = (),
    device: torch.device | str | None = None,
) -> dict[WeightName, torch.Tensor]:
    """Load one or more tensors from the model's sharded safetensors files."""
    requested_names = _as_weight_list(weight_names)
    index_path = model_path / "model.safetensors.index.json"

    with index_path.open() as index_file:
        weight_map = json.load(index_file)["weight_map"]

    names_by_shard: dict[str, list[WeightName]] = {}
    for name in requested_names:
        try:
            shard_name = weight_map[name]
        except KeyError as exc:
            raise KeyError(f"{name!r} is not present in {index_path}") from exc
        names_by_shard.setdefault(shard_name, []).append(name)

    tensors: dict[WeightName, torch.Tensor] = {}
    target_device = str(device) if device is not None else "cpu"
    for shard_name, names in names_by_shard.items():
        with safe_open(model_path / shard_name, framework="pt", device=target_device) as shard:
            for name in names:
                tensors[name] = shard.get_tensor(name)

    return tensors


def load_weight(
    model_path: Path = MODEL_PATH,
    weight_name: WeightName = "",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load a single tensor by name."""
    return load_weights(model_path, weight_name, device=device)[weight_name]
