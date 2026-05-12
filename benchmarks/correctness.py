import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_PATH = Path("models/Qwen2.5-7B-Instruct")
PROMPTS_PATH = Path("benchmarks/prompts/mt_bench_subset.jsonl")
EMBEDDING_WEIGHT_NAME = "model.embed_tokens.weight"

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
    weight_names: WeightName | Iterable[WeightName] = EMBEDDING_WEIGHT_NAME,
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
    weight_name: WeightName = EMBEDDING_WEIGHT_NAME,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load a single tensor by name."""
    return load_weights(model_path, weight_name, device=device)[weight_name]


def run_embedding_layer(
    model_path: Path = MODEL_PATH,
    prompts_path: Path = PROMPTS_PATH,
    prompt_index: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Tokenize one prompt and pass its token IDs through the embedding layer."""
    prompt_record = load_prompt(prompts_path, prompt_index=prompt_index)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(prompt_record["prompt"], return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    embedding_weight = load_weight(model_path, EMBEDDING_WEIGHT_NAME, device=device)

    with torch.no_grad():
        embeddings = F.embedding(input_ids, embedding_weight)

    return prompt_record, input_ids, embeddings


if __name__ == "__main__":
    prompt, token_ids, embedding_output = run_embedding_layer()

    print(f"Prompt: {prompt['id']} ({prompt.get('category', 'unknown')})")
    print(f"Input IDs shape: {tuple(token_ids.shape)}")
    print(f"Embedding output shape: {tuple(embedding_output.shape)}")
