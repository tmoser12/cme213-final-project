"""Load and validate each model's safetensors weights for the shared GPU."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from runtime.core.config import RuntimeConfig
from runtime.core import shapes

_TORCH_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}


def _shard_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        return sorted({model_dir / f for f in data["weight_map"].values()})
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors found in {model_dir}")


def expected_keys(cfg: RuntimeConfig) -> dict[str, tuple[int, ...]]:
    """HF tensor name -> expected shape."""
    w = shapes.weight_shapes(cfg)
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": w["embed_tokens"],
        "model.norm.weight": w["final_norm"],
    }
    if not cfg.tie_word_embeddings:
        expected["lm_head.weight"] = w["lm_head"]

    for layer in range(cfg.num_hidden_layers):
        p = f"model.layers.{layer}"
        expected[f"{p}.input_layernorm.weight"] = w["input_layernorm"]
        expected[f"{p}.post_attention_layernorm.weight"] = w["post_attention_layernorm"]
        expected[f"{p}.self_attn.q_proj.weight"] = w["q_proj"]
        expected[f"{p}.self_attn.k_proj.weight"] = w["k_proj"]
        expected[f"{p}.self_attn.v_proj.weight"] = w["v_proj"]
        expected[f"{p}.self_attn.o_proj.weight"] = w["o_proj"]
        expected[f"{p}.self_attn.q_proj.bias"] = w["q_proj_bias"]
        expected[f"{p}.self_attn.k_proj.bias"] = w["k_proj_bias"]
        expected[f"{p}.self_attn.v_proj.bias"] = w["v_proj_bias"]
        expected[f"{p}.mlp.gate_proj.weight"] = w["gate_proj"]
        expected[f"{p}.mlp.up_proj.weight"] = w["up_proj"]
        expected[f"{p}.mlp.down_proj.weight"] = w["down_proj"]

    return expected


def _load_raw(model_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in _shard_files(model_dir):
        tensors.update(load_file(str(shard)))
    return tensors


def validate_weights(weights: dict[str, torch.Tensor], cfg: RuntimeConfig) -> None:
    """Raise ValueError on missing keys or shape mismatches."""
    expected = expected_keys(cfg)
    missing = [k for k in expected if k not in weights]
    if missing:
        raise ValueError(f"missing {len(missing)} weight(s), e.g. {missing[:3]}")

    mismatches = []
    for name, exp_shape in expected.items():
        got = tuple(weights[name].shape)
        if got != exp_shape:
            mismatches.append(f"{name}: expected {exp_shape}, got {got}")

    if mismatches:
        raise ValueError("shape mismatches:\n  " + "\n  ".join(mismatches[:5]))

    if cfg.tie_word_embeddings and "lm_head.weight" in weights:
        raise ValueError("tie_word_embeddings=true but lm_head.weight found on disk")


# WALKTHROUGH [WEIGHTS] — see demo/WALKTHROUGH.md.
# Load named tensors, validate their shapes, then cast/copy each to the GPU.
# No Hugging Face model forward is constructed; names only identify parameters.
def load_weights(
    cfg: RuntimeConfig,
    device: str | torch.device = "cuda",
    dtype: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Load safetensors from cfg.model_path, validate shapes, cast, move to device.

    Transfers one tensor at a time to limit peak host memory during GPU load.
    Returns dict keyed by HF tensor names.
    """
    cfg.validate()
    model_dir = Path(cfg.model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_path not found: {model_dir}")

    target_dtype = _TORCH_DTYPE[dtype or cfg.dtype]
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA not available")

    raw = _load_raw(model_dir)
    validate_weights(raw, cfg)
    expected = expected_keys(cfg)

    loaded: dict[str, torch.Tensor] = {}
    for name in expected:
        loaded[name] = raw[name].to(device=dev, dtype=target_dtype, non_blocking=False)
        del raw[name]

    return loaded
