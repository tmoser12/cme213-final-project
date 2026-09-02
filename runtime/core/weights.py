"""Load and validate model weights from safetensors.

Assumption: exactly ONE model occupies a GPU at a time. Never load 7B and 0.5B
onto the same device. VRAM budget / max_seq_len math is based on free HBM after
that single model's weights are resident (production target: 7B on 24GB Turing).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from runtime.core.config import RuntimeConfig
from runtime.core.memory import max_seq_len_after_weights, plan_memory
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


def gpu_memory_snapshot(device: int | str = 0) -> dict[str, int | float | str]:
    """Current GPU HBM usage via torch.cuda."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run on a GPU node (srun --gres=gpu:1)")

    dev = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
    torch.cuda.synchronize(dev)
    props = torch.cuda.get_device_properties(dev)
    reserved = torch.cuda.memory_reserved(dev)
    allocated = torch.cuda.memory_allocated(dev)
    total = props.total_memory

    return {
        "device_name": props.name,
        "total_bytes": total,
        "total_mib": total / (1024 * 1024),
        "allocated_bytes": allocated,
        "allocated_mib": allocated / (1024 * 1024),
        "reserved_bytes": reserved,
        "reserved_mib": reserved / (1024 * 1024),
        "free_bytes": total - reserved,
        "free_mib": (total - reserved) / (1024 * 1024),
    }


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


def memory_report(weights: dict[str, torch.Tensor]) -> dict[str, int | float]:
    """Bytes occupied by loaded weight tensors."""
    total = sum(t.numel() * t.element_size() for t in weights.values())
    return {
        "num_tensors": len(weights),
        "total_bytes": total,
        "total_mib": total / (1024 * 1024),
    }


def vram_budget(
    cfg: RuntimeConfig,
    weights: dict[str, torch.Tensor],
    batch: int = 1,
    reserve_mib: float = 512.0,
    device: int = 0,
) -> dict:
    """
    VRAM snapshot after this model's weights are on GPU + max seq len for buffers.

    Assumes this is the ONLY model on the device. free_bytes = total HBM minus
    what is already reserved (including these weights). Use with CONFIG_7B for
    production buffer sizing on the 24GB Turing nodes.
    """
    snap = gpu_memory_snapshot(device)
    w_report = memory_report(weights)
    reserve_bytes = int(reserve_mib * 1024 * 1024)
    seq_info = max_seq_len_after_weights(
        cfg,
        free_vram_bytes=snap["free_bytes"],
        batch=batch,
        reserve_bytes=reserve_bytes,
    )
    plan = seq_info.get("plan_at_max_seq_len")
    return {
        "model": cfg.name,
        "single_model_on_gpu": True,
        "gpu": snap,
        "weights": w_report,
        "reserve_mib": reserve_mib,
        "max_seq_len": seq_info["max_seq_len"],
        "runtime_bytes_per_seq": seq_info["runtime_bytes_per_seq"],
        "buffer_budget_mib": seq_info["buffer_budget_bytes"] / (1024 * 1024),
        "runtime_mib_at_max_seq_len": (
            plan["runtime_mib"] if plan else 0.0
        ),
        "headroom_mib_at_max_seq_len": (
            (snap["free_bytes"] - reserve_bytes - plan["runtime_bytes"]) / (1024 * 1024)
            if plan and seq_info["max_seq_len"] > 0
            else 0.0
        ),
    }


def startup_report(cfg: RuntimeConfig, weights: dict[str, torch.Tensor]) -> dict:
    """Summary printed at runtime init."""
    report: dict = {
        "model": cfg.name,
        "model_path": cfg.model_path,
        "device": str(next(iter(weights.values())).device),
        "dtype": str(next(iter(weights.values())).dtype),
        "weights": memory_report(weights),
        "estimated_weight_mib": shapes.total_weight_bytes(cfg) / (1024 * 1024),
    }
    if next(iter(weights.values())).device.type == "cuda":
        report["vram"] = vram_budget(cfg, weights)
    return report


def load_weights_on_gpu(
    cfg: RuntimeConfig,
    batch: int = 1,
    reserve_mib: float = 512.0,
    device: str | torch.device = "cuda",
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Load one model's weights to GPU HBM and return (weights, vram_budget).

    Clears the CUDA cache first so the budget reflects a single model, not
    leftovers from a prior load. Do not call twice for different models on the
    same GPU without freeing the first model's tensors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    weights = load_weights(cfg, device=device)
    for t in weights.values():
        if t.device.type != "cuda":
            raise RuntimeError(f"weights not on CUDA: {t.device}")
    budget = vram_budget(cfg, weights, batch=batch, reserve_mib=reserve_mib)
    return weights, budget
