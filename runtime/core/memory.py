"""Activation and KV-cache memory estimates."""

from __future__ import annotations

from runtime.core.config import RuntimeConfig
from runtime.core import shapes


def plan_memory(
    cfg: RuntimeConfig,
    batch: int | None = None,
    max_seq_len: int | None = None,
) -> dict:
    """
    Estimate buffer sizes for a forward pass.
    Returns a plain dict — no registry/framework overhead.
    """
    cfg.validate()
    batch = batch if batch is not None else cfg.max_batch
    max_seq_len = max_seq_len if max_seq_len is not None else cfg.max_seq_len

    if batch <= 0 or max_seq_len <= 0:
        raise ValueError("batch and max_seq_len must be positive")

    buffers = {
        "hidden_ping": shapes.hidden(batch, max_seq_len, cfg),
        "hidden_pong": shapes.hidden(batch, max_seq_len, cfg),
        "q_states": shapes.q_states(batch, max_seq_len, cfg),
        "k_states": shapes.kv_states(batch, max_seq_len, cfg),
        "v_states": shapes.kv_states(batch, max_seq_len, cfg),
        "mlp_gate": (batch, max_seq_len, cfg.intermediate_size),
        "logits": shapes.logits(batch, max_seq_len, cfg),
        "kv_cache_keys": shapes.kv_cache(batch, max_seq_len, cfg),
        "kv_cache_values": shapes.kv_cache(batch, max_seq_len, cfg),
    }

    buffer_bytes = {name: shapes.nbytes(shape, cfg) for name, shape in buffers.items()}
    kv_bytes = buffer_bytes["kv_cache_keys"] + buffer_bytes["kv_cache_values"]
    activation_bytes = sum(v for k, v in buffer_bytes.items() if not k.startswith("kv_cache"))
    weight_bytes = shapes.total_weight_bytes(cfg)

    return {
        "batch": batch,
        "max_seq_len": max_seq_len,
        "buffers": buffers,
        "buffer_bytes": buffer_bytes,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "kv_cache_bytes": kv_bytes,
        "runtime_bytes": activation_bytes + kv_bytes,
        "total_bytes": weight_bytes + activation_bytes + kv_bytes,
        "weight_mib": weight_bytes / (1024 * 1024),
        "activation_mib": activation_bytes / (1024 * 1024),
        "kv_cache_mib": kv_bytes / (1024 * 1024),
        "runtime_mib": (activation_bytes + kv_bytes) / (1024 * 1024),
    }


def runtime_bytes_per_seq(cfg: RuntimeConfig, batch: int = 1) -> int:
    """Bytes for activations + KV cache per unit of max_seq_len (linear in seq)."""
    return plan_memory(cfg, batch=batch, max_seq_len=1)["runtime_bytes"]


def max_seq_len_for_budget(
    cfg: RuntimeConfig,
    budget_bytes: int,
    batch: int = 1,
) -> int:
    """
    Largest max_seq_len such that runtime buffers fit in budget_bytes.

    All activation and KV buffers in plan_memory() scale linearly with seq length.
    """
    if budget_bytes <= 0:
        return 0
    per_seq = runtime_bytes_per_seq(cfg, batch)
    if per_seq <= 0:
        return 0
    max_s = budget_bytes // per_seq
    return min(max_s, cfg.max_position_embeddings)


def max_seq_len_after_weights(
    cfg: RuntimeConfig,
    free_vram_bytes: int,
    batch: int = 1,
    reserve_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """
    Compute max sequence length given VRAM remaining after ONE model's weights
    are loaded on the GPU. Pass free_vram_bytes from gpu_memory_snapshot()
    after loading only that model (not both 7B and 0.5B).

    reserve_bytes: headroom kept for CUDA kernels, scratch, fragmentation.
    """
    budget = free_vram_bytes - reserve_bytes
    max_s = max_seq_len_for_budget(cfg, budget, batch)
    plan = plan_memory(cfg, batch=batch, max_seq_len=max(max_s, 1))
    return {
        "batch": batch,
        "free_vram_bytes": free_vram_bytes,
        "reserve_bytes": reserve_bytes,
        "buffer_budget_bytes": budget,
        "runtime_bytes_per_seq": runtime_bytes_per_seq(cfg, batch),
        "max_seq_len": max_s,
        "runtime_bytes_at_max": plan["runtime_bytes"] if max_s > 0 else 0,
        "plan_at_max_seq_len": plan_memory(cfg, batch=batch, max_seq_len=max_s) if max_s > 0 else None,
    }
