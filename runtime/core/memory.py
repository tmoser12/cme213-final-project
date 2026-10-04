"""Activation and KV-cache memory estimates."""

from __future__ import annotations

from runtime.core.config import RuntimeConfig
from runtime.core import shapes


# WALKTHROUGH [PLAN] — see demo/WALKTHROUGH.md.
# Estimate reserved tensor sizes. This is not a peak-memory measurement: packed
# QKV copies, eager intermediates, and graph-pool allocations add more storage.
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
        "rope_cos": shapes.rope_table(max_seq_len, cfg),
        "rope_sin": shapes.rope_table(max_seq_len, cfg),
    }

    buffer_bytes = {name: shapes.nbytes(shape, cfg) for name, shape in buffers.items()}
    buffer_bytes["cache_position"] = shapes.cache_position_bytes()
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
