"""Shape helpers derived from RuntimeConfig."""

from __future__ import annotations

from runtime.core.config import RuntimeConfig


def hidden(batch: int, seq: int, cfg: RuntimeConfig) -> tuple[int, int, int]:
    return (batch, seq, cfg.hidden_size)


def q_states(batch: int, seq: int, cfg: RuntimeConfig) -> tuple[int, int, int, int]:
    return (batch, cfg.num_attention_heads, seq, cfg.head_dim)


def kv_states(batch: int, seq: int, cfg: RuntimeConfig) -> tuple[int, int, int, int]:
    return (batch, cfg.num_key_value_heads, seq, cfg.head_dim)


def logits(batch: int, seq: int, cfg: RuntimeConfig) -> tuple[int, int, int]:
    return (batch, seq, cfg.vocab_size)


def kv_cache(batch: int, max_seq: int, cfg: RuntimeConfig) -> tuple[int, int, int, int, int]:
    """Layer-major: [num_layers, batch, num_kv_heads, max_seq, head_dim]."""
    return (
        cfg.num_hidden_layers,
        batch,
        cfg.num_key_value_heads,
        max_seq,
        cfg.head_dim,
    )


def numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def nbytes(shape: tuple[int, ...], cfg: RuntimeConfig) -> int:
    return numel(shape) * cfg.dtype_bytes


def weight_shapes(cfg: RuntimeConfig) -> dict[str, tuple[int, ...]]:
    """Canonical weight shapes — HF/PyTorch Linear layout [out_features, in_features]."""
    h, i, kv = cfg.hidden_size, cfg.intermediate_size, cfg.kv_dim
    return {
        "embed_tokens": (cfg.vocab_size, h),
        "final_norm": (h,),
        "lm_head": (cfg.vocab_size, h),
        "q_proj": (h, h),
        "k_proj": (kv, h),
        "v_proj": (kv, h),
        "o_proj": (h, h),
        "gate_proj": (i, h),
        "up_proj": (i, h),
        "down_proj": (h, i),
        "q_proj_bias": (h,),
        "k_proj_bias": (kv,),
        "v_proj_bias": (kv,),
        "input_layernorm": (h,),
        "post_attention_layernorm": (h,),
    }


def total_weight_bytes(cfg: RuntimeConfig) -> int:
    """Rough FP16 weight memory estimate including q/k/v biases."""
    w = weight_shapes(cfg)
    per_layer = sum(
        nbytes(w[k], cfg)
        for k in (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "q_proj_bias", "k_proj_bias", "v_proj_bias",
            "input_layernorm", "post_attention_layernorm",
        )
    )
    global_bytes = nbytes(w["embed_tokens"], cfg) + nbytes(w["final_norm"], cfg)
    if not cfg.tie_word_embeddings:
        global_bytes += nbytes(w["lm_head"], cfg)
    return global_bytes + per_layer * cfg.num_hidden_layers
