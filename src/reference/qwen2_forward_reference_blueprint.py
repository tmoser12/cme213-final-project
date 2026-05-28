"""
Qwen2 Forward-Pass Blueprint (for C++/CUDA runtime implementation)
==================================================================

Purpose
-------
This file is a readable architecture guide that translates the relevant
`modeling_qwen2.py` forward path into a clear, implementation-oriented form.

It is NOT intended to be run as a production model implementation.
Think of it as a "spec in Python form" for the host-side CUDA runtime.

Primary mapping targets from Hugging Face reference:
  - apply_rotary_pos_emb / rotate_half
  - Qwen2Attention.forward
  - Qwen2MLP.forward
  - Qwen2DecoderLayer.forward
  - Qwen2Model.forward
  - Qwen2ForCausalLM.forward (lm_head projection)
"""

from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Data containers you will mirror in C++ (conceptually)
# ---------------------------------------------------------------------------


@dataclass
class Qwen2ConfigShape:
    """Shape-only config needed for forward orchestration."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float


@dataclass
class RuntimeContext:
    """
    Mutable decode state.

    In real runtime code this will hold:
      - KV cache pointers/strides/capacity
      - current sequence positions
      - allocator/workspace handles
      - stream/event handles
    """

    cache_position_start: int = 0


# ---------------------------------------------------------------------------
# Utility logic that mirrors Qwen2 reference helpers
# ---------------------------------------------------------------------------


def rotate_half(x):
    """
    Reference behavior:
      x1 = first half of head dim
      x2 = second half of head dim
      return [-x2, x1]

    In CUDA implementation this is usually fused inside RoPE kernel.
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # Pseudocode concat; exact tensor ops omitted in this blueprint.
    return ("concat", -x2, x1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Mirrors HF `apply_rotary_pos_emb`.
    Q/K expected shape convention: [B, heads, T, Dh]
    cos/sin convention (before broadcast): [B, T, Dh]
    """

    # In HF:
    #   cos = cos.unsqueeze(1)
    #   sin = sin.unsqueeze(1)
    #   q_embed = q * cos + rotate_half(q) * sin
    #   k_embed = k * cos + rotate_half(k) * sin
    q_embed = ("q * cos + rotate_half(q) * sin", q, cos, sin)
    k_embed = ("k * cos + rotate_half(k) * sin", k, cos, sin)
    return q_embed, k_embed


def repeat_kv(k_or_v, num_kv_groups: int):
    """
    Mirrors HF `repeat_kv`:
      [B, n_kv_heads, T, Dh] -> [B, n_heads, T, Dh]
    by repeating along head axis when n_kv_heads < n_heads.
    """

    if num_kv_groups == 1:
        return k_or_v
    return ("repeat_kv_groups", k_or_v, num_kv_groups)


# ---------------------------------------------------------------------------
# Forward-pass blueprint: one decoder layer
# ---------------------------------------------------------------------------


def decoder_layer_forward_blueprint(
    hidden_states,
    layer_idx: int,
    attention_mask,
    position_embeddings: Tuple[object, object],  # (cos, sin)
    ctx: RuntimeContext,
    cfg: Qwen2ConfigShape,
):
    """
    Mirrors `Qwen2DecoderLayer.forward` sequencing exactly:

      residual = x
      x = input_rmsnorm(x)
      x = self_attention(x, ...)
      x = residual + x

      residual = x
      x = post_attention_rmsnorm(x)
      x = mlp(x)
      x = residual + x
    """

    # ---- Residual branch 1 (attention block) ----
    residual_1 = hidden_states
    x_norm_1 = ("rmsnorm_input_layer", layer_idx, hidden_states, cfg.rms_norm_eps)

    # Attention subgraph (mirrors Qwen2Attention.forward)
    q = ("linear_q_proj", layer_idx, x_norm_1)  # [B, T, H] -> [B, T, n_heads*Dh]
    k = ("linear_k_proj", layer_idx, x_norm_1)  # [B, T, n_kv_heads*Dh]
    v = ("linear_v_proj", layer_idx, x_norm_1)

    q = ("reshape_transpose_to_B_h_T_Dh", q)
    k = ("reshape_transpose_to_B_hkv_T_Dh", k)
    v = ("reshape_transpose_to_B_hkv_T_Dh", v)

    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # KV cache update/get (prefill appends full T; decode appends 1 token)
    k_full = ("kv_cache_update_and_get_k", layer_idx, k, ctx)
    v_full = ("kv_cache_update_and_get_v", layer_idx, v, ctx)

    # Repeat KV heads if grouped-query attention is used.
    num_kv_groups = cfg.num_heads // cfg.num_kv_heads
    k_rep = repeat_kv(k_full, num_kv_groups)
    v_rep = repeat_kv(v_full, num_kv_groups)

    scores = ("matmul_q_kT_scaled", q, k_rep, cfg.head_dim)  # / sqrt(Dh)
    scores_masked = ("add_causal_mask", scores, attention_mask)
    probs = ("softmax_fp32_then_cast_back", scores_masked)
    attn_ctx = ("matmul_probs_v", probs, v_rep)  # [B, n_heads, T, Dh]
    attn_ctx_merged = ("transpose_reshape_to_B_T_H", attn_ctx)
    attn_out = ("linear_o_proj", layer_idx, attn_ctx_merged)

    x_after_attn = ("residual_add", residual_1, attn_out)

    # ---- Residual branch 2 (MLP block) ----
    residual_2 = x_after_attn
    x_norm_2 = ("rmsnorm_post_attn", layer_idx, x_after_attn, cfg.rms_norm_eps)

    # Qwen2MLP forward:
    # down_proj( act(gate_proj(x)) * up_proj(x) )
    gate = ("linear_gate_proj", layer_idx, x_norm_2)
    up = ("linear_up_proj", layer_idx, x_norm_2)
    gate_act = ("hidden_act_silu", gate)
    mlp_intermediate = ("elementwise_mul", gate_act, up)
    mlp_out = ("linear_down_proj", layer_idx, mlp_intermediate)

    x_after_mlp = ("residual_add", residual_2, mlp_out)
    return x_after_mlp


# ---------------------------------------------------------------------------
# Full model forward blueprint (Qwen2Model + LM head)
# ---------------------------------------------------------------------------


def qwen2_forward_blueprint(
    input_ids,
    attention_mask,
    ctx: RuntimeContext,
    cfg: Qwen2ConfigShape,
):
    """
    End-to-end sequence to mirror in runtime:

      1) Embed tokens
      2) Build/refresh cache positions
      3) Build causal mask
      4) Precompute RoPE cos/sin for this step
      5) Loop over decoder layers
      6) Final RMSNorm
      7) LM head projection -> logits
    """

    # 1) Embedding lookup (Qwen2Model.embed_tokens)
    hidden_states = ("embed_tokens", input_ids)

    # 2) Cache position / position_ids
    seq_len = ("shape_T", hidden_states)
    cache_position = ("arange", ctx.cache_position_start, "to", ctx.cache_position_start, "+", seq_len)
    position_ids = ("unsqueeze_batch_dim", cache_position)

    # 3) Causal mask update (mirrors _update_causal_mask behavior)
    causal_mask = ("build_or_update_causal_mask", attention_mask, hidden_states, cache_position)

    # 4) Shared rotary embeddings for all layers in this forward pass
    position_embeddings = ("rotary_embedding", hidden_states, position_ids)  # (cos, sin)

    # 5) Decoder stack
    for layer_idx in range(cfg.num_layers):
        hidden_states = decoder_layer_forward_blueprint(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            ctx=ctx,
            cfg=cfg,
        )

    # 6) Final RMSNorm (Qwen2Model.norm)
    hidden_states = ("final_rmsnorm", hidden_states, cfg.rms_norm_eps)

    # 7) LM head (Qwen2ForCausalLM.lm_head)
    logits = ("lm_head_linear_projection", hidden_states)  # [B, T, vocab_size]

    # Decode step update:
    # for token-by-token generation, increment cache start by the number of new tokens.
    ctx.cache_position_start = ("cache_position_start + seq_len", ctx.cache_position_start, seq_len)
    return logits


# ---------------------------------------------------------------------------
# Optional: tiny "one glance" checklist for C++ implementation order
# ---------------------------------------------------------------------------


IMPLEMENTATION_CHECKLIST = [
    "Embedding lookup",
    "Causal mask construction / update",
    "RoPE cos/sin generation",
    "Per-layer: input RMSNorm",
    "Per-layer: QKV projections + reshape/transposes",
    "Per-layer: RoPE on Q/K",
    "Per-layer: KV cache update + fetch",
    "Per-layer: repeat_kv for grouped-query attention",
    "Per-layer: attention score/mask/softmax/value matmul",
    "Per-layer: output projection + residual add",
    "Per-layer: post-attn RMSNorm + SwiGLU MLP + residual add",
    "Final RMSNorm",
    "LM head projection to logits",
]

