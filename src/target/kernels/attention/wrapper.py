import math
import torch
import torch.nn as nn

from src.target.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()


class CustomQwen2Attention(nn.Module):
    """Drop-in replacement for Qwen2Attention.

    Pre-concatenates q/k/v weights and biases at construction so the runtime
    QKV projection is a single cuBLAS GEMM. o_proj weight is adopted by
    reference (no concat needed). Forward signature mirrors Qwen2Attention so
    patch_model() can splice us into HF's DecoderLayer.
    """

    def __init__(self, original_attention):
        super().__init__()
        cfg = original_attention.config
        self.num_heads     = cfg.num_attention_heads
        self.num_kv_heads  = cfg.num_key_value_heads
        self.head_dim      = original_attention.head_dim
        self.hidden_size   = cfg.hidden_size
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV weight: rows in order [Q | K | V].
        W_qkv = torch.cat([
            original_attention.q_proj.weight,
            original_attention.k_proj.weight,
            original_attention.v_proj.weight,
        ], dim=0).contiguous()
        b_qkv = torch.cat([
            original_attention.q_proj.bias,
            original_attention.k_proj.bias,
            original_attention.v_proj.bias,
        ], dim=0).contiguous()
        self.W_qkv = nn.Parameter(W_qkv, requires_grad=False)
        self.b_qkv = nn.Parameter(b_qkv, requires_grad=False)

        # Output projection: adopt by reference (no bias on o_proj for Qwen2).
        self.W_o = original_attention.o_proj.weight

    def forward(self, hidden_states, position_embeddings=None,
                past_key_value=None, attention_mask=None, **_):
        B, S, _ = hidden_states.shape
        M = B * S
        H_q  = self.num_heads    * self.head_dim
        H_kv = self.num_kv_heads * self.head_dim

        # 1) Fused QKV projection: (B, S, H) -> (M, H_q + 2*H_kv).
        qkv = custom_ops.qkv_proj_forward(
            hidden_states.view(M, self.hidden_size), self.W_qkv, self.b_qkv)

        # 2) Split Q/K/V, reshape to (B, H_*, S, D). The downstream custom
        #    kernels all require contiguous fp16 tensors, so we materialize
        #    after the transpose. The .reshape on the slice copies once
        #    (qkv slice is row-strided, not contiguous); .contiguous() after
        #    transpose copies once more. Two copies per Q/K/V is unavoidable
        #    short of changing W_qkv's row order.
        q = qkv[:, :H_q].reshape(B, S, self.num_heads,    self.head_dim).transpose(1, 2).contiguous()
        k = qkv[:, H_q:H_q + H_kv].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = qkv[:, H_q + H_kv:].reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 3) In-place RoPE on Q and K. cos/sin arrive as (B, S, D); our
        #    rope_kernel indexes them as (B, S, D) directly (HF unsqueezes
        #    to (B, 1, S, D) for broadcast, we don't need to).
        cos, sin = position_embeddings
        custom_ops.rope_forward(q, k, cos, sin)

        # 4) KV cache. This scaffold path is for the standalone attention
        #    benchmark — a single attention call with no rolling cache, so
        #    we allocate a transient cache of size S and write_pos = 0.
        #    Speculative-decode integration will pass in the persistent
        #    paged cache + a non-zero write_pos (see
        #    spec_decoding_project_plan.md and kernel_monkey_patching_plan.md).
        if past_key_value is not None:
            raise NotImplementedError(
                "CustomQwen2Attention: persistent KV cache path not wired up yet")
        cache_k = torch.empty_like(k)
        cache_v = torch.empty_like(v)
        custom_ops.kv_write_forward(k, v, cache_k, cache_v, 0)

        # 5) Fused causal SDPA, GQA-aware. cur_len = S since this scaffold
        #    is prefill-only (no history).
        o = custom_ops.fused_attn_forward(
            q, cache_k, cache_v, S, self.softmax_scale)

        # 6) Output projection. Mirror HF's reshape: transpose heads back
        #    to the inner axis, flatten to (M, H_q).
        y = custom_ops.o_proj_forward(
            o.transpose(1, 2).contiguous().view(M, H_q), self.W_o)

        return y.view(B, S, self.hidden_size)


def patch_model(model):
    """Replace every layer.self_attn with CustomQwen2Attention."""
    n = 0
    for layer in model.model.layers:
        layer.self_attn = CustomQwen2Attention(layer.self_attn)
        n += 1
    print(f"✅ Patched {n} attention layers with CustomQwen2Attention.")
