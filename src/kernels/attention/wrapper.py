import math
import torch
import torch.nn as nn

from src.kernels.attention.jit import load_attention_ops

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
        # Scaffold only. Real implementation will:
        #   1) qkv = custom_ops.qkv_proj_forward(x.view(M, H), W_qkv, b_qkv)
        #   2) split qkv -> q/k/v, reshape to [B, H_*, S, D]
        #   3) custom_ops.rope_forward(q, k, cos, sin)
        #   4) custom_ops.kv_write_forward(k, v, cache_k, cache_v, write_pos)
        #   5) o = custom_ops.fused_attn_forward(q, cache_k, cache_v,
        #                                       write_pos + S, softmax_scale)
        #   6) y = custom_ops.o_proj_forward(o.view(M, H_q), W_o)
        # past_key_value layout (HF DynamicCache vs project paged cache) is
        # TBD per spec_decoding_project_plan.md.
        raise NotImplementedError("CustomQwen2Attention.forward: scaffold only")


def patch_model(model):
    """Replace every layer.self_attn with CustomQwen2Attention."""
    n = 0
    for layer in model.model.layers:
        layer.self_attn = CustomQwen2Attention(layer.self_attn)
        n += 1
    print(f"✅ Patched {n} attention layers with CustomQwen2Attention.")
