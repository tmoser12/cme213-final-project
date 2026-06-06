"""Target-model attention kernels."""

from runtime.production_kernels.target.attention import ops

__all__ = [
    "ops",
    "qkv_proj_forward",
    "rope_kv_write_forward",
    "fused_attn_forward",
    "decode_attn_forward",
    "small_q_attn_forward",
    "o_proj_forward",
]

qkv_proj_forward = ops.qkv_proj_forward
rope_kv_write_forward = ops.rope_kv_write_forward
fused_attn_forward = ops.fused_attn_forward
decode_attn_forward = ops.decode_attn_forward
small_q_attn_forward = ops.small_q_attn_forward
o_proj_forward = ops.o_proj_forward
