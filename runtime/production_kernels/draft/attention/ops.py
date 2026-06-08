"""Python host ABI for draft attention (AOT extension in this directory)."""

from __future__ import annotations

from typing import Optional

import torch

_EXT_MODULE = "runtime.production_kernels.draft.attention.draft_attention_ops"
_BUILD_HINT = "Run: bash scripts/build_kernels.sh draft attention"


def _load_ext():
    try:
        from . import draft_attention_ops as ext
    except ImportError as exc:
        raise ImportError(_BUILD_HINT) from exc
    path = getattr(ext, "__file__", "")
    if "torch_extensions" in path:
        raise ImportError(
            "Loaded JIT extension instead of AOT build. " + _BUILD_HINT
        )
    if "production_kernels/draft/attention" not in path.replace("\\", "/"):
        raise ImportError(
            f"Expected AOT extension under draft/attention/, got: {path!r}. "
            + _BUILD_HINT
        )
    return ext


def qkv_proj_forward(x, w_qkv, b_qkv):
    return _load_ext().qkv_proj_forward(x, w_qkv, b_qkv)


def rope_kv_write_forward(new_k, new_v, cache_k, cache_v, write_pos, cos, sin):
    _load_ext().rope_kv_write_forward(new_k, new_v, cache_k, cache_v, write_pos, cos, sin)


def rope_kv_write_forward_dev(new_k, new_v, cache_k, cache_v, write_pos, cos, sin):
    """write_pos is a 0-d int64 CUDA tensor (read on device) — for CUDA-graph capture."""
    _load_ext().rope_kv_write_forward_dev(new_k, new_v, cache_k, cache_v, write_pos, cos, sin)


def fused_attn_forward(
    q,
    cache_k,
    cache_v,
    cur_len,
    softmax_scale,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
):
    return _load_ext().fused_attn_forward(
        q, cache_k, cache_v, cur_len, softmax_scale, cos, sin
    )


def decode_attn_forward(
    q,
    cache_k,
    cache_v,
    cur_len,
    softmax_scale,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
):
    return _load_ext().decode_attn_forward(
        q, cache_k, cache_v, cur_len, softmax_scale, cos, sin
    )


def small_q_attn_forward(
    q,
    cache_k,
    cache_v,
    cur_len,
    softmax_scale,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
):
    return _load_ext().small_q_attn_forward(
        q, cache_k, cache_v, cur_len, softmax_scale, cos, sin
    )


def decode_attn_forward_dev(
    q,
    cache_k,
    cache_v,
    cur_len,
    softmax_scale,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
):
    """cur_len is a 0-d int64 CUDA tensor (read on device) — for CUDA-graph capture."""
    return _load_ext().decode_attn_forward_dev(
        q, cache_k, cache_v, cur_len, softmax_scale, cos, sin
    )


def small_q_attn_forward_dev(
    q,
    cache_k,
    cache_v,
    cur_len,
    softmax_scale,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
):
    """cur_len is a 0-d int64 CUDA tensor (read on device) — for CUDA-graph capture."""
    return _load_ext().small_q_attn_forward_dev(
        q, cache_k, cache_v, cur_len, softmax_scale, cos, sin
    )


def o_proj_forward(x, w_o):
    return _load_ext().o_proj_forward(x, w_o)


EXTENSION_MODULE = _EXT_MODULE
