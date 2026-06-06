"""Qwen2 decoder executor — composes target CUDA kernels into a full forward pass."""

from __future__ import annotations

import math

import torch

from runtime.buffers import RuntimeBuffers
from runtime.core.config import RuntimeConfig

_ATTN_HEAD_DIM = 128  # fused/decode attention kernels are templated on D=128 (7B)


def _same_device(a: torch.device, b: torch.device) -> bool:
    """``cuda`` and ``cuda:0`` compare equal."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)


def _import_kernels(kernel_set: str = "target"):
    if kernel_set != "target":
        raise ValueError(f"unsupported kernel_set: {kernel_set!r} (only 'target' is wired)")
    from runtime.production_kernels.target.attention import ops as attn_ops
    from runtime.production_kernels.target.embedding.ops import embedding_forward
    from runtime.production_kernels.target.residual_ops.ops import (
        lm_head_forward,
        residual_add_forward,
    )
    from runtime.production_kernels.target.rmsnorm.ops import forward as rmsnorm_forward
    from runtime.production_kernels.target.swiglu.ops import swiglu_forward

    return {
        "attn": attn_ops,
        "embedding_forward": embedding_forward,
        "rmsnorm_forward": rmsnorm_forward,
        "residual_add_forward": residual_add_forward,
        "swiglu_forward": swiglu_forward,
        "lm_head_forward": lm_head_forward,
    }


def _stack_qkv(
    weights: dict[str, torch.Tensor],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = f"model.layers.{layer}.self_attn"
    w_qkv = torch.cat(
        [
            weights[f"{p}.q_proj.weight"],
            weights[f"{p}.k_proj.weight"],
            weights[f"{p}.v_proj.weight"],
        ],
        dim=0,
    ).contiguous()
    b_qkv = torch.cat(
        [
            weights[f"{p}.q_proj.bias"],
            weights[f"{p}.k_proj.bias"],
            weights[f"{p}.v_proj.bias"],
        ],
        dim=0,
    ).contiguous()
    return w_qkv, b_qkv


class Qwen2Executor:
    """
    Config-driven decoder loop over pre-built CUDA kernel ops.

    Host mirrors ``cache_position`` for ``write_pos`` / ``cur_len`` ints passed to
    pybind (no ``.item()`` in the hot path). The device scalar stays in lockstep
    via ``fill_`` / ``add_`` for future CUDA Graph capture.
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        weights: dict[str, torch.Tensor],
        buffers: RuntimeBuffers,
        *,
        kernel_set: str = "target",
    ) -> None:
        cfg.validate()
        if cfg.head_dim != _ATTN_HEAD_DIM:
            raise ValueError(
                f"attention kernels require head_dim={_ATTN_HEAD_DIM}; "
                f"got {cfg.head_dim} ({cfg.name}). Use the 7B config."
            )
        if buffers.cfg.name != cfg.name:
            raise ValueError("buffers were allocated for a different config")
        if buffers.batch < 1:
            raise ValueError("invalid buffer batch size")

        self.cfg = cfg
        self.weights = weights
        self.buffers = buffers
        self._ops = _import_kernels(kernel_set)
        self._softmax_scale = 1.0 / math.sqrt(cfg.head_dim)
        self._cache_pos = 0

        self._qkv_weights: list[torch.Tensor] = []
        self._qkv_bias: list[torch.Tensor] = []
        for layer in range(cfg.num_hidden_layers):
            w, b = _stack_qkv(weights, layer)
            self._qkv_weights.append(w)
            self._qkv_bias.append(b)

    @property
    def device(self) -> torch.device:
        return self.buffers.device

    def _lm_head_weight(self) -> torch.Tensor:
        if self.cfg.tie_word_embeddings:
            return self.weights["model.embed_tokens.weight"]
        return self.weights["lm_head.weight"]

    def reset_kv_cache(self) -> None:
        """Zero KV caches and reset the sequence cursor (start of a new prompt)."""
        self.buffers.kv_cache_k.zero_()
        self.buffers.kv_cache_v.zero_()
        self.buffers.reset_cache_position(0)
        self._cache_pos = 0

    def _advance_cache_pos(self, n_tokens: int) -> None:
        self._cache_pos += n_tokens
        if n_tokens == 1:
            self.buffers.cache_position.add_(1)
        else:
            self.buffers.cache_position.fill_(self._cache_pos)

    def _validate_input_ids(self, input_ids: torch.Tensor, seq_len: int) -> None:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2-D [batch, seq]")
        if input_ids.shape[0] != self.buffers.batch:
            raise ValueError(
                f"batch mismatch: input_ids {input_ids.shape[0]} vs buffers {self.buffers.batch}"
            )
        if seq_len > self.buffers.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds buffer max_seq_len {self.buffers.max_seq_len}"
            )
        if self._cache_pos + seq_len > self.buffers.max_seq_len:
            raise ValueError(
                f"KV cache would overflow: pos={self._cache_pos} + {seq_len} "
                f"> max_seq_len={self.buffers.max_seq_len}"
            )

    def _run_attention(
        self,
        x: torch.Tensor,
        layer: int,
        seq_len: int,
        *,
        decode: bool,
    ) -> torch.Tensor:
        cfg = self.cfg
        buf = self.buffers
        ops = self._ops["attn"]
        batch = buf.batch
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        d = cfg.head_dim
        hq = cfg.hidden_size
        hkv = cfg.kv_dim

        flat = x.reshape(batch * seq_len, cfg.hidden_size)
        qkv = ops.qkv_proj_forward(flat, self._qkv_weights[layer], self._qkv_bias[layer])

        q = qkv[:, :hq].reshape(batch, seq_len, nh, d).transpose(1, 2).contiguous()
        k = qkv[:, hq : hq + hkv].reshape(batch, seq_len, nkv, d).transpose(1, 2).contiguous()
        v = qkv[:, hq + hkv :].reshape(batch, seq_len, nkv, d).transpose(1, 2).contiguous()

        cache_k = buf.kv_cache_k_layer(layer)
        cache_v = buf.kv_cache_v_layer(layer)
        write_pos = self._cache_pos
        cos, sin = buf.rope_embeddings(write_pos, seq_len)

        ops.rope_kv_write_forward(k, v, cache_k, cache_v, write_pos, cos, sin)

        if decode:
            cur_len = write_pos + seq_len
            attn_ctx = ops.decode_attn_forward(
                q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
            )
        else:
            cur_len = seq_len
            attn_ctx = ops.fused_attn_forward(
                q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
            )

        w_o = self.weights[f"model.layers.{layer}.self_attn.o_proj.weight"]
        flat_ctx = attn_ctx.transpose(1, 2).reshape(batch * seq_len, hq)
        out = ops.o_proj_forward(flat_ctx, w_o)
        return out.reshape(batch, seq_len, cfg.hidden_size)

    def run_decoder_layer(
        self,
        hidden: torch.Tensor,
        layer: int,
        seq_len: int,
        *,
        decode: bool = False,
    ) -> torch.Tensor:
        """
        Run one decoder layer (parity / debugging).

        Caller must set ``_cache_pos`` and per-layer KV state appropriately.
        Does not advance ``cache_position`` — use ``prefill`` / ``decode_step`` for full forwards.
        """
        return self._run_decoder_layer(hidden, layer, seq_len, decode=decode)

    def _run_decoder_layer(
        self,
        hidden: torch.Tensor,
        layer: int,
        seq_len: int,
        *,
        decode: bool,
    ) -> torch.Tensor:
        cfg = self.cfg
        ops = self._ops
        prefix = f"model.layers.{layer}"
        w = self.weights

        residual_1 = hidden
        residual_2: torch.Tensor | None = None
        attn_out: torch.Tensor | None = None
        mlp_out: torch.Tensor | None = None
        normed: torch.Tensor | None = None

        for step in cfg.layer_order:
            if step == "input_rmsnorm":
                normed = ops["rmsnorm_forward"](
                    hidden,
                    w[f"{prefix}.input_layernorm.weight"],
                    cfg.rms_norm_eps,
                )
            elif step == "attention":
                if normed is None:
                    raise RuntimeError("layer_order: attention before input_rmsnorm")
                attn_out = self._run_attention(normed, layer, seq_len, decode=decode)
            elif step == "residual_add":
                if residual_2 is None:
                    if attn_out is None:
                        raise RuntimeError("layer_order: residual_add before attention")
                    hidden = ops["residual_add_forward"](residual_1, attn_out)
                    residual_2 = hidden
                else:
                    if mlp_out is None:
                        raise RuntimeError("layer_order: second residual_add before swiglu_mlp")
                    hidden = ops["residual_add_forward"](residual_2, mlp_out)
            elif step == "post_attn_rmsnorm":
                normed = ops["rmsnorm_forward"](
                    hidden,
                    w[f"{prefix}.post_attention_layernorm.weight"],
                    cfg.rms_norm_eps,
                )
            elif step == "swiglu_mlp":
                if normed is None:
                    raise RuntimeError("layer_order: swiglu_mlp before post_attn_rmsnorm")
                mlp = f"{prefix}.mlp"
                mlp_out = ops["swiglu_forward"](
                    normed,
                    w[f"{mlp}.gate_proj.weight"],
                    w[f"{mlp}.up_proj.weight"],
                    w[f"{mlp}.down_proj.weight"],
                )
            else:
                raise ValueError(f"unknown layer_order step: {step!r}")

        return hidden

    def _forward_stack(self, hidden: torch.Tensor, seq_len: int, *, decode: bool) -> torch.Tensor:
        cfg = self.cfg
        ops = self._ops

        for layer in range(cfg.num_hidden_layers):
            hidden = self._run_decoder_layer(hidden, layer, seq_len, decode=decode)

        hidden = ops["rmsnorm_forward"](
            hidden,
            self.weights["model.norm.weight"],
            cfg.rms_norm_eps,
        )
        return ops["lm_head_forward"](hidden, self._lm_head_weight())

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Process a prompt of shape ``[batch, seq]`` and return logits ``[batch, seq, vocab]``.

        Resets the KV cache. After return, ``cache_position`` equals ``seq``.
        """
        if not _same_device(input_ids.device, self.device):
            raise ValueError(f"input_ids on {input_ids.device}, expected {self.device}")

        seq_len = input_ids.shape[1]
        self._validate_input_ids(input_ids, seq_len)
        self.reset_kv_cache()

        hidden = self._ops["embedding_forward"](
            input_ids,
            self.weights["model.embed_tokens.weight"],
        )

        logits = self._forward_stack(hidden, seq_len, decode=False)
        self._advance_cache_pos(seq_len)
        return logits

    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Run one decode step for a single new token per batch row.

        ``token_id`` shape ``[batch]`` (int64, on device). Returns logits ``[batch, 1, vocab]``.
        """
        if not _same_device(token_id.device, self.device):
            raise ValueError(f"token_id on {token_id.device}, expected {self.device}")
        if token_id.dim() == 0:
            token_id = token_id.unsqueeze(0)
        if token_id.dim() != 1 or token_id.shape[0] != self.buffers.batch:
            raise ValueError("token_id must be [batch]")

        seq_len = 1
        input_ids = token_id.unsqueeze(1)
        self._validate_input_ids(input_ids, seq_len)

        hidden = self._ops["embedding_forward"](
            input_ids,
            self.weights["model.embed_tokens.weight"],
        )

        logits = self._forward_stack(hidden, seq_len, decode=True)
        self._advance_cache_pos(1)
        return logits

    def greedy_extend(
        self,
        input_ids: torch.Tensor,
        n_new_tokens: int,
    ) -> torch.Tensor:
        """
        Greedy-decode ``n_new_tokens`` after ``input_ids`` and return the full sequence.

        Returns int64 tensor ``[batch, prompt_len + n_new_tokens]``. First new token
        is taken from prefill logits; later tokens use ``decode_step``.
        """
        if n_new_tokens < 0:
            raise ValueError("n_new_tokens must be non-negative")
        if n_new_tokens == 0:
            return input_ids

        out = input_ids[0].tolist()
        logits = self.prefill(input_ids)
        for i in range(n_new_tokens):
            next_t = int(logits[0, -1].argmax().item())
            out.append(next_t)
            if i < n_new_tokens - 1:
                last = torch.tensor([next_t], dtype=torch.int64, device=input_ids.device)
                logits = self.decode_step(last)
        return torch.tensor([out], dtype=torch.int64, device=input_ids.device)
