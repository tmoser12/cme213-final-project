"""Qwen2 decoder executor — composes target CUDA kernels into a full forward pass."""

from __future__ import annotations

import importlib
import math

import torch

from runtime.buffers import RuntimeBuffers
from runtime.core.config import KERNEL_SETS, RuntimeConfig
from runtime.nvtx import rng

# Head dim each compiled attention suite is templated on (constexpr D in kernel.cu).
# The literal lived in the executor before; it is a property of the kernel set,
# keyed by cfg.kernel_set — not a model dim (cfg.head_dim is derived).
_KERNEL_SET_HEAD_DIM = {"target": 128, "draft": 64}


def _same_device(a: torch.device, b: torch.device) -> bool:
    """``cuda`` and ``cuda:0`` compare equal."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)


def _import_kernels(kernel_set: str):
    """Load the op callables for one kernel suite (``target`` or ``draft``)."""
    if kernel_set not in KERNEL_SETS:
        raise ValueError(f"unsupported kernel_set: {kernel_set!r} (expected one of {KERNEL_SETS})")
    base = f"runtime.production_kernels.{kernel_set}"
    attn_ops = importlib.import_module(f"{base}.attention").ops
    embedding_forward = importlib.import_module(f"{base}.embedding.ops").embedding_forward
    residual = importlib.import_module(f"{base}.residual_ops.ops")
    rmsnorm_forward = importlib.import_module(f"{base}.rmsnorm.ops").forward
    swiglu_forward = importlib.import_module(f"{base}.swiglu.ops").swiglu_forward

    return {
        "attn": attn_ops,
        "embedding_forward": embedding_forward,
        "rmsnorm_forward": rmsnorm_forward,
        "residual_add_forward": residual.residual_add_forward,
        "swiglu_forward": swiglu_forward,
        "lm_head_forward": residual.lm_head_forward,
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
        kernel_set: str | None = None,
    ) -> None:
        cfg.validate()
        # Default to the suite named by the config; kwarg is an explicit override.
        resolved_set = kernel_set if kernel_set is not None else cfg.kernel_set
        expected_d = _KERNEL_SET_HEAD_DIM.get(resolved_set)
        if expected_d is None:
            raise ValueError(
                f"unsupported kernel_set: {resolved_set!r} (expected one of {KERNEL_SETS})"
            )
        if cfg.head_dim != expected_d:
            raise ValueError(
                f"{resolved_set!r} attention kernels require head_dim={expected_d}; "
                f"got {cfg.head_dim} ({cfg.name}). Check the config's kernel_set."
            )
        if buffers.cfg.name != cfg.name:
            raise ValueError("buffers were allocated for a different config")
        if buffers.batch < 1:
            raise ValueError("invalid buffer batch size")

        self.cfg = cfg
        self.kernel_set = resolved_set
        self.weights = weights
        self.buffers = buffers
        self._ops = _import_kernels(resolved_set)
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
        nh = cfg.num_attention_heads # 28
        nkv = cfg.num_key_value_heads # 4
        d = cfg.head_dim # 128
        hq = cfg.hidden_size # q_heads * head_dim = 28 * 128 = 3584
        hkv = cfg.kv_dim # kv_heads * head_dim = 4 * 128 = 512

        flat = x.reshape(batch * seq_len, cfg.hidden_size)  # [B*S, hq]
        # qkv fused projection: [B*S, hq] -> [B*S, hq + 2*hkv]  (3584 + 2*512 = 4608)
        with rng("qkv_proj"):
            qkv = ops.qkv_proj_forward(flat, self._qkv_weights[layer], self._qkv_bias[layer])

        # split + reshape to per-head layout [B, heads, S, d]
        q = qkv[:, :hq].reshape(batch, seq_len, nh, d).transpose(1, 2).contiguous()  # [B, nh=28, S, d=128]
        k = qkv[:, hq : hq + hkv].reshape(batch, seq_len, nkv, d).transpose(1, 2).contiguous()  # [B, nkv=4, S, d=128]
        v = qkv[:, hq + hkv :].reshape(batch, seq_len, nkv, d).transpose(1, 2).contiguous()  # [B, nkv=4, S, d=128]

        cache_k = buf.kv_cache_k_layer(layer)  # [B, nkv=4, max_seq, d=128]
        cache_v = buf.kv_cache_v_layer(layer)  # [B, nkv=4, max_seq, d=128]
        write_pos = self._cache_pos
        cos, sin = buf.rope_embeddings(write_pos, seq_len)  # cos/sin: [B, S, d=128]

        # rope-rotates k, then writes k,v into cache at [.., write_pos:write_pos+S, ..] (in place)
        with rng("rope_kv_write"):
            ops.rope_kv_write_forward(k, v, cache_k, cache_v, write_pos, cos, sin)

        if decode:
            cur_len = write_pos + seq_len  # full context length incl. cached tokens
            # q [B, nh, S, d] attends over cache[:cur_len] -> ctx [B, nh, S, d=128]
            with rng("decode_attn"):
                attn_ctx = ops.decode_attn_forward(
                    q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                )
        else:
            cur_len = seq_len  # prefill: context == current sequence
            # q [B, nh, S, d] causal attn over cache[:cur_len] -> ctx [B, nh, S, d=128]
            with rng("fused_attn"):
                attn_ctx = ops.fused_attn_forward(
                    q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                )

        w_o = self.weights[f"model.layers.{layer}.self_attn.o_proj.weight"]
        # back to token-major and flatten heads: [B, nh, S, d] -> [B, S, nh, d] -> [B*S, hq=3584]
        flat_ctx = attn_ctx.transpose(1, 2).reshape(batch * seq_len, hq)
        with rng("o_proj"):
            out = ops.o_proj_forward(flat_ctx, w_o)  # [B*S, hq] -> [B*S, hq=3584]
        return out.reshape(batch, seq_len, cfg.hidden_size)  # [B, S, hq=3584]

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
                with rng("input_rmsnorm"):
                    normed = ops["rmsnorm_forward"](
                        hidden,
                        w[f"{prefix}.input_layernorm.weight"],
                        cfg.rms_norm_eps,
                    )
            elif step == "attention":
                if normed is None:
                    raise RuntimeError("layer_order: attention before input_rmsnorm")
                with rng("attention"):
                    attn_out = self._run_attention(normed, layer, seq_len, decode=decode)
            elif step == "residual_add":
                with rng("residual_add"):
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
                with rng("post_attn_rmsnorm"):
                    normed = ops["rmsnorm_forward"](
                        hidden,
                        w[f"{prefix}.post_attention_layernorm.weight"],
                        cfg.rms_norm_eps,
                    )
            elif step == "swiglu_mlp":
                if normed is None:
                    raise RuntimeError("layer_order: swiglu_mlp before post_attn_rmsnorm")
                mlp = f"{prefix}.mlp"
                with rng("swiglu_mlp"):
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
            with rng(f"layer{layer}"):
                hidden = self._run_decoder_layer(hidden, layer, seq_len, decode=decode)

        with rng("final_rmsnorm"):
            hidden = ops["rmsnorm_forward"](
                hidden,
                self.weights["model.norm.weight"],
                cfg.rms_norm_eps,
            )
        with rng("lm_head"):
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

        with rng("prefill"):
            with rng("embedding"):
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

        with rng("decode"):
            with rng("embedding"):
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

