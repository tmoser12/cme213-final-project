"""Qwen2 decoder executor — composes target CUDA kernels into a full forward pass."""

from __future__ import annotations

import math

import torch

from runtime.buffers import RuntimeBuffers
from runtime.core.config import RuntimeConfig
from runtime.executor_graph import GraphExecutorMixin, _same_device
from runtime.speculative.types import ForwardMode, MAX_VERIFY_GAMMA, MAX_VERIFY_SEQ_LEN

# Attention kernels are templated on head_dim per role: target=128 (7B), draft=64 (0.5B).
_ATTN_HEAD_DIM = {"target": 128, "draft": 64}


def _import_kernels(kernel_set: str = "target"):
    if kernel_set not in _ATTN_HEAD_DIM:
        raise ValueError(
            f"unsupported kernel_set: {kernel_set!r} (expected one of {sorted(_ATTN_HEAD_DIM)})"
        )
    import importlib

    base = f"runtime.production_kernels.{kernel_set}"
    attn_ops = importlib.import_module(f"{base}.attention.ops")
    embedding = importlib.import_module(f"{base}.embedding.ops")
    residual = importlib.import_module(f"{base}.residual_ops.ops")
    rmsnorm = importlib.import_module(f"{base}.rmsnorm.ops")
    swiglu = importlib.import_module(f"{base}.swiglu.ops")

    return {
        "attn": attn_ops,
        "embedding_forward": embedding.embedding_forward,
        "rmsnorm_forward": rmsnorm.forward,
        "residual_add_forward": residual.residual_add_forward,
        "swiglu_forward": swiglu.swiglu_forward,
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


class Qwen2Executor(GraphExecutorMixin):
    """
    Config-driven decoder loop over pre-built CUDA kernel ops.

    Forward stages use explicit ``ForwardMode`` (PREFILL / VERIFY / DECODE).
    VERIFY is speculative decoding verification (S=γ, ``small_q_attn_forward``),
    not standard single-token decode.
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        weights: dict[str, torch.Tensor],
        buffers: RuntimeBuffers,
        *,
        kernel_set: str | None = None,
        use_cuda_graph: bool = False,
    ) -> None:
        cfg.validate()
        # Default the kernel set from the config (target=7B, draft=0.5B).
        kernel_set = kernel_set if kernel_set is not None else cfg.kernel_set
        expected_head_dim = _ATTN_HEAD_DIM.get(kernel_set)
        if expected_head_dim is None:
            raise ValueError(f"unsupported kernel_set: {kernel_set!r}")
        if cfg.head_dim != expected_head_dim:
            raise ValueError(
                f"kernel_set={kernel_set!r} attention kernels require head_dim="
                f"{expected_head_dim}; got {cfg.head_dim} ({cfg.name})."
            )
        if buffers.cfg.name != cfg.name:
            raise ValueError("buffers were allocated for a different config")
        if buffers.batch < 1:
            raise ValueError("invalid buffer batch size")

        self.cfg = cfg
        self.kernel_set = kernel_set
        self.weights = weights
        self.buffers = buffers
        self._ops = _import_kernels(kernel_set)
        self._softmax_scale = 1.0 / math.sqrt(cfg.head_dim)
        self._cache_pos = 0
        self._p1_logits: torch.Tensor | None = None
        self._pending_bonus: int | None = None

        # CUDA-graph capture/replay state (decode + verify) lives in the mixin.
        self._init_graph_state(use_cuda_graph)

        self._qkv_weights: list[torch.Tensor] = []
        self._qkv_bias: list[torch.Tensor] = []
        for layer in range(cfg.num_hidden_layers):
            w, b = _stack_qkv(weights, layer)
            self._qkv_weights.append(w)
            self._qkv_bias.append(b)

    @property
    def device(self) -> torch.device:
        return self.buffers.device

    @property
    def cache_pos(self) -> int:
        """Number of tokens written to KV cache (host mirror)."""
        return self._cache_pos

    @property
    def p1_logits(self) -> torch.Tensor | None:
        """Target p_1 from last ``prefill()`` or ``commit_pending_bonus()`` — ``[vocab]``."""
        return self._p1_logits

    @property
    def pending_bonus_token(self) -> int | None:
        """Bonus token deferred for ``commit_pending_bonus()`` (MPI-friendly path)."""
        return self._pending_bonus

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
        self._p1_logits = None
        self._pending_bonus = None

    def defer_bonus_token(self, token_id: int) -> None:
        """
        Queue a sampled bonus for the next bundled ``verify_gamma`` call.

        After sample + rollback, MPI can notify the draft immediately. The next
        target forward prepends this token before the γ draft ids (one VERIFY,
        not a separate S=1 commit). Call ``commit_pending_bonus()`` only after
        the final generation step.
        """
        if self._pending_bonus is not None:
            raise RuntimeError("bonus already pending; consume or flush before deferring another")
        self._pending_bonus = int(token_id)

    def take_pending_bonus(self) -> int | None:
        """Pop deferred bonus for bundled verify, or ``None`` if first iter after prefill."""
        token = self._pending_bonus
        self._pending_bonus = None
        return token

    def commit_pending_bonus(self) -> torch.Tensor | None:
        """
        Run ``decode_step`` on a deferred bonus after the **final** iteration.

        Normal iterations bundle the bonus into the next ``verify_gamma`` instead.
        Returns logits ``[batch, 1, vocab]`` or ``None`` if nothing was pending.
        """
        if self._pending_bonus is None:
            return None
        token_id = torch.tensor(
            [self._pending_bonus], dtype=torch.int64, device=self.device
        )
        self._pending_bonus = None
        logits = self.decode_step(token_id)
        self._p1_logits = logits[0, 0, :].detach()
        return logits

    def rollback_cache(self, to_pos: int) -> None:
        """
        Rewind sequence cursor after speculative verify + partial accept.

        Stale KV in rejected slots is ignored on subsequent forwards (``cur_len``
        bounds attention). No GPU zero-fill required.
        """
        if to_pos < 0 or to_pos > self.buffers.max_seq_len:
            raise ValueError(f"to_pos {to_pos} out of range")
        self._cache_pos = to_pos
        self.buffers.cache_position.fill_(to_pos)

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
        mode: ForwardMode,
        static_attn: bool = False,
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

        if static_attn:
            # Graph-capturable path: positions come from device scalars and
            # cos/sin from static buffers (refreshed before each replay via
            # `_static_ctx`), so the captured graph isn't frozen at the
            # capture-time position. S=1 -> decode kernel; S>1 -> verify kernel.
            cos, sin, cur_len = self._static_ctx
            ops.rope_kv_write_forward_dev(k, v, cache_k, cache_v, buf.cache_position, cos, sin)
            if seq_len == 1:
                attn_ctx = ops.decode_attn_forward_dev(
                    q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                )
            else:
                attn_ctx = ops.small_q_attn_forward_dev(
                    q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                )
        else:
            write_pos = self._cache_pos
            cos, sin = buf.rope_embeddings(write_pos, seq_len)

            ops.rope_kv_write_forward(k, v, cache_k, cache_v, write_pos, cos, sin)

            if mode == ForwardMode.PREFILL:
                cur_len = seq_len
                attn_ctx = ops.fused_attn_forward(
                    q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                )
            elif mode == ForwardMode.VERIFY:
                cur_len = write_pos + seq_len
                if seq_len == 1:
                    attn_ctx = ops.decode_attn_forward(
                        q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                    )
                else:
                    attn_ctx = ops.small_q_attn_forward(
                        q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                    )
            elif mode == ForwardMode.DECODE:
                if seq_len != 1:
                    raise ValueError("DECODE mode requires seq_len == 1")
                cur_len = write_pos + seq_len
                attn_ctx = ops.decode_attn_forward(
                    q, cache_k, cache_v, cur_len, self._softmax_scale, cos, sin
                )
            else:
                raise ValueError(f"unknown ForwardMode: {mode!r}")

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
        mode: ForwardMode | None = None,
        decode: bool = False,
    ) -> torch.Tensor:
        """
        Run one decoder layer (parity / debugging).

        Does not advance ``cache_position`` — use ``prefill`` / ``verify_gamma`` /
        ``decode_step`` for full forwards.
        """
        if mode is None:
            mode = ForwardMode.DECODE if decode else ForwardMode.PREFILL
        return self._run_decoder_layer(hidden, layer, seq_len, mode=mode)

    def _run_decoder_layer(
        self,
        hidden: torch.Tensor,
        layer: int,
        seq_len: int,
        *,
        mode: ForwardMode,
        static_attn: bool = False,
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
                attn_out = self._run_attention(
                    normed, layer, seq_len, mode=mode, static_attn=static_attn
                )
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

    def _forward_stack(
        self,
        hidden: torch.Tensor,
        seq_len: int,
        *,
        mode: ForwardMode,
        static_attn: bool = False,
    ) -> torch.Tensor:
        for layer in range(self.cfg.num_hidden_layers):
            hidden = self._run_decoder_layer(
                hidden, layer, seq_len, mode=mode, static_attn=static_attn
            )

        hidden = self._ops["rmsnorm_forward"](
            hidden,
            self.weights["model.norm.weight"],
            self.cfg.rms_norm_eps,
        )
        return self._ops["lm_head_forward"](hidden, self._lm_head_weight())

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        ``ForwardMode.PREFILL``: process prompt ``[batch, seq]``; reset KV cache.

        Stores ``p1_logits`` (p_1) from the last prompt position. Returns full logits.
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

        logits = self._forward_stack(hidden, seq_len, mode=ForwardMode.PREFILL)
        self._p1_logits = logits[0, -1, :].detach()
        self._advance_cache_pos(seq_len)
        return logits

    def verify_gamma(
        self,
        draft_token_ids: torch.Tensor,
        *,
        leading_bonus: int | None = None,
        commit_bonus: int | None = None,
    ) -> torch.Tensor:
        """
        ``ForwardMode.VERIFY``: evaluate γ draft tokens against cached prefix.

        **First post-prefill iter:** ``leading_bonus=None`` — forward over the γ
        drafts only (S=γ). Returns ``[batch, γ, vocab]`` = ``p_2 … p_{γ+1}``;
        combine with ``p1_logits`` for sampling.

        **Subsequent iters:** pass the deferred bonus from the prior iter as
        ``leading_bonus``. One forward over ``[bonus, d_1, …, d_γ]`` (S=γ+1).
        Returns ``[batch, γ+1, vocab]`` = ``p_1 … p_{γ+1}`` for accept/reject on
        the γ drafts only (bonus is written to KV but not re-verified).

        ``commit_bonus`` is a deprecated alias for ``leading_bonus``.
        """
        if commit_bonus is not None and leading_bonus is not None:
            raise ValueError("pass only one of leading_bonus or commit_bonus")
        if leading_bonus is None:
            leading_bonus = commit_bonus
        if not _same_device(draft_token_ids.device, self.device):
            raise ValueError(
                f"draft_token_ids on {draft_token_ids.device}, expected {self.device}"
            )
        if self._p1_logits is None:
            raise RuntimeError("verify_gamma requires prefill() first")
        if draft_token_ids.dim() == 1:
            draft_token_ids = draft_token_ids.unsqueeze(0)
        if draft_token_ids.dim() != 2 or draft_token_ids.shape[0] != self.buffers.batch:
            raise ValueError("draft_token_ids must be [batch, γ]")

        gamma = draft_token_ids.shape[1]
        if gamma < 1 or gamma > MAX_VERIFY_GAMMA:
            raise ValueError(f"γ must be in [1, {MAX_VERIFY_GAMMA}], got {gamma}")

        if leading_bonus is not None:
            seq_len = gamma + 1
            if seq_len > MAX_VERIFY_SEQ_LEN:
                raise ValueError(
                    f"bundled verify S=γ+1={seq_len} exceeds kernel limit "
                    f"{MAX_VERIFY_SEQ_LEN}"
                )
            bonus = torch.full(
                (draft_token_ids.shape[0], 1),
                int(leading_bonus),
                dtype=torch.int64,
                device=draft_token_ids.device,
            )
            input_ids = torch.cat([bonus, draft_token_ids], dim=1)
        else:
            seq_len = gamma
            input_ids = draft_token_ids

        self._validate_input_ids(input_ids, seq_len)

        hidden = self._ops["embedding_forward"](
            input_ids,
            self.weights["model.embed_tokens.weight"],
        )

        logits = self._forward_stack(hidden, seq_len, mode=ForwardMode.VERIFY)
        self._advance_cache_pos(seq_len)
        return logits

    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        ``ForwardMode.DECODE``: one committed token (S=1).

        Returns logits ``[batch, 1, vocab]``.
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

        logits = self._forward_stack(hidden, seq_len, mode=ForwardMode.DECODE)
        self._advance_cache_pos(1)
        return logits

    def greedy_extend(
        self,
        input_ids: torch.Tensor,
        n_new_tokens: int,
        *,
        use_cuda_graph: bool | None = None,
    ) -> torch.Tensor:
        """
        Greedy-decode ``n_new_tokens`` after ``input_ids`` and return the full sequence.

        Returns int64 tensor ``[batch, prompt_len + n_new_tokens]``. First new token
        is taken from prefill logits; later tokens use ``decode_step`` (or
        ``decode_step_graph`` when ``use_cuda_graph`` — defaults to the executor's
        ``use_cuda_graph`` flag).
        """
        if n_new_tokens < 0:
            raise ValueError("n_new_tokens must be non-negative")
        if n_new_tokens == 0:
            return input_ids

        graph = self.use_cuda_graph if use_cuda_graph is None else use_cuda_graph
        decode = self.decode_step_graph if graph else self.decode_step

        out = input_ids[0].tolist()
        logits = self.prefill(input_ids)
        for i in range(n_new_tokens):
            next_t = int(logits[0, -1].argmax().item())
            out.append(next_t)
            if i < n_new_tokens - 1:
                last = torch.tensor([next_t], dtype=torch.int64, device=input_ids.device)
                logits = decode(last)
        return torch.tensor([out], dtype=torch.int64, device=input_ids.device)
