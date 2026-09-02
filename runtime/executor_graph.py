"""CUDA-graph capture/replay for the Qwen2 executor (decode + verify).

`GraphExecutorMixin` is folded into `Qwen2Executor`. It owns the static-buffer
forwards and the graph capture/replay, keeping `executor.py` focused on the eager
forward math. The methods here use the core executor's interface on ``self``:

    self.cfg, self.buffers, self.weights, self.device, self._ops,
    self._cache_pos, self._softmax_scale, self._p1_logits,
    self._forward_stack(...), self._advance_cache_pos(n), self._validate_input_ids(...)

and they set ``self._static_ctx = (cos, sin, cur_len)``, which the core's
``_run_attention(static_attn=True)`` reads. One decode graph serves all S=1 steps;
one VERIFY graph is captured per query length S (= γ or γ+1).

Concepts: docs/cuda_graphs_explained.md, cuda_graph_issues_and_concepts.md.
"""

from __future__ import annotations

import torch

from runtime.speculative.types import ForwardMode, MAX_VERIFY_GAMMA, MAX_VERIFY_SEQ_LEN


def _same_device(a: torch.device, b: torch.device) -> bool:
    """``cuda`` and ``cuda:0`` compare equal."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)


class GraphExecutorMixin:
    """Static-buffer forwards + CUDA-graph capture/replay for decode and verify."""

    def _init_graph_state(self, use_cuda_graph: bool) -> None:
        """Initialize graph state (called from ``Qwen2Executor.__init__``)."""
        # One decode graph serves all S=1 steps / prompts; one VERIFY graph per
        # query length S. Positions live in device scalars and KV/buffer addresses
        # are stable, so a captured graph is reused across prompts.
        self.use_cuda_graph = use_cuda_graph
        self._decode_graph: "torch.cuda.CUDAGraph | None" = None
        self._decode_graph_logits: torch.Tensor | None = None
        self._graph_pool = None
        # Active static-attention context (cos, sin, cur_len device scalar) read by
        # the static path of _run_attention; set by the prepare/forward step.
        self._static_ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # Per-S VERIFY graph state, keyed by seq_len S.
        self._verify_state: dict[int, dict] = {}

    # --- Static / graph-capturable decode (Phase 3) ------------------------

    def _prepare_decode_static(self, token_id: torch.Tensor) -> None:
        """
        Fill the static decode buffers for the current step. Runs OUTSIDE any
        captured region (the per-replay update step): writes the token and the
        device position scalars in place. RoPE is gathered inside the forward via
        ``buffers.static_rope()`` (device position) — no host slice/copy here.
        """
        buf = self.buffers
        buf.static_input_ids.copy_(token_id.reshape(buf.batch, 1))
        buf.cache_position.fill_(self._cache_pos)        # write_pos for this step
        buf.static_cur_len.fill_(self._cache_pos + 1)    # cur_len = write_pos + 1

    def _decode_forward_static(self) -> torch.Tensor:
        """
        S=1 decode forward reading ONLY static buffers + device-scalar (`_dev`)
        attention ops — no `torch.cat`, no per-step input tensors. This is the
        region captured into a CUDA graph. Returns logits ``[batch, 1, vocab]``.
        Assumes ``_prepare_decode_static`` ran first.
        """
        cos, sin = self.buffers.static_rope(1)   # gathered in-graph from cache_position
        self._static_ctx = (cos, sin, self.buffers.static_cur_len)
        hidden = self._ops["embedding_forward"](
            self.buffers.static_input_ids,
            self.weights["model.embed_tokens.weight"],
        )
        return self._forward_stack(hidden, 1, mode=ForwardMode.DECODE, static_attn=True)

    def decode_step_static(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        Eager equivalent of ``decode_step`` via the static/`_dev` decode path.

        Numerically identical to ``decode_step``; the building block the CUDA-graph
        decode captures and replays. Returns ``[batch, 1, vocab]``.
        """
        if not _same_device(token_id.device, self.device):
            raise ValueError(f"token_id on {token_id.device}, expected {self.device}")
        if token_id.dim() == 0:
            token_id = token_id.unsqueeze(0)
        if token_id.dim() != 1 or token_id.shape[0] != self.buffers.batch:
            raise ValueError("token_id must be [batch]")
        self._validate_input_ids(token_id.unsqueeze(1), 1)
        self._prepare_decode_static(token_id)
        logits = self._decode_forward_static()
        self._advance_cache_pos(1)
        return logits

    # --- CUDA-graph decode (Phase 4) ---------------------------------------

    def _capture_decode_graph(self) -> None:
        """
        Capture the static S=1 decode forward into a CUDA graph.

        Assumes the static buffers are already primed (``_prepare_decode_static``
        ran for the current step). Warms up on a side stream — needed so cuBLAS
        workspaces and the allocator are initialized before capture — then records
        ``_decode_forward_static``. Capture only *records*; the captured ``logits``
        buffer is not valid until the first ``replay()``.
        """
        # Side-stream warmup (KV writes here are idempotent: same token/position).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._decode_forward_static()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        pool = self._graph_pool or torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            logits = self._decode_forward_static()

        self._decode_graph = graph
        self._decode_graph_logits = logits
        self._graph_pool = pool

    def decode_step_graph(self, token_id: torch.Tensor) -> torch.Tensor:
        """
        One decode token via CUDA-graph replay — the launch-bound path collapsed
        to a single ``replay()``.

        Lazily captures the graph on first call (one warmup + capture), then for
        every step: update the static buffers in place, replay, advance. Returns
        ``[batch, 1, vocab]`` — the **live** captured output buffer, overwritten on
        the next call, so read/clone it before the next ``decode_step_graph``.
        """
        if not _same_device(token_id.device, self.device):
            raise ValueError(f"token_id on {token_id.device}, expected {self.device}")
        if token_id.dim() == 0:
            token_id = token_id.unsqueeze(0)
        if token_id.dim() != 1 or token_id.shape[0] != self.buffers.batch:
            raise ValueError("token_id must be [batch]")
        self._validate_input_ids(token_id.unsqueeze(1), 1)

        self._prepare_decode_static(token_id)
        if self._decode_graph is None:
            self._capture_decode_graph()
        self._decode_graph.replay()
        self._advance_cache_pos(1)
        return self._decode_graph_logits

    # --- CUDA-graph VERIFY (Phase 6) ---------------------------------------
    # One graph per query length S (= γ or γ+1). 7B verify is compute-dense, so
    # this is ~1.00× on the target; it exists so the option is available and as
    # the template for the draft executor where it does pay off.

    def _verify_buffers(self, seq_len: int) -> dict:
        """Lazily allocate per-S contiguous static buffers for the VERIFY graph."""
        st = self._verify_state.get(seq_len)
        if st is None:
            b, dev = self.buffers.batch, self.device
            st = {
                "input_ids": torch.zeros((b, seq_len), dtype=torch.int64, device=dev),
                "cur_len": torch.zeros((), dtype=torch.int64, device=dev),
                "graph": None,
                "logits": None,
            }
            self._verify_state[seq_len] = st
        return st

    def _prepare_verify_static(self, input_ids: torch.Tensor, seq_len: int) -> dict:
        """Fill the per-S verify buffers in place (outside the captured region)."""
        st = self._verify_buffers(seq_len)
        st["input_ids"].copy_(input_ids)
        self.buffers.cache_position.fill_(self._cache_pos)        # write_pos
        st["cur_len"].fill_(self._cache_pos + seq_len)            # cur_len = write_pos + S
        # RoPE gathered in-graph via static_rope(seq_len); no host slice/copy here.
        return st

    def _verify_forward_static(self, st: dict, seq_len: int) -> torch.Tensor:
        """VERIFY forward over S query rows using only static buffers + `_dev` ops."""
        cos, sin = self.buffers.static_rope(seq_len)   # gathered in-graph from cache_position
        self._static_ctx = (cos, sin, st["cur_len"])
        hidden = self._ops["embedding_forward"](
            st["input_ids"], self.weights["model.embed_tokens.weight"]
        )
        return self._forward_stack(hidden, seq_len, mode=ForwardMode.VERIFY, static_attn=True)

    def _capture_verify_graph(self, st: dict, seq_len: int) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._verify_forward_static(st, seq_len)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        pool = self._graph_pool or torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            logits = self._verify_forward_static(st, seq_len)
        st["graph"] = graph
        st["logits"] = logits
        self._graph_pool = pool

    def verify_gamma_graph(
        self,
        draft_token_ids: torch.Tensor,
        *,
        leading_bonus: int | None = None,
    ) -> torch.Tensor:
        """
        CUDA-graph version of ``verify_gamma`` (see that method for semantics).

        Captures one graph per query length S (= γ without a bonus, γ+1 with one)
        and replays it. Returns the **live** logits buffer ``[batch, S, vocab]``,
        overwritten on the next call to a graph method.
        """
        if not _same_device(draft_token_ids.device, self.device):
            raise ValueError(
                f"draft_token_ids on {draft_token_ids.device}, expected {self.device}"
            )
        if self._p1_logits is None:
            raise RuntimeError("verify_gamma_graph requires prefill() first")
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
                    f"bundled verify S=γ+1={seq_len} exceeds kernel limit {MAX_VERIFY_SEQ_LEN}"
                )
            bonus = torch.full(
                (draft_token_ids.shape[0], 1), int(leading_bonus),
                dtype=torch.int64, device=draft_token_ids.device,
            )
            input_ids = torch.cat([bonus, draft_token_ids], dim=1)
        else:
            seq_len = gamma
            input_ids = draft_token_ids

        self._validate_input_ids(input_ids, seq_len)

        st = self._prepare_verify_static(input_ids, seq_len)
        if st["graph"] is None:
            self._capture_verify_graph(st, seq_len)
        st["graph"].replay()
        self._advance_cache_pos(seq_len)
        return st["logits"]
