"""CUDA-graph capture/replay for the Qwen2 executor (decode).

`GraphExecutorMixin` is folded into `Qwen2Executor`. It owns the static-buffer
forwards and the graph capture/replay, keeping `executor.py` focused on the eager
forward math. The methods here use the core executor's interface on ``self``:

    self.cfg, self.buffers, self.weights, self.device, self._ops,
    self._cache_pos, self._softmax_scale, self._p1_logits,
    self._forward_stack(...), self._advance_cache_pos(n), self._validate_input_ids(...)

and they set ``self._static_ctx = (cos, sin, cur_len)``, which the core's
``_run_attention(static_attn=True)`` reads. One decode graph serves all S=1 steps;
the draft uses it for each proposed token.

"""

from __future__ import annotations

import torch

from runtime.speculative.types import ForwardMode


def _same_device(a: torch.device, b: torch.device) -> bool:
    """``cuda`` and ``cuda:0`` compare equal."""
    return a.type == b.type and (a.index or 0) == (b.index or 0)


class GraphExecutorMixin:
    """Static-buffer forwards + CUDA-graph capture/replay for decode."""

    def _init_graph_state(self, use_cuda_graph: bool) -> None:
        """Initialize graph state (called from ``Qwen2Executor.__init__``)."""
        # One decode graph serves all S=1 steps / prompts. Positions live in
        # device scalars and KV/buffer addresses
        # are stable, so a captured graph is reused across prompts.
        self.use_cuda_graph = use_cuda_graph
        self._decode_graph: "torch.cuda.CUDAGraph | None" = None
        self._decode_graph_logits: torch.Tensor | None = None
        self._graph_pool = None
        # Active static-attention context (cos, sin, cur_len device scalar) read by
        # the static path of _run_attention; set by the prepare/forward step.
        self._static_ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    # --- Static / graph-capturable decode ------------------------

    # WALKTHROUGH [GRAPH_INPUT] — see demo/WALKTHROUGH.md.
    # Keep addresses fixed and copy new values into them. Host integers captured as
    # launch arguments would otherwise freeze the cache position at capture time.
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

    # WALKTHROUGH [GRAPH_FORWARD] — see demo/WALKTHROUGH.md.
    # Capture the entire one-token model forward, including RoPE lookup and each
    # layer, rather than only attention. Python executes here at capture, not replay.
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


    # --- CUDA-graph decode ---------------------------------------

    # WALKTHROUGH [GRAPH_CAPTURE] — see demo/WALKTHROUGH.md.
    # Warm library workspaces on a side stream before recording GPU operations.
    # CUDA graph capture records existing compiled kernels; it does not compile them.
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

    # WALKTHROUGH [GRAPH_REPLAY] — see demo/WALKTHROUGH.md.
    # A replay reuses graph-owned intermediates and logits. The caller must consume
    # or clone the live output before another replay overwrites it.
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
