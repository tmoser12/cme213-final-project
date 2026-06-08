"""Pre-allocated GPU buffers for inference (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from runtime.core.config import RuntimeConfig
from runtime.core.memory import plan_memory
from runtime.core import shapes

_TORCH_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}

# Largest static query length a graph captures (decode S=1, verify S=γ+1 ≤ 8).
# Matches speculative.types.MAX_VERIFY_SEQ_LEN; kept local to avoid a cross-import.
MAX_STATIC_SEQ_LEN = 8


def _dtype(cfg: RuntimeConfig) -> torch.dtype:
    return _TORCH_DTYPE[cfg.dtype]


def _empty(shape: tuple[int, ...], *, cfg: RuntimeConfig, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=_dtype(cfg), device=device)


def build_rope_tables(
    cfg: RuntimeConfig,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cos/sin for positions [0, max_seq_len).

    Returns tensors shaped [max_seq_len, head_dim] in cfg.dtype (fp16).
    Matches HF Qwen2RotaryEmbedding (default rope, no scaling).
    """
    dim = cfg.head_dim
    theta = cfg.rope_theta
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(_dtype(cfg)), emb.sin().to(_dtype(cfg))


@dataclass
class RuntimeBuffers:
    """Device-resident buffers sized by plan_memory(); no per-step allocation."""

    cfg: RuntimeConfig
    batch: int
    max_seq_len: int
    device: torch.device

    hidden_a: torch.Tensor
    hidden_b: torch.Tensor
    q_states: torch.Tensor
    k_states: torch.Tensor
    v_states: torch.Tensor
    mlp_gate: torch.Tensor
    logits: torch.Tensor
    kv_cache_k: torch.Tensor
    kv_cache_v: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    cache_position: torch.Tensor

    # --- Static graph scratch (Phase 3 + RoPE folding) ---------------------
    # Fixed-address inputs for the CUDA-graphed decode/verify forwards. Updated in
    # place before each replay; read by the captured graph. Sub-KB total, so
    # intentionally excluded from the seq-scaling memory plan / nbytes().
    static_input_ids: torch.Tensor   # [batch, 1] int64 — the decode token
    static_cur_len: torch.Tensor     # 0-d int64 — cur_len = cache_position + S
    # Constant [0, 1, …, MAX_STATIC_SEQ_LEN) row offsets. The captured forward gathers
    # RoPE rows as rope_cos[cache_position + rope_arange[:S]] *inside the graph* (an
    # index_select driven by the device position), so no per-step host slice/copy is
    # needed and the slice isn't frozen at capture time.
    rope_arange: torch.Tensor        # [MAX_STATIC_SEQ_LEN] int64

    @property
    def kv_head_dim(self) -> int:
        return self.kv_cache_k.shape[-1]

    def nbytes(self) -> int:
        tensors = (
            self.hidden_a,
            self.hidden_b,
            self.q_states,
            self.k_states,
            self.v_states,
            self.mlp_gate,
            self.logits,
            self.kv_cache_k,
            self.kv_cache_v,
            self.rope_cos,
            self.rope_sin,
            self.cache_position,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def memory_report(self) -> dict[str, int]:
        """Byte counts keyed like plan_memory()['buffer_bytes']."""
        cfg = self.cfg

        def _nbytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        return {
            "hidden_ping": _nbytes(self.hidden_a),
            "hidden_pong": _nbytes(self.hidden_b),
            "q_states": _nbytes(self.q_states),
            "k_states": _nbytes(self.k_states),
            "v_states": _nbytes(self.v_states),
            "mlp_gate": _nbytes(self.mlp_gate),
            "logits": _nbytes(self.logits),
            "kv_cache_keys": _nbytes(self.kv_cache_k),
            "kv_cache_values": _nbytes(self.kv_cache_v),
            "rope_cos": _nbytes(self.rope_cos),
            "rope_sin": _nbytes(self.rope_sin),
            "cache_position": _nbytes(self.cache_position),
        }

    def kv_cache_k_layer(self, layer: int) -> torch.Tensor:
        """Per-layer K cache view: [batch, num_kv_heads, max_seq, head_dim]."""
        return self.kv_cache_k[layer]

    def kv_cache_v_layer(self, layer: int) -> torch.Tensor:
        """Per-layer V cache view: [batch, num_kv_heads, max_seq, head_dim]."""
        return self.kv_cache_v[layer]

    def rope_embeddings(
        self,
        start: int,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        RoPE cos/sin for positions [start, start + length).

        Returns [batch, length, head_dim] tensors for attention kernels.
        """
        cos = self.rope_cos[start : start + length]
        sin = self.rope_sin[start : start + length]
        cos = cos.unsqueeze(0).expand(self.batch, -1, -1)
        sin = sin.unsqueeze(0).expand(self.batch, -1, -1)
        return cos, sin

    def static_rope(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather RoPE cos/sin for the current device position, INSIDE the captured
        graph: rows = cache_position + rope_arange[:S]; cos/sin = table[rows].

        Returns ``[1, S, head_dim]`` views. The index_select reads the device
        position scalar (updated before each replay), so the slice is not frozen
        at capture time and no host-side copy is needed.
        """
        rows = self.rope_arange[:seq_len] + self.cache_position  # [S] int64, in-graph
        cos = self.rope_cos.index_select(0, rows).unsqueeze(0).expand(self.batch, -1, -1)
        sin = self.rope_sin.index_select(0, rows).unsqueeze(0).expand(self.batch, -1, -1)
        return cos, sin                                          # [batch, S, head_dim]

    def swap_hidden(self) -> None:
        """Exchange ping-pong hidden buffers (metadata only, zero GPU cost)."""
        self.hidden_a, self.hidden_b = self.hidden_b, self.hidden_a

    def reset_cache_position(self, value: int = 0) -> None:
        """Reset device-side sequence cursor (e.g. at start of prefill)."""
        self.cache_position.fill_(value)


def allocate_buffers(
    cfg: RuntimeConfig,
    batch: int | None = None,
    max_seq_len: int | None = None,
    device: str | torch.device = "cuda",
) -> RuntimeBuffers:
    """
    Allocate all runtime buffers on device, sized by plan_memory().

    Uses torch.empty for activations (no zero-fill cost) and zeros for KV cache.
    """
    cfg.validate()
    batch = batch if batch is not None else cfg.max_batch
    max_seq_len = max_seq_len if max_seq_len is not None else cfg.max_seq_len
    if batch <= 0 or max_seq_len <= 0:
        raise ValueError("batch and max_seq_len must be positive")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA not available")

    plan = plan_memory(cfg, batch=batch, max_seq_len=max_seq_len)
    buf = plan["buffers"]
    kv_d = shapes.kv_cache_head_dim(cfg)
    if kv_d != cfg.head_dim:
        raise ValueError(
            f"KV cache head_dim padding ({kv_d}) != cfg.head_dim ({cfg.head_dim}); "
            "attention kernels expect unpadded head_dim — fix shapes.kv_cache_head_dim"
        )
    row_bytes = kv_d * cfg.dtype_bytes
    if row_bytes % 16 != 0:
        raise ValueError(f"KV cache row not 16-byte aligned: {row_bytes} bytes")

    rope_cos, rope_sin = build_rope_tables(cfg, max_seq_len, dev)

    kv_shape = buf["kv_cache_keys"]
    return RuntimeBuffers(
        cfg=cfg,
        batch=batch,
        max_seq_len=max_seq_len,
        device=dev,
        hidden_a=_empty(buf["hidden_ping"], cfg=cfg, device=dev),
        hidden_b=_empty(buf["hidden_pong"], cfg=cfg, device=dev),
        q_states=_empty(buf["q_states"], cfg=cfg, device=dev),
        k_states=_empty(buf["k_states"], cfg=cfg, device=dev),
        v_states=_empty(buf["v_states"], cfg=cfg, device=dev),
        mlp_gate=_empty(buf["mlp_gate"], cfg=cfg, device=dev),
        logits=_empty(buf["logits"], cfg=cfg, device=dev),
        kv_cache_k=torch.zeros(kv_shape, dtype=_dtype(cfg), device=dev),
        kv_cache_v=torch.zeros(kv_shape, dtype=_dtype(cfg), device=dev),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        cache_position=torch.zeros((), dtype=torch.int64, device=dev),
        static_input_ids=torch.zeros((batch, 1), dtype=torch.int64, device=dev),
        static_cur_len=torch.zeros((), dtype=torch.int64, device=dev),
        rope_arange=torch.arange(MAX_STATIC_SEQ_LEN, dtype=torch.int64, device=dev),
    )


def buffer_fits_vram_budget(
    cfg: RuntimeConfig,
    weights: dict[str, torch.Tensor],
    buffers: RuntimeBuffers,
    reserve_mib: float = 512.0,
) -> dict:
    """
    Check that weights + buffers leave at least reserve_mib headroom on the GPU.

    Call after load_weights_on_gpu + allocate_buffers on the same device.
    """
    from runtime.core.weights import gpu_memory_snapshot, memory_report

    dev_index = buffers.device.index if buffers.device.index is not None else 0
    snap = gpu_memory_snapshot(dev_index)
    w_bytes = memory_report(weights)["total_bytes"]
    b_bytes = buffers.nbytes()
    reserve_bytes = int(reserve_mib * 1024 * 1024)
    fits = snap["free_bytes"] >= reserve_bytes
    return {
        "fits": fits,
        "weights_bytes": w_bytes,
        "buffers_bytes": b_bytes,
        "reserve_bytes": reserve_bytes,
        "gpu_free_bytes": snap["free_bytes"],
        "gpu_reserved_bytes": snap["reserved_bytes"],
        "gpu_total_bytes": snap["total_bytes"],
    }
