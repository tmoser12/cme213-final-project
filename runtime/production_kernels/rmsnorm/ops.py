"""
RMSNorm op — production host interface.

Called directly by the inference runtime (no HuggingFace monkey patching).
Matches the plan ABI: init / workspace_bytes / forward.
"""

from __future__ import annotations

import torch

from runtime.production_kernels.rmsnorm.jit import get_ops

_ops = None


def init() -> None:
    """Load and warm the JIT-compiled CUDA extension."""
    global _ops
    _ops = get_ops()


def workspace_bytes(batch: int = 1, seq_len: int = 1, hidden_size: int = 0) -> int:
    """Scratch memory required (0 — kernel allocates output via PyTorch)."""
    return 0


def forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    RMSNorm forward pass.

    Args:
        input:  [batch, seq_len, hidden_size] FP16 CUDA contiguous
        weight: [hidden_size] FP16 CUDA contiguous (per-layer norm weight)
        eps:    variance epsilon (from config rms_norm_eps)

    Returns:
        output: same shape/dtype/device as input
    """
    global _ops
    if _ops is None:
        init()
    return _ops.forward(input, weight, eps)
