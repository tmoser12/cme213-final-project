"""Target-model residual add + LM head kernels."""

from runtime.production_kernels.target.residual_ops.ops import (
    lm_head_forward,
    residual_add_forward,
)

__all__ = ["residual_add_forward", "lm_head_forward"]
