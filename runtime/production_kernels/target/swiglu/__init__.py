"""Target-model SwiGLU MLP kernel."""

from runtime.production_kernels.target.swiglu.ops import swiglu_forward

__all__ = ["swiglu_forward"]
