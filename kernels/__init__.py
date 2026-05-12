"""
kernels
Python entry points for the custom CUDA kernels.

Build prerequisite (one-time, on a GPU node):
    cd kernels && python setup.py build_ext --inplace
"""

import torch
from . import _C  # built by setup.py, contains the pybind11 module


def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Indexed row gather equivalent to torch.nn.functional.embedding for the
    no-padding-idx, no-scale, no-renorm case.

    Args:
        input_ids: int64 CUDA tensor, any shape S. Values in [0, V).
        weight:    fp16  CUDA tensor of shape [V, H], contiguous.

    Returns:
        fp16 CUDA tensor of shape S + (H,).
    """
    return _C.embedding_forward(input_ids.contiguous(), weight.contiguous())
