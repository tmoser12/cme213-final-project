"""
src/kernels/embedding/wrapper.py
nn.Module wrapper around the custom CUDA embedding kernel.

Drop-in replacement for nn.Embedding's forward pass. Constructor takes an
existing nn.Embedding and adopts its weight by reference, matching the
CustomQwenMLP(original_mlp) pattern from kernel_monkey_patching_plan.md.

The JIT-compiled extension is loaded once at import time -- first import on
a fresh machine pays a 30-60s nvcc build; subsequent imports hit the cache.
"""

import torch
import torch.nn as nn

from src.kernels.embedding.jit import load_embedding_ops

_ops = load_embedding_ops()


class CustomEmbedding(nn.Module):
    def __init__(self, original: nn.Embedding):
        super().__init__()
        # Share weight tensor with `original` -- do not copy.
        self.weight = original.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _ops.embedding_forward(input_ids, self.weight)
