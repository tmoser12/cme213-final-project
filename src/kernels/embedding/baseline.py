"""
src/kernels/embedding/baseline.py
PyTorch reference implementation of the embedding lookup.

This is the "correct" answer we measure the custom CUDA kernel against. It
simply forwards to F.embedding (which is what nn.Embedding does internally).

Constructor takes an existing nn.Embedding and adopts its weight by reference,
mirroring the CustomQwenMLP(original_mlp) pattern from
kernel_monkey_patching_plan.md so baseline/wrapper share the same weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineEmbedding(nn.Module):
    def __init__(self, original: nn.Embedding):
        super().__init__()
        # Share weight tensor with `original` -- do not copy.
        self.weight = original.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)
