"""Zero-cost NVTX range helper for profiling the executor.

Ranges are emitted only when ``RUNTIME_NVTX=1`` is set in the environment, so the
hot path pays nothing in production. Under nsys (``--trace=...,nvtx``) the pushed
tags show up in the ``nvtx_sum`` report, attributing GPU time per op / layer /
phase — in particular disambiguating the cuBLAS GEMMs (qkv_proj, o_proj, the
swiglu gate/up/down, lm_head) that otherwise all appear as ``turing_*gemm``.
"""

from __future__ import annotations

import contextlib
import os

import torch

ENABLED = os.environ.get("RUNTIME_NVTX") == "1"


@contextlib.contextmanager
def rng(tag: str):
    """Push an NVTX range named ``tag`` for the duration of the block (no-op if disabled)."""
    if not ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(tag)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
