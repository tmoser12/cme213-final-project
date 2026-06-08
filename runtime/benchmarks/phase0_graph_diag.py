#!/usr/bin/env python3
"""
runtime/benchmarks/phase0_graph_diag.py — localize the capture NaN (Phase 0 follow-up).

phase0_graph_probe.py showed capture itself yields NaN. This script captures
progressively larger chunks of the decode forward at a FIXED position and reports
NaN/inf + min/max at each stage, to find where the NaN first appears:

  A. eager full forward (control)
  B. capture+replay: single rmsnorm op
  C. capture+replay: one decoder layer
  D. capture+replay: full stack MINUS lm_head (hidden states)
  E. capture+replay: full forward (with lm_head)

Uses the canonical side-stream warmup pattern + an explicit graph pool handle.

Run: bash slurm/run_python.sh runtime/benchmarks/phase0_graph_diag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor
from runtime.speculative.types import ForwardMode

DEVICE = "cuda"
PROMPT_LEN = 32
MAX_SEQ_LEN = 256


def stats(name: str, t: torch.Tensor) -> None:
    tf = t.float()
    n_nan = torch.isnan(tf).sum().item()
    n_inf = torch.isinf(tf).sum().item()
    finite = tf[torch.isfinite(tf)]
    lo = finite.min().item() if finite.numel() else float("nan")
    hi = finite.max().item() if finite.numel() else float("nan")
    print(f"    {name:32s} nan={n_nan:<8d} inf={n_inf:<6d} min={lo:+.3e} max={hi:+.3e}")


def capture_replay(label: str, fn, warmup_state_reset=None):
    """Warmup (side stream) -> capture -> 2 replays, printing stats each step."""
    print(f"  [{label}]")
    # eager reference
    if warmup_state_reset:
        warmup_state_reset()
    eager = fn().clone()
    stats("eager", eager)

    # side-stream warmup (canonical pattern)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            if warmup_state_reset:
                warmup_state_reset()
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    try:
        if warmup_state_reset:
            warmup_state_reset()
        with torch.cuda.graph(g, pool=pool):
            captured = fn()
    except Exception as exc:
        print(f"    CAPTURE FAILED: {type(exc).__name__}: {exc}")
        return
    torch.cuda.synchronize()
    stats("captured", captured)

    for i in range(2):
        g.replay()
        torch.cuda.synchronize()
        stats(f"replay#{i+1}", captured)
    print()


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.")
        return 2

    cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    weights, _ = load_weights_on_gpu(cfg, batch=1, device=DEVICE)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=MAX_SEQ_LEN, device=DEVICE)
    ex = Qwen2Executor(cfg, weights, buffers)

    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), dtype=torch.int64, device=DEVICE)
    ex.prefill(prompt)
    fixed_pos = ex.cache_pos
    print(f"fixed cache position = {fixed_pos}\n")

    ops = ex._ops
    embed_w = weights["model.embed_tokens.weight"]
    static_ids = torch.full((1, 1), 123, dtype=torch.int64, device=DEVICE)

    def set_pos():
        ex._cache_pos = fixed_pos

    # Precompute a fixed hidden input for op/layer-level tests.
    set_pos()
    base_hidden = ops["embedding_forward"](static_ids, embed_w).clone()

    # A. eager full forward control
    print("A. eager full forward (control)")
    set_pos()
    full_eager = ex._forward_stack(ops["embedding_forward"](static_ids, embed_w), 1,
                                   mode=ForwardMode.DECODE)
    stats("full forward logits", full_eager)
    print()

    # B. single rmsnorm op
    norm_w = weights["model.layers.0.input_layernorm.weight"]
    capture_replay(
        "B. single rmsnorm",
        lambda: ops["rmsnorm_forward"](base_hidden, norm_w, cfg.rms_norm_eps),
    )

    # C. one decoder layer (layer 0)
    capture_replay(
        "C. one decoder layer (L0)",
        lambda: ex._run_decoder_layer(base_hidden, 0, 1, mode=ForwardMode.DECODE),
        warmup_state_reset=set_pos,
    )

    # D. full stack minus lm_head (hidden states only)
    def stack_no_lmhead():
        set_pos()
        h = ops["embedding_forward"](static_ids, embed_w)
        for layer in range(cfg.num_hidden_layers):
            h = ex._run_decoder_layer(h, layer, 1, mode=ForwardMode.DECODE)
        return ops["rmsnorm_forward"](h, weights["model.norm.weight"], cfg.rms_norm_eps)

    capture_replay("D. full stack minus lm_head", stack_no_lmhead, warmup_state_reset=set_pos)

    # E. full forward with lm_head
    def full_forward():
        set_pos()
        h = ops["embedding_forward"](static_ids, embed_w)
        return ex._forward_stack(h, 1, mode=ForwardMode.DECODE)

    capture_replay("E. full forward (with lm_head)", full_forward, warmup_state_reset=set_pos)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
