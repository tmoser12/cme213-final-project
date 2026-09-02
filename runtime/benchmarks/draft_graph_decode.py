#!/usr/bin/env python3
"""
runtime/benchmarks/draft_graph_decode.py — eager vs CUDA-graph decode for the 0.5B DRAFT.

Unlike the 7B target (memory-bound → 1.00×, see target_graph_benchmarks.md), the
0.5B draft has ~14× less weight traffic, so per-token GPU work is small and the
~150 host kernel launches matter. The decode graph should give a REAL speedup here.

Run: bash slurm/run_python.sh runtime/benchmarks/draft_graph_decode.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.core.config import RuntimeConfig, CONFIG_05B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor

DEVICE = "cuda"
PROMPT_LEN = 32
MAX_SEQ = 512
N_DECODE = 256


def _tok(t: int) -> torch.Tensor:
    return torch.tensor([t], dtype=torch.int64, device=DEVICE)


def _time_loop(step_fn, first_tok: int, n: int) -> float:
    tok = first_tok
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = step_fn(_tok(tok))
        tok = int(out[0, -1].argmax().item())
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _time_host_dispatch(step_fn, tok: int, n: int) -> float:
    """CPU time to queue n steps (fixed token, no per-step sync)."""
    fixed = _tok(tok)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step_fn(fixed)
    host = time.perf_counter() - t0
    torch.cuda.synchronize()
    return host


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Run via slurm/run_python.sh.")
        return 2

    cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
    print(f"GPU: {torch.cuda.get_device_name(0)}   model: {cfg.name} (kernel_set={cfg.kernel_set})")
    weights, _ = load_weights_on_gpu(cfg, batch=1, device=DEVICE)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=MAX_SEQ, device=DEVICE)
    ex = Qwen2Executor(cfg, weights, buffers)
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), dtype=torch.int64, device=DEVICE)

    # EAGER
    logits = ex.prefill(prompt)
    first = int(logits[0, -1].argmax().item())
    for _ in range(8):
        ex.decode_step(_tok(first))
    dt_eager = _time_loop(ex.decode_step, first, N_DECODE)

    # GRAPH (capture on first graphed step)
    logits = ex.prefill(prompt)
    first = int(logits[0, -1].argmax().item())
    for _ in range(8):
        ex.decode_step_graph(_tok(first))
    dt_graph = _time_loop(ex.decode_step_graph, first, N_DECODE)

    eager_tps = N_DECODE / dt_eager
    graph_tps = N_DECODE / dt_graph
    print("\n============ DRAFT decode tokens/sec (0.5B, batch=1) ============")
    print(f"  eager : {1000*dt_eager/N_DECODE:6.3f} ms/tok   {eager_tps:7.1f} tok/s")
    print(f"  graph : {1000*dt_graph/N_DECODE:6.3f} ms/tok   {graph_tps:7.1f} tok/s")
    print(f"  speedup: {graph_tps / eager_tps:.2f}x")

    # Host dispatch (the part graphs remove)
    logits = ex.prefill(prompt); tok = int(logits[0, -1].argmax().item())
    host_eager = _time_host_dispatch(ex.decode_step, tok, 64)
    logits = ex.prefill(prompt); tok = int(logits[0, -1].argmax().item())
    ex.decode_step_graph(_tok(tok))
    logits = ex.prefill(prompt); tok = int(logits[0, -1].argmax().item())
    host_graph = _time_host_dispatch(ex.decode_step_graph, tok, 64)
    print("\n  host-side dispatch (CPU time to queue one step, no GPU wait):")
    print(f"    eager : {1000*host_eager/64:6.3f} ms/tok  (~150 launches/token)")
    print(f"    graph : {1000*host_graph/64:6.3f} ms/tok  (1 replay + prepare)")

    # Correctness
    logits = ex.prefill(prompt); tok = int(logits[0, -1].argmax().item())
    lg_e = ex.decode_step(_tok(tok)).clone()
    logits = ex.prefill(prompt); tok = int(logits[0, -1].argmax().item())
    lg_g = ex.decode_step_graph(_tok(tok))
    diff = (lg_e.float() - lg_g.float()).abs().max().item()
    print(f"\n  correctness: graph vs eager next-token logits max|Δ| = {diff:.4f}"
          f"  ({'bit-exact' if diff == 0 else 'CHECK'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
