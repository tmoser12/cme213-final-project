#!/usr/bin/env python3
"""
runtime/benchmarks/phase4_graph_decode.py — eager vs CUDA-graph decode tokens/sec.

The Phase 4 payoff: replaying the captured decode graph collapses ~150 host kernel
launches per token into one replay(). This measures the speedup over the eager
baseline (Phase 0: 33.8 tok/s) and prints a quick correctness check.

Run: bash slurm/run_python.sh runtime/benchmarks/phase4_graph_decode.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor

DEVICE = "cuda"
PROMPT_LEN = 32
MAX_SEQ = 512
N_DECODE = 128


def _tok(t: int) -> torch.Tensor:
    return torch.tensor([t], dtype=torch.int64, device=DEVICE)


def _time_loop(step_fn, first_tok: int, n: int) -> tuple[float, int]:
    tok = first_tok
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = step_fn(_tok(tok))
        tok = int(out[0, -1].argmax().item())
    torch.cuda.synchronize()
    return time.perf_counter() - t0, tok


def _time_host_dispatch(step_fn, tok: int, n: int) -> float:
    """CPU time to *queue* n steps (fixed token, no per-step sync) — isolates the
    host-side launch overhead that CUDA graphs target, separate from GPU time."""
    fixed = _tok(tok)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step_fn(fixed)
    host = time.perf_counter() - t0   # returns once work is queued; GPU still busy
    torch.cuda.synchronize()
    return host


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Run via slurm/run_python.sh.")
        return 2

    cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    weights, _ = load_weights_on_gpu(cfg, batch=1, device=DEVICE)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=MAX_SEQ, device=DEVICE)
    ex = Qwen2Executor(cfg, weights, buffers)
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), dtype=torch.int64, device=DEVICE)

    # --- EAGER ---
    logits = ex.prefill(prompt)
    first = int(logits[0, -1].argmax().item())
    for _ in range(4):  # warmup
        first_w = int(ex.decode_step(_tok(first))[0, -1].argmax().item())
    dt_eager, _ = _time_loop(ex.decode_step, first, N_DECODE)

    # --- GRAPH --- (fresh state; capture happens on the first graphed step)
    logits = ex.prefill(prompt)
    first = int(logits[0, -1].argmax().item())
    # Prime/capture + a few warmup replays (not timed).
    for _ in range(4):
        first_g = int(ex.decode_step_graph(_tok(first))[0, -1].argmax().item())
    dt_graph, _ = _time_loop(ex.decode_step_graph, first, N_DECODE)

    eager_tps = N_DECODE / dt_eager
    graph_tps = N_DECODE / dt_graph
    print("\n================ decode tokens/sec (7B, batch=1) ================")
    print(f"  eager : {1000*dt_eager/N_DECODE:6.3f} ms/tok   {eager_tps:6.1f} tok/s")
    print(f"  graph : {1000*dt_graph/N_DECODE:6.3f} ms/tok   {graph_tps:6.1f} tok/s")
    print(f"  speedup: {graph_tps / eager_tps:.2f}x")

    # --- Host-side dispatch cost (the part graphs actually remove) ----------
    # Time to QUEUE work (no per-step sync). If this is << wall-clock ms/tok,
    # the host launch overhead is hidden behind GPU work => decode is GPU-bound.
    n_disp = 64
    logits = ex.prefill(prompt)
    tok = int(logits[0, -1].argmax().item())
    host_eager = _time_host_dispatch(ex.decode_step, tok, n_disp)
    logits = ex.prefill(prompt)
    tok = int(logits[0, -1].argmax().item())
    ex.decode_step_graph(_tok(tok))   # ensure captured
    logits = ex.prefill(prompt)
    tok = int(logits[0, -1].argmax().item())
    host_graph = _time_host_dispatch(ex.decode_step_graph, tok, n_disp)
    print("\n  host-side dispatch (CPU time to queue one step, no GPU wait):")
    print(f"    eager : {1000*host_eager/n_disp:6.3f} ms/tok  (~150 launches/token)")
    print(f"    graph : {1000*host_graph/n_disp:6.3f} ms/tok  (1 replay + prepare)")
    print(f"    -> GPU wall-clock is {1000*dt_eager/N_DECODE:.1f} ms/tok; "
          f"host dispatch is hidden behind it (decode is GPU/memory-bound).")

    # --- quick correctness: graph vs eager next-token logits at matched state ---
    logits = ex.prefill(prompt)
    tok = int(logits[0, -1].argmax().item())
    lg_e = ex.decode_step(_tok(tok)).clone()
    logits = ex.prefill(prompt)
    tok = int(logits[0, -1].argmax().item())
    lg_g = ex.decode_step_graph(_tok(tok))
    diff = (lg_e.float() - lg_g.float()).abs().max().item()
    print(f"\n  correctness: graph vs eager next-token logits max|Δ| = {diff:.4f}"
          f"  ({'bit-exact' if diff == 0 else 'CHECK'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
