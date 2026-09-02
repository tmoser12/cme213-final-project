#!/usr/bin/env python3
"""
runtime/benchmarks/phase0_graph_probe.py — CUDA Graph Phase 0.

Two things, no kernel changes:

  0.1 BASELINE  — eager decode tokens/sec for the 7B engine (the number to beat).

  0.2 HYPOTHESIS PROBE — the decisive test. Capture the *current* eager decode
      forward (which allocates every intermediate via torch::empty) with
      torch.cuda.graph at a FIXED cache position, then replay it several times
      with IDENTICAL inputs.

      Position is held fixed on purpose: that neutralizes the known host-int
      write_pos/cur_len baking problem, so this isolates ONE question —
      does graph-pool replay of an allocation-heavy forward reproduce the eager
      result? If yes, the previous attempt's "rewrite every op allocation-free" claim is
      unnecessary and we proceed with the lean plan (device-scalar positions +
      static inputs). If no, we investigate the culprit op.

Run:
  bash slurm/run_python.sh runtime/benchmarks/phase0_graph_probe.py
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
from runtime.speculative.types import ForwardMode

DEVICE = "cuda"
PROMPT_LEN = 32
MAX_SEQ_LEN = 256
N_DECODE_BASELINE = 64
N_REPLAYS = 3
# fp16 logits over ~152k vocab: eager vs replay should be ~0. The prior failure
# was 614 / 3978, so anything below this threshold is an unambiguous PASS.
PASS_TOL = 1.0


def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def setup():
    cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    weights, budget = load_weights_on_gpu(cfg, batch=1, device=DEVICE)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=MAX_SEQ_LEN, device=DEVICE)
    executor = Qwen2Executor(cfg, weights, buffers)
    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size, (1, PROMPT_LEN), dtype=torch.int64, device=DEVICE)
    return cfg, executor, prompt


# ---------------------------------------------------------------------------
# 0.1 Baseline: eager decode tokens/sec
# ---------------------------------------------------------------------------
def baseline(executor: Qwen2Executor, prompt: torch.Tensor) -> None:
    _banner("0.1  BASELINE — eager decode tokens/sec (7B, batch=1)")

    logits = executor.prefill(prompt)
    next_tok = int(logits[0, -1].argmax().item())

    # warmup decode steps (not timed)
    for _ in range(4):
        t = torch.tensor([next_tok], dtype=torch.int64, device=prompt.device)
        out = executor.decode_step(t)
        next_tok = int(out[0, -1].argmax().item())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_DECODE_BASELINE):
        t = torch.tensor([next_tok], dtype=torch.int64, device=prompt.device)
        out = executor.decode_step(t)
        next_tok = int(out[0, -1].argmax().item())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ms_per_tok = 1000.0 * dt / N_DECODE_BASELINE
    tok_per_s = N_DECODE_BASELINE / dt
    print(f"  decoded {N_DECODE_BASELINE} tokens in {dt:.3f}s")
    print(f"  -> {ms_per_tok:.3f} ms/token   |   {tok_per_s:.1f} tokens/sec (EAGER baseline)")


# ---------------------------------------------------------------------------
# 0.2 Hypothesis probe: capture + replay at a FIXED position
# ---------------------------------------------------------------------------
def hypothesis_probe(executor: Qwen2Executor, prompt: torch.Tensor) -> bool:
    _banner("0.2  HYPOTHESIS PROBE — graph-pool replay at fixed position")

    # Re-prefill from a clean cache so position is well-defined.
    executor.prefill(prompt)
    fixed_pos = executor.cache_pos  # = PROMPT_LEN; we decode at this slot, never advancing
    print(f"  fixed cache position = {fixed_pos}")

    # Static input buffer: the one input that would vary per real step.
    static_input_ids = torch.zeros((1, 1), dtype=torch.int64, device=prompt.device)
    static_input_ids.fill_(123)  # arbitrary fixed token, held constant for all runs

    embed_w = executor.weights["model.embed_tokens.weight"]
    ops = executor._ops

    def fixed_decode_forward() -> torch.Tensor:
        """Decode forward at the fixed position; does NOT advance the cursor."""
        executor._cache_pos = fixed_pos  # keep write_pos/cur_len constant across runs
        hidden = ops["embedding_forward"](static_input_ids, embed_w)
        return executor._forward_stack(hidden, 1, mode=ForwardMode.DECODE)

    # Eager reference (identical inputs, identical position).
    eager_logits = fixed_decode_forward().clone()
    torch.cuda.synchronize()

    # Warmup (initializes cuBLAS workspaces etc. before capture).
    for _ in range(4):
        _ = fixed_decode_forward()
    torch.cuda.synchronize()

    # Capture.
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_logits = fixed_decode_forward()
    except Exception as exc:  # capture itself failing is a finding
        print(f"  CAPTURE FAILED: {type(exc).__name__}: {exc}")
        return False
    torch.cuda.synchronize()

    # NOTE: with a correctly-streamed graph, capture RECORDS but does not EXECUTE
    # the kernels — `captured_logits` holds its pool-init value (zeros) until the
    # first replay(). So we only judge correctness AFTER replay. (Pre-fix, the
    # kernels wrongly ran on the default stream during capture, so this buffer
    # held real values; that was the bug, not a feature.)
    print(f"  capture vs eager (pre-replay, expected nonzero): "
          f"max|Δ| = {_max_abs_diff(captured_logits, eager_logits):.4f}")

    # Replay several times with identical inputs; each must match eager.
    replay_diffs = []
    for i in range(N_REPLAYS):
        graph.replay()
        torch.cuda.synchronize()
        d = _max_abs_diff(captured_logits, eager_logits)
        replay_diffs.append(d)
        print(f"  replay #{i + 1}  vs eager : max|Δ| = {d:.4f}")

    worst = max(replay_diffs)
    ok = worst < PASS_TOL
    _banner(f"PROBE RESULT: {'PASS' if ok else 'FAIL'}  (worst replay max|Δ| = {worst:.4f}, tol = {PASS_TOL})")
    if ok:
        print("  -> graph replay reproduces eager. Kernels capture correctly (current-stream")
        print("     launches) and torch::empty intermediates are graph-safe via the pool.")
    else:
        print("  -> replay diverges even at fixed position with identical inputs.")
        print("     The allocation hypothesis needs revisiting; identify the culprit op.")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Run via slurm/run_python.sh.")
        return 2
    cfg, executor, prompt = setup()
    baseline(executor, prompt)
    ok = hypothesis_probe(executor, prompt)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
