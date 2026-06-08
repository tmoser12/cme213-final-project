"""Full-forward-pass profiling driver for the native Qwen2 runtime.

Runs a real prefill followed by N decode steps through ``Qwen2Executor`` so nsys
can capture an end-to-end decoder timeline built from the AOT kernel ``.so``s.
Weights load HF-free (``load_weights_on_gpu``) so the timeline isn't polluted by
``from_pretrained`` / ``torch.compile`` and we keep a single VRAM copy.

The captured region (prefill + decode) is bracketed with cudaProfilerStart/Stop
so warmup and weight-load stay out of the .nsys-rep. Per-op / per-layer / per-
phase attribution comes from the executor's NVTX ranges, which are emitted only
when ``RUNTIME_NVTX=1`` (the shell wrapper sets it).

``--graph`` profiles the CUDA-graph replay path (``decode_step_graph`` /
``verify_gamma_graph``) instead of eager launches; prefill is always eager (never
graphed). Under replay the executor's NVTX ranges do **not** fire (host-side, not
recorded into the graph), so pass nsys ``--cuda-graph-trace=node`` to expand the
replayed graph into its kernel nodes — the timeline then shows the whole forward
stack packed back-to-back (the non-launch-bound view) with per-kernel GPU time.
``--forward {decode,verify}`` selects the S=1 decode graph or the S=γ verify graph.

Run under nsys via ``scripts/profile_forward.sh``; or directly on a GPU node:

    RUNTIME_NVTX=1 srun --partition=gpu-turing --gres=gpu:1 \\
        python -m runtime.tools.profile_forward --model target --seq-len 512

``--sweep`` instead times prefill/decode latency + tok/s across several prompt
lengths (no nsys, NVTX off) and writes a ruled .txt table next to the .nsys-reps
— the pre-CUDA-graph baseline. Driven by ``scripts/benchmark_forward.sh``:

    srun --partition=gpu-turing --gres=gpu:1 \\
        python -m runtime.tools.profile_forward --sweep --model draft \\
            --seq-lens 128,512,1024,2048 --decode-steps 32
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from runtime.buffers import allocate_buffers
from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights_on_gpu
from runtime.executor import Qwen2Executor
from runtime.nvtx import ENABLED as NVTX_ENABLED

_CONFIG = {"target": CONFIG_7B, "draft": CONFIG_05B}


def _build_cfg(model: str) -> RuntimeConfig:
    cfg = RuntimeConfig.from_yaml(_CONFIG[model], project_root=_PROJECT_ROOT)
    cfg.validate()
    return cfg


def profile_forward(
    model: str,
    seq_len: int,
    decode_steps: int,
    batch: int,
    *,
    use_graph: bool = False,
    forward: str = "decode",
    gamma: int = 4,
    warmup_decode: int = 8,
    sync_each_decode: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run on a GPU node via slurm")

    device = "cuda"
    cfg = _build_cfg(model)
    path = "graph" if use_graph else "eager"
    print(f"[profile_forward] model={model} ({cfg.name})  forward={forward}  path={path}  "
          f"seq_len={seq_len}  steps={decode_steps}  batch={batch}  "
          f"NVTX={'on' if NVTX_ENABLED else 'OFF'}")
    if use_graph:
        print("[profile_forward] NOTE: NVTX ranges do not fire under graph replay (host-side, "
              "not captured); run nsys with --cuda-graph-trace=node to see the graph's kernels.")
    elif not NVTX_ENABLED:
        print("[profile_forward] WARNING: RUNTIME_NVTX!=1 — per-op ranges will be absent.")

    # Buffers hold the prompt plus every step's tokens (γ per verify step) with a
    # little pad; size for the longer of warmup vs the captured run.
    step_tokens = 1 if forward == "decode" else gamma
    max_seq_len = seq_len + max(decode_steps, warmup_decode) * step_tokens + 8

    weights, budget = load_weights_on_gpu(cfg, batch=batch, device=device)
    print(f"[profile_forward] weights loaded: {budget['weights']['total_mib']:.0f} MiB")
    buffers = allocate_buffers(cfg, batch=batch, max_seq_len=max_seq_len, device=device)
    executor = Qwen2Executor(cfg, weights, buffers, use_cuda_graph=use_graph)

    prompt = torch.randint(0, cfg.vocab_size, (batch, seq_len), dtype=torch.int64, device=device)

    # Per-iteration forward: decode reads the next token from the prior logits;
    # verify replays a fixed γ draft block (content is irrelevant to timing). In
    # graph mode the *_graph twin self-captures on first call, so the warmup below
    # builds the graph and the bracket replays it.
    if forward == "decode":
        _fwd = executor.decode_step_graph if use_graph else executor.decode_step
        step = lambda lg: _fwd(lg[:, -1].argmax(dim=-1))
    else:  # verify
        _fwd = executor.verify_gamma_graph if use_graph else executor.verify_gamma
        _draft = torch.randint(0, cfg.vocab_size, (batch, gamma), dtype=torch.int64, device=device)
        step = lambda lg: _fwd(_draft)

    # ---- Warmup (OUTSIDE the capture bracket) --------------------------------
    # Mirror the captured workload's SHAPES: prefill at seq_len then step at the real
    # context depth, so allocator growth / cuBLAS algo selection / module loads are
    # paid here, not in the capture. In graph mode this first run also *captures* the
    # decode/verify graph, so the bracket below replays an already-built graph.
    print("[profile_forward] warming up (same shapes as capture)...")
    with torch.no_grad():
        logits = executor.prefill(prompt)
        for _ in range(warmup_decode):
            logits = step(logits)
    torch.cuda.synchronize()

    # Warmup advanced the cursor; clear it so the captured prefill's pre-reset length
    # validation sees a fresh cache (prefill validates before it resets the KV cache).
    executor.reset_kv_cache()

    # ---- Captured region: real prefill (eager) + graph/eager steps ------------
    # Sync after each step so every range bounds exactly one step's GPU work (true
    # per-step latency) rather than host enqueue time gated by launch backpressure.
    print(f"[profile_forward] capturing prefill + {forward} ({path})...")
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        logits = executor.prefill(prompt)              # eager — prefill is never graphed
        torch.cuda.synchronize()
        for _ in range(decode_steps):
            logits = step(logits)
            if sync_each_decode:
                torch.cuda.synchronize()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("[profile_forward] done.")


# --------------------------------------------------------------------------- #
# Timed latency sweep (no nsys): prefill / decode wall-times + tok/s table.    #
# --------------------------------------------------------------------------- #
# This is the pre-CUDA-graph baseline. We deliberately measure wall-clock with a
# device sync around each region: decode here is host-launch-bound, so the host
# time IS the latency and must be counted. Per-token sync also means each decode
# sample is one token's true latency (matching the capture path's convention),
# and the p90/max columns then surface the occasional multi-ms host-side stall.


def _p90(values: list[float]) -> float:
    """Nearest-rank 90th percentile — exposes the host-jitter tail in decode."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(0.90 * len(s)) - 1))]


def _time_prefill(
    executor: Qwen2Executor, prompt: torch.Tensor, reps: int
) -> tuple[list[float], torch.Tensor]:
    """Return (prefill wall-times in ms, last logits).

    Resets the KV cache before each rep so every prefill is independent — and so
    its pre-reset length validation sees a fresh cursor (``prefill`` validates
    against the current position *before* it resets). The returned logits come
    from the final rep and seed the decode timing below.
    """
    times: list[float] = []
    logits: torch.Tensor | None = None
    for _ in range(reps):
        executor.reset_kv_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = executor.prefill(prompt)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    assert logits is not None
    return times, logits


def _time_decode(
    executor: Qwen2Executor, seed_logits: torch.Tensor, decode_steps: int,
    use_graph: bool = False,
) -> list[float]:
    """Per-token decode wall-times (ms), continuing from the prefilled context."""
    decode = executor.decode_step_graph if use_graph else executor.decode_step
    tok = seed_logits[:, -1].argmax(dim=-1)
    per_step: list[float] = []
    for _ in range(decode_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = decode(tok)
        torch.cuda.synchronize()
        per_step.append((time.perf_counter() - t0) * 1e3)
        tok = logits[:, -1].argmax(dim=-1)
    return per_step


def _render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Right-aligned, ``|``-separated, ruled table (matches the micro-bench .txt's)."""
    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    widths = [max(len(c) for c in col) for col in cols]
    fmt = lambda r: " | ".join(c.rjust(widths[i]) for i, c in enumerate(r))
    return [fmt(headers), "-+-".join("-" * w for w in widths)] + [fmt(r) for r in rows]


def _build_report_text(
    model: str, cfg: RuntimeConfig, batch: int, decode_steps: int, reps: int,
    rows: list[dict], path: str = "eager",
) -> str:
    headers = ["Seq", "Prefill (ms)", "Prefill (tok/s)", "Decode (ms/tok)",
               "Decode p90 (ms)", "Decode max (ms)", "Decode (tok/s)"]
    body = [[
        str(r["seq"]),
        f"{r['prefill_ms']:.2f}",
        f"{r['prefill_toks']:.0f}",
        f"{r['decode_ms']:.3f}",
        f"{r['decode_p90']:.3f}",
        f"{r['decode_max']:.3f}",
        f"{r['decode_toks']:.1f}",
    ] for r in rows]
    table = _render_table(headers, body)
    width = max(len(line) for line in table)
    try:
        dev = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — cosmetic header only
        dev = "unknown-gpu"
    title = f"Full-Forward Latency Sweep — {model} ({cfg.name})"
    head = [
        title,
        "=" * max(width, len(title)),
        f"device={dev}  dtype=fp16  batch={batch}  decode_steps={decode_steps}  "
        f"prefill_reps={reps}  decode_path={path} (prefill always eager)",
        f"layers={cfg.num_hidden_layers}  hidden={cfg.hidden_size}  "
        f"heads={cfg.num_attention_heads}q/{cfg.num_key_value_heads}kv  vocab={cfg.vocab_size}",
        "prefill (tok/s)=seq/prefill_time (prompt ingest); decode latency synced "
        "per token (median), p90/max show the host-jitter tail",
        "",
    ]
    return "\n".join(head + table) + "\n"


def benchmark_forward_sweep(
    model: str,
    seq_lens: list[int],
    decode_steps: int,
    batch: int,
    reps: int,
    out_dir: Path,
    use_graph: bool = False,
    warmup_decode: int = 8,
) -> Path:
    """Time prefill + decode across ``seq_lens`` and write a ruled .txt table.

    Weights load once; buffers are allocated once at the largest length and reused
    for every shorter prompt. Each length is warmed at its own shape before timing
    so cuBLAS algo selection / allocator growth / module loads stay out of the
    numbers (the same reason the nsys capture warms before bracketing).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run on a GPU node via slurm")

    device = "cuda"
    cfg = _build_cfg(model)
    path = "graph" if use_graph else "eager"
    max_seq_len = max(seq_lens) + decode_steps + 8
    print(f"[benchmark] model={model} ({cfg.name})  seq_lens={seq_lens}  "
          f"decode_steps={decode_steps}  batch={batch}  reps={reps}  decode_path={path}")

    weights, budget = load_weights_on_gpu(cfg, batch=batch, device=device)
    print(f"[benchmark] weights loaded: {budget['weights']['total_mib']:.0f} MiB")
    buffers = allocate_buffers(cfg, batch=batch, max_seq_len=max_seq_len, device=device)
    executor = Qwen2Executor(cfg, weights, buffers)
    vocab = cfg.vocab_size

    rows: list[dict] = []
    with torch.no_grad():
        for sl in seq_lens:
            prompt = torch.randint(0, vocab, (batch, sl), dtype=torch.int64, device=device)

            # Warmup at THIS shape (prefill + a few decode), outside timing. In graph
            # mode the first decode captures the graph (one capture serves all lengths).
            decode = executor.decode_step_graph if use_graph else executor.decode_step
            executor.reset_kv_cache()
            logits = executor.prefill(prompt)
            tok = logits[:, -1].argmax(dim=-1)
            for _ in range(warmup_decode):
                logits = decode(tok)
                tok = logits[:, -1].argmax(dim=-1)
            torch.cuda.synchronize()

            pre_ms_list, seed = _time_prefill(executor, prompt, reps)
            dec_ms_list = _time_decode(executor, seed, decode_steps, use_graph=use_graph)

            pre_ms = statistics.median(pre_ms_list)
            dec_ms = statistics.median(dec_ms_list)
            rows.append({
                "seq": sl,
                "prefill_ms": pre_ms,
                "prefill_toks": sl / (pre_ms / 1e3),
                "decode_ms": dec_ms,
                "decode_p90": _p90(dec_ms_list),
                "decode_max": max(dec_ms_list),
                "decode_toks": 1e3 / dec_ms,
            })
            print(f"  seq={sl:<5d} prefill {pre_ms:8.2f} ms ({sl / (pre_ms / 1e3):8.0f} tok/s)"
                  f"   decode {dec_ms:6.3f} ms/tok ({1e3 / dec_ms:7.1f} tok/s)"
                  f"  p90 {_p90(dec_ms_list):.3f}  max {max(dec_ms_list):.3f}")

    report = _build_report_text(model, cfg, batch, decode_steps, reps, rows, path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forward_sweep_{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path.write_text(report)
    print("\n" + report)
    print(f"[benchmark] wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("target", "draft"), required=True)
    p.add_argument("--seq-len", type=int, default=512,
                   help="single-capture mode: prompt = prefill = decode context depth")
    p.add_argument("--decode-steps", type=int, default=None,
                   help="decode steps (default 8 for capture, 32 for --sweep)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--graph", action="store_true",
                   help="profile the CUDA-graph replay path (decode_step_graph / "
                        "verify_gamma_graph) instead of eager. Works with capture and "
                        "--sweep. NVTX ranges don't fire under replay; pair with nsys "
                        "--cuda-graph-trace=node.")
    p.add_argument("--forward", choices=("decode", "verify"), default="decode",
                   help="capture mode: which forward to profile — S=1 decode or S=γ verify")
    p.add_argument("--gamma", type=int, default=4,
                   help="--forward verify: query length S (= γ draft tokens)")
    p.add_argument("--no-sync-each-decode", action="store_true",
                   help="capture mode: don't sync per decode step; NVTX ranges then show "
                        "the async pipeline (host enqueue) instead of per-token latency")
    # ---- timed sweep mode (no nsys) ----
    p.add_argument("--sweep", action="store_true",
                   help="time prefill/decode latency + tok/s across --seq-lens and write a "
                        "ruled .txt table to results/profiles/<model>/full/")
    p.add_argument("--seq-lens", type=str, default="128,512,1024,2048",
                   help="--sweep: comma-separated prompt lengths (batch=1)")
    p.add_argument("--reps", type=int, default=3,
                   help="--sweep: prefill timing repetitions (median reported)")
    args = p.parse_args()

    if args.sweep:
        seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
        if not seq_lens:
            p.error("--seq-lens parsed to an empty list")
        benchmark_forward_sweep(
            model=args.model,
            seq_lens=seq_lens,
            decode_steps=args.decode_steps or 32,
            batch=args.batch,
            reps=args.reps,
            out_dir=_PROJECT_ROOT / "results" / "profiles" / args.model / "full",
            use_graph=args.graph,
        )
        return

    profile_forward(
        model=args.model,
        seq_len=args.seq_len,
        decode_steps=args.decode_steps or 8,
        batch=args.batch,
        use_graph=args.graph,
        forward=args.forward,
        gamma=args.gamma,
        sync_each_decode=not args.no_sync_each_decode,
    )


if __name__ == "__main__":
    main()
