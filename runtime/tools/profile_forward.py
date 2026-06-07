"""Full-forward-pass profiling driver for the native Qwen2 runtime.

Runs a real prefill followed by N decode steps through ``Qwen2Executor`` so nsys
can capture an end-to-end decoder timeline built from the AOT kernel ``.so``s.
Weights load HF-free (``load_weights_on_gpu``) so the timeline isn't polluted by
``from_pretrained`` / ``torch.compile`` and we keep a single VRAM copy.

The captured region (prefill + decode) is bracketed with cudaProfilerStart/Stop
so warmup and weight-load stay out of the .nsys-rep. Per-op / per-layer / per-
phase attribution comes from the executor's NVTX ranges, which are emitted only
when ``RUNTIME_NVTX=1`` (the shell wrapper sets it).

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
    warmup_decode: int = 8,
    sync_each_decode: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run on a GPU node via slurm")

    device = "cuda"
    cfg = _build_cfg(model)
    print(f"[profile_forward] model={model} ({cfg.name})  seq_len={seq_len}  "
          f"decode_steps={decode_steps}  batch={batch}  NVTX={'on' if NVTX_ENABLED else 'OFF'}")
    if not NVTX_ENABLED:
        print("[profile_forward] WARNING: RUNTIME_NVTX!=1 — per-op ranges will be absent.")

    # Buffers must hold the prompt plus every decoded token, with a little pad.
    max_seq_len = seq_len + decode_steps + 8

    weights, budget = load_weights_on_gpu(cfg, batch=batch, device=device)
    print(f"[profile_forward] weights loaded: {budget['weights']['total_mib']:.0f} MiB")
    buffers = allocate_buffers(cfg, batch=batch, max_seq_len=max_seq_len, device=device)
    executor = Qwen2Executor(cfg, weights, buffers)

    vocab = cfg.vocab_size
    prompt = torch.randint(0, vocab, (batch, seq_len), dtype=torch.int64, device=device)

    # ---- Warmup (OUTSIDE the capture bracket) --------------------------------
    # Must mirror the captured workload's SHAPES: prefill at seq_len then decode
    # at the real context depth. Warming at a smaller length leaves the caching
    # allocator, cuBLAS workspace/algo selection, and CUDA module loads for the
    # 512-shaped path unpaid, so those one-time costs leak into the capture and
    # spike the first few decode steps. prefill() resets the KV cache, so the
    # captured prefill below reuses every block this warmup grew.
    print("[profile_forward] warming up (same shapes as capture)...")
    with torch.no_grad():
        logits = executor.prefill(prompt)
        tok = logits[:, -1].argmax(dim=-1)
        for _ in range(warmup_decode):
            logits = executor.decode_step(tok)
            tok = logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()

    # Warmup advanced the cache cursor to seq_len + warmup_decode; clear it so the
    # captured prefill's pre-reset validation sees a fresh cache (prefill validates
    # input length against the *current* position before it resets the KV cache).
    executor.reset_kv_cache()

    # ---- Captured region: real prefill + decode steps -------------------------
    # Sync after each decode step so every NVTX "decode" range bounds exactly one
    # token's GPU work (true per-token latency) rather than host enqueue time
    # gated by launch-queue backpressure.
    print("[profile_forward] capturing prefill + decode...")
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        logits = executor.prefill(prompt)              # NVTX "prefill"
        torch.cuda.synchronize()
        tok = logits[:, -1].argmax(dim=-1)
        for _ in range(decode_steps):
            logits = executor.decode_step(tok)         # NVTX "decode"
            if sync_each_decode:
                torch.cuda.synchronize()
            tok = logits[:, -1].argmax(dim=-1)
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
    executor: Qwen2Executor, seed_logits: torch.Tensor, decode_steps: int
) -> list[float]:
    """Per-token decode wall-times (ms), continuing from the prefilled context."""
    tok = seed_logits[:, -1].argmax(dim=-1)
    per_step: list[float] = []
    for _ in range(decode_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = executor.decode_step(tok)
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
    rows: list[dict],
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
        f"device={dev}  dtype=fp16  batch={batch}  decode_steps={decode_steps}  prefill_reps={reps}",
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
    max_seq_len = max(seq_lens) + decode_steps + 8
    print(f"[benchmark] model={model} ({cfg.name})  seq_lens={seq_lens}  "
          f"decode_steps={decode_steps}  batch={batch}  reps={reps}")

    weights, budget = load_weights_on_gpu(cfg, batch=batch, device=device)
    print(f"[benchmark] weights loaded: {budget['weights']['total_mib']:.0f} MiB")
    buffers = allocate_buffers(cfg, batch=batch, max_seq_len=max_seq_len, device=device)
    executor = Qwen2Executor(cfg, weights, buffers)
    vocab = cfg.vocab_size

    rows: list[dict] = []
    with torch.no_grad():
        for sl in seq_lens:
            prompt = torch.randint(0, vocab, (batch, sl), dtype=torch.int64, device=device)

            # Warmup at THIS shape (prefill + a few decode), outside timing.
            executor.reset_kv_cache()
            logits = executor.prefill(prompt)
            tok = logits[:, -1].argmax(dim=-1)
            for _ in range(warmup_decode):
                logits = executor.decode_step(tok)
                tok = logits[:, -1].argmax(dim=-1)
            torch.cuda.synchronize()

            pre_ms_list, seed = _time_prefill(executor, prompt, reps)
            dec_ms_list = _time_decode(executor, seed, decode_steps)

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

    report = _build_report_text(model, cfg, batch, decode_steps, reps, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forward_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
        )
        return

    profile_forward(
        model=args.model,
        seq_len=args.seq_len,
        decode_steps=args.decode_steps or 8,
        batch=args.batch,
        sync_each_decode=not args.no_sync_each_decode,
    )


if __name__ == "__main__":
    main()
