#!/usr/bin/env python3
"""MPI transfer benchmark matching runtime speculative-decoding wire payloads.

Payloads (see runtime/plan.md and test_mpi/protocol.py):
  Draft -> target: int32[gamma], float16[gamma+1, vocab] (logits sent as uint8)
  Target -> draft: int32[4] (n_accepted, bonus_token, prefix_len, cache_pos_after)

Each rank binds to cuda:{rank}. Draft logits are allocated on GPU, copied D2H
before MPI send (production path). Target optionally H2D's received logits to
cuda:0 after recv.

Usage:
  bash test_mpi/run_mpi.sh --slurm-gpu --benchmark
  mpirun -np 2 python -m test_mpi.benchmark --trials 50
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch
from mpi4py import MPI

from runtime.core.config import CONFIG_7B, RuntimeConfig
from test_mpi.gpu_check import bind_rank_to_cuda, log_gpu_binding, verify_distinct_gpus
from test_mpi.protocol import (
    DRAFT_RANK,
    TARGET_RANK,
    DraftPayload,
    TargetResult,
    draft_logits_byte_count,
    recv_draft_payload,
    recv_target_result,
    send_draft_payload,
    send_target_result,
)


@dataclass(frozen=True)
class DirectionStats:
    label: str
    bytes_moved: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    bandwidth_gbps: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = RuntimeConfig.from_yaml(CONFIG_7B)
    p = argparse.ArgumentParser(description="MPI payload benchmark (2 ranks, 2 GPUs)")
    p.add_argument("--gamma", type=int, default=4, help="speculative tokens per step")
    p.add_argument(
        "--vocab",
        type=int,
        default=cfg.vocab_size,
        help=f"vocabulary size (default: {cfg.vocab_size} from qwen2.5-7b.yaml)",
    )
    p.add_argument("--warmup", type=int, default=5, help="untimed warmup iterations")
    p.add_argument("--trials", type=int, default=30, help="timed iterations")
    p.add_argument(
        "--h2d-after-recv",
        action="store_true",
        default=True,
        help="target copies received logits to cuda:0 after MPI recv (default: on)",
    )
    p.add_argument(
        "--no-h2d-after-recv",
        action="store_false",
        dest="h2d_after_recv",
        help="skip target H2D after draft payload recv",
    )
    return p.parse_args(argv)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _gather_times(comm: MPI.Intracomm, local: list[float]) -> list[float]:
    """Root gets concatenated timings from all ranks (non-root returns [])."""
    gathered = comm.gather(local, root=TARGET_RANK)
    if comm.Get_rank() != TARGET_RANK:
        return []
    return [t for part in gathered for t in part]


def _summarize(label: str, times_ms: list[float], nbytes: int) -> DirectionStats | None:
    if not times_ms:
        return None
    mean_s = sum(times_ms) / len(times_ms) / 1000.0
    bw = (nbytes / mean_s / 1e9) if mean_s > 0 else 0.0
    return DirectionStats(
        label=label,
        bytes_moved=nbytes,
        mean_ms=sum(times_ms) / len(times_ms),
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        p50_ms=_percentile(times_ms, 50),
        bandwidth_gbps=bw,
    )


def _format_stats(stats: DirectionStats | None) -> str:
    if stats is None:
        return "(no samples)"
    kb = stats.bytes_moved / 1024
    return (
        f"{stats.label}: {stats.bytes_moved:,} B ({kb:.1f} KiB) | "
        f"mean={stats.mean_ms:.3f} ms p50={stats.p50_ms:.3f} ms "
        f"min={stats.min_ms:.3f} max={stats.max_ms:.3f} ms | "
        f"{stats.bandwidth_gbps:.2f} GB/s"
    )


def _make_gpu_payload(
    *,
    gamma: int,
    vocab: int,
    device: torch.device,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU buffers mirroring draft executor outputs."""
    ids = torch.randint(
        0,
        vocab,
        (gamma,),
        dtype=torch.int32,
        device=device,
    )
    # Unique-ish fp16 logits so D2H/MPI cannot be optimized away.
    logits = torch.randn(gamma + 1, vocab, dtype=torch.float16, device=device)
    logits.add_(float(step) * 1e-4)
    return ids, logits


def _draft_to_target_bytes(gamma: int, vocab: int) -> int:
    return gamma * np.dtype(np.int32).itemsize + draft_logits_byte_count(gamma, vocab)


def _run_draft_to_target(
    comm: MPI.Intracomm,
    *,
    gamma: int,
    vocab: int,
    warmup: int,
    trials: int,
    h2d_after_recv: bool,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (draft_d2h_ms, draft_mpi_ms, target_mpi_ms, target_h2d_ms) per trial."""
    rank = comm.Get_rank()
    device = torch.device(f"cuda:{rank}")

    draft_d2h_ms: list[float] = []
    draft_mpi_ms: list[float] = []
    target_mpi_ms: list[float] = []
    target_h2d_ms: list[float] = []

    total_iters = warmup + trials
    for i in range(total_iters):
        if rank == DRAFT_RANK:
            ids_gpu, logits_gpu = _make_gpu_payload(
                gamma=gamma, vocab=vocab, device=device, step=i
            )
            torch.cuda.synchronize(device)
            t0 = MPI.Wtime()
            ids_cpu = ids_gpu.cpu().numpy()
            logits_cpu = logits_gpu.cpu().numpy()
            torch.cuda.synchronize(device)
            t1 = MPI.Wtime()

            payload = DraftPayload(
                draft_token_ids=ids_cpu.astype(np.int64).tolist(),
                draft_logits=np.ascontiguousarray(logits_cpu, dtype=np.float16),
            )
            t2 = MPI.Wtime()
            send_draft_payload(comm, payload, dest=TARGET_RANK)
            t3 = MPI.Wtime()

            if i >= warmup:
                draft_d2h_ms.append((t1 - t0) * 1000.0)
                draft_mpi_ms.append((t3 - t2) * 1000.0)

        elif rank == TARGET_RANK:
            t0 = MPI.Wtime()
            payload = recv_draft_payload(
                comm, source=DRAFT_RANK, gamma=gamma, vocab=vocab
            )
            t1 = MPI.Wtime()

            h2d_ms = 0.0
            if h2d_after_recv:
                t2 = MPI.Wtime()
                _ = torch.from_numpy(payload.draft_logits).to(device, non_blocking=False)
                torch.cuda.synchronize(device)
                t3 = MPI.Wtime()
                h2d_ms = (t3 - t2) * 1000.0

            if i >= warmup:
                target_mpi_ms.append((t1 - t0) * 1000.0)
                target_h2d_ms.append(h2d_ms)

        comm.Barrier()

    return draft_d2h_ms, draft_mpi_ms, target_mpi_ms, target_h2d_ms


def _run_target_to_draft(
    comm: MPI.Intracomm,
    *,
    warmup: int,
    trials: int,
) -> tuple[list[float], list[float]]:
    """Return (target_send_ms, draft_recv_ms) per trial."""
    rank = comm.Get_rank()
    target_send_ms: list[float] = []
    draft_recv_ms: list[float] = []

    total_iters = warmup + trials
    for i in range(total_iters):
        result = TargetResult(
            n_accepted=i % 4,
            bonus_token=1000 + i,
            prefix_len=128 + i,
            cache_pos_after=130 + i,
        )
        if rank == TARGET_RANK:
            t0 = MPI.Wtime()
            send_target_result(comm, result, dest=DRAFT_RANK)
            t1 = MPI.Wtime()
            if i >= warmup:
                target_send_ms.append((t1 - t0) * 1000.0)
        elif rank == DRAFT_RANK:
            t0 = MPI.Wtime()
            _ = recv_target_result(comm, source=TARGET_RANK)
            t1 = MPI.Wtime()
            if i >= warmup:
                draft_recv_ms.append((t1 - t0) * 1000.0)
        comm.Barrier()

    return target_send_ms, draft_recv_ms


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print(f"error: expected 2 MPI ranks, got {size}", file=sys.stderr)
        return 1

    if args.gamma < 1 or args.gamma > 7:
        if rank == 0:
            print("error: --gamma must be in [1, 7]", file=sys.stderr)
        return 1

    try:
        binding = bind_rank_to_cuda(rank)
    except RuntimeError as exc:
        print(f"[rank {rank}] GPU setup failed: {exc}", file=sys.stderr, flush=True)
        return 3

    role = "target" if rank == TARGET_RANK else "draft"
    log_gpu_binding(binding, role=role)
    comm.Barrier()
    try:
        verify_distinct_gpus(comm, binding)
    except RuntimeError as exc:
        print(f"[rank {rank}] {exc}", file=sys.stderr, flush=True)
        return 3
    comm.Barrier()

    draft_bytes = _draft_to_target_bytes(args.gamma, args.vocab)
    target_bytes = 4 * np.dtype(np.int32).itemsize

    if rank == TARGET_RANK:
        print(
            f"MPI benchmark: gamma={args.gamma} vocab={args.vocab} "
            f"warmup={args.warmup} trials={args.trials} "
            f"h2d_after_recv={args.h2d_after_recv}",
            flush=True,
        )
        print(
            f"Payload sizes: draft->target {draft_bytes:,} B | "
            f"target->draft {target_bytes} B",
            flush=True,
        )

    (
        draft_d2h_ms,
        draft_mpi_ms,
        target_mpi_ms,
        target_h2d_ms,
    ) = _run_draft_to_target(
        comm,
        gamma=args.gamma,
        vocab=args.vocab,
        warmup=args.warmup,
        trials=args.trials,
        h2d_after_recv=args.h2d_after_recv,
    )

    target_send_ms, draft_recv_ms = _run_target_to_draft(
        comm, warmup=args.warmup, trials=args.trials
    )

    comm.Barrier()

    draft_d2h_ms = _gather_times(comm, draft_d2h_ms)
    draft_mpi_ms = _gather_times(comm, draft_mpi_ms)
    target_mpi_ms = _gather_times(comm, target_mpi_ms)
    target_h2d_ms = _gather_times(comm, target_h2d_ms)
    target_send_ms = _gather_times(comm, target_send_ms)
    draft_recv_ms = _gather_times(comm, draft_recv_ms)

    if rank == TARGET_RANK:
        draft_mpi = _summarize("draft MPI send (blocking)", draft_mpi_ms, draft_bytes)
        target_mpi = _summarize("target MPI recv", target_mpi_ms, draft_bytes)
        d2h = _summarize("draft D2H (ids+logits)", draft_d2h_ms, draft_bytes)
        if args.h2d_after_recv:
            h2d = _summarize("target H2D (logits)", target_h2d_ms, draft_bytes)
        t2d_send = _summarize("target MPI send", target_send_ms, target_bytes)
        t2d_recv = _summarize("draft MPI recv", draft_recv_ms, target_bytes)

        print("\n--- Draft -> target (dominant payload) ---", flush=True)
        print(_format_stats(d2h), flush=True)
        print(_format_stats(draft_mpi), flush=True)
        print(_format_stats(target_mpi), flush=True)
        if args.h2d_after_recv:
            print(_format_stats(h2d), flush=True)

        draft_side_total = [
            d + m for d, m in zip(draft_d2h_ms, draft_mpi_ms, strict=True)
        ]
        target_side_total = list(target_mpi_ms)
        if args.h2d_after_recv:
            target_side_total = [
                m + h for m, h in zip(target_mpi_ms, target_h2d_ms, strict=True)
            ]
        iter_draft = _summarize(
            "draft-side total (D2H + MPI send)", draft_side_total, draft_bytes
        )
        iter_target = _summarize(
            "target-side total (MPI recv + H2D)", target_side_total, draft_bytes
        )
        print(_format_stats(iter_draft), flush=True)
        print(_format_stats(iter_target), flush=True)

        print("\n--- Target -> draft (sync metadata) ---", flush=True)
        print(_format_stats(t2d_send), flush=True)
        print(_format_stats(t2d_recv), flush=True)

        combined_ms = [
            d + t + s + r
            for d, t, s, r in zip(
                draft_side_total,
                target_side_total,
                target_send_ms,
                draft_recv_ms,
                strict=True,
            )
        ]
        round_trip = _summarize(
            "full iteration (both directions, both ranks)",
            combined_ms,
            draft_bytes + target_bytes,
        )
        print("\n--- One speculative iteration (approx) ---", flush=True)
        print(_format_stats(round_trip), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
