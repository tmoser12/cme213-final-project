#!/usr/bin/env python3
"""Two-rank MPI loop mimicking speculative decoding without real models.

Rank 0 (target): receive draft payload -> mock verify -> host accept/reject -> reply.
Rank 1 (draft):  generate gamma draft tokens -> send payload -> receive sync result.

Usage:
  mpirun -np 2 python -m mpi_prototype.main --steps 5 --gamma 4
  bash mpi_prototype/run_mpi.sh --steps 10
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
from mpi4py import MPI

from mpi_prototype.gpu_check import bind_rank_to_cuda, log_gpu_binding, verify_distinct_gpus
from mpi_prototype.mock_models import MockDraftModel, MockTargetModel, sync_draft_prefix
from mpi_prototype.protocol import (
    DRAFT_RANK,
    TAG_TARGET_RESULT,
    TARGET_RANK,
    recv_draft_payload,
    send_draft_payload,
    pack_target_result,
    unpack_target_result,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock MPI speculative decoding")
    p.add_argument("--gamma", type=int, default=4, help="speculative tokens per step")
    p.add_argument("--steps", type=int, default=5, help="iterations to run")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (rank-offset per rank)")
    p.add_argument("--vocab", type=int, default=64, help="mock vocabulary size")
    p.add_argument(
        "--prefix",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="initial token ids shared by both ranks",
    )
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="bind rank r to cuda:r, run a GPU probe, fail if ranks share a device",
    )
    return p.parse_args(argv)


def _draft_loop(
    comm: MPI.Intracomm,
    *,
    gamma: int,
    steps: int,
    vocab: int,
    prefix: list[int],
    seed: int,
) -> list[int]:
    draft = MockDraftModel(vocab_size=vocab, prefix=list(prefix))
    rng = random.Random(seed + DRAFT_RANK)

    for step in range(steps):
        payload = draft.draft_gamma(gamma, rng)
        send_draft_payload(comm, payload, dest=TARGET_RANK)

        result_buf = np.empty(4, dtype=np.int32)
        comm.Recv(result_buf, source=TARGET_RANK, tag=TAG_TARGET_RESULT)
        result = unpack_target_result(result_buf)

        sync_draft_prefix(draft, payload.draft_token_ids, result)

        if step == 0 or step == steps - 1:
            print(
                f"[draft step {step}] sent gamma={gamma} "
                f"accepted={result.n_accepted} bonus={result.bonus_token} "
                f"prefix_len={len(draft.prefix)}",
                flush=True,
            )

    return draft.prefix


def _target_loop(
    comm: MPI.Intracomm,
    *,
    gamma: int,
    steps: int,
    vocab: int,
    prefix: list[int],
    seed: int,
) -> list[int]:
    target = MockTargetModel(vocab_size=vocab, prefix=list(prefix))
    rng = random.Random(seed + TARGET_RANK)

    for step in range(steps):
        payload = recv_draft_payload(comm, source=DRAFT_RANK, gamma=gamma, vocab=vocab)

        result = target.verify_and_accept(payload, rng)
        comm.Send(pack_target_result(result), dest=DRAFT_RANK, tag=TAG_TARGET_RESULT)

        if step == 0 or step == steps - 1:
            print(
                f"[target step {step}] verified gamma={len(payload.draft_token_ids)} "
                f"accepted={result.n_accepted} bonus={result.bonus_token} "
                f"prefix_len={len(target.prefix)} pending_bonus={target._pending_bonus}",
                flush=True,
            )

    target.flush_pending_bonus()
    return target.prefix


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

    gpu_binding = None
    if args.require_gpu:
        try:
            gpu_binding = bind_rank_to_cuda(rank)
        except RuntimeError as exc:
            print(f"[rank {rank}] GPU setup failed: {exc}", file=sys.stderr, flush=True)
            return 3
        role = "target" if rank == TARGET_RANK else "draft"
        log_gpu_binding(gpu_binding, role=role)
        comm.Barrier()
        try:
            verify_distinct_gpus(comm, gpu_binding)
        except RuntimeError as exc:
            print(f"[rank {rank}] {exc}", file=sys.stderr, flush=True)
            return 3
        comm.Barrier()

    if rank == TARGET_RANK:
        final_prefix = _target_loop(
            comm,
            gamma=args.gamma,
            steps=args.steps,
            vocab=args.vocab,
            prefix=list(args.prefix),
            seed=args.seed,
        )
    elif rank == DRAFT_RANK:
        final_prefix = _draft_loop(
            comm,
            gamma=args.gamma,
            steps=args.steps,
            vocab=args.vocab,
            prefix=list(args.prefix),
            seed=args.seed,
        )
    else:
        return 1

    comm.Barrier()

    # Rank 0 gathers both prefixes for a final consistency check.
    all_prefixes = comm.gather(final_prefix, root=TARGET_RANK)
    if rank == TARGET_RANK:
        draft_prefix = all_prefixes[DRAFT_RANK]
        if draft_prefix != final_prefix:
            print(
                f"error: prefix mismatch target={final_prefix} draft={draft_prefix}",
                file=sys.stderr,
            )
            return 2
        print(
            f"OK: {args.steps} steps, gamma={args.gamma}, "
            f"final prefix len={len(final_prefix)} tokens={final_prefix}"
            + (" (2-GPU verified)" if args.require_gpu else ""),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
