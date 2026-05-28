from __future__ import annotations

import json
import sys

import torch
from mpi4py import MPI

from .config import parse_config
from .utils import load_prompt, prompt_sha256, rank_log, summarize_result
from .worker_draft import run_draft_worker
from .worker_target import run_target_worker


def main() -> int:
    config = parse_config()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != 2:
        if rank == 0:
            print(
                f"ERROR: expected exactly 2 MPI ranks, got {world_size}.",
                file=sys.stderr,
                flush=True,
            )
        return 2

    if not torch.cuda.is_available():
        if rank == 0:
            print("ERROR: CUDA is unavailable. This baseline requires GPUs.", file=sys.stderr)
        return 3

    if torch.cuda.device_count() < 2:
        if rank == 0:
            print(
                f"ERROR: found {torch.cuda.device_count()} visible GPUs, expected at least 2.",
                file=sys.stderr,
            )
        return 4

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    role = "draft" if rank == 0 else "target"
    rank_log(rank, f"Initialized as role={role}, device={device}, world_size={world_size}")

    prompt = load_prompt(config.prompt_text, config.prompt_file) if rank == 0 else None
    prompt = comm.bcast(prompt, root=0)
    prompt_hash = prompt_sha256(prompt)
    rank_log(rank, f"Prompt broadcast complete, sha256={prompt_hash[:12]}...")

    # Synchronize before timing-heavy model work for cleaner rank comparisons.
    comm.Barrier()
    if rank == 0:
        result = run_draft_worker(config, rank=rank, device=device, prompt=prompt)
    else:
        result = run_target_worker(config, rank=rank, device=device, prompt=prompt)
    comm.Barrier()

    gathered = comm.gather(result, root=0)
    rank_log(rank, f"Finished worker run: {summarize_result(result)}")

    if rank == 0:
        print("=== MPI Minimal Baseline Summary ===", flush=True)
        print(f"prompt_sha256={prompt_hash}", flush=True)
        print(f"prompt_chars={len(prompt)}", flush=True)
        for item in sorted(gathered, key=lambda x: x["rank"]):
            print(json.dumps(item, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

