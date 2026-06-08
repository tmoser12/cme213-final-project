#!/usr/bin/env python3
"""Two-rank MPI speculative decoding coordinator (Phase 8c).

Rank 0 (target, cuda:0): 7B `Qwen2Executor`. recv draft payload → `target_speculative_step`
  (verify + host accept/reject + bonus) → send result.
Rank 1 (draft,  cuda:1): 0.5B draft via `DraftRunner` (CUDA-graphed). generate γ drafts →
  send payload → recv result → `apply_target_feedback`.

This replaces the in-process draft payload of `spec_decode.py` with MPI messages, so the
draft (GPU 1) generates the next γ while the target (GPU 0) verifies — overlapping the two
models across GPUs instead of serializing them on one card.

mpi4py is imported lazily in `main()` so importing this module (e.g. package discovery)
does not initialize MPI. Launch with `slurm/run_speculative.sh` (mpirun -np 2).

  bash slurm/run_speculative.sh --slurm-gpu --steps 32 --gamma 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np
import torch

from runtime.buffers import allocate_buffers
from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights_on_gpu
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.mpi_protocol import (
    DRAFT_RANK,
    TARGET_RANK,
    DraftPayload,
    recv_draft_payload,
    recv_target_result,
    send_draft_payload,
    send_target_result,
    TargetResult,
)
from runtime.speculative.target_step import flush_pending_bonus, target_speculative_step

# Shared prompt both ranks prefill (Qwen2.5 chat preamble tokens).
DEFAULT_PROMPT = [151643, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 220]


def _bind_device(rank: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run on a GPU node (--gres=gpu:2)")
    if rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"rank {rank} wants cuda:{rank} but only {torch.cuda.device_count()} GPU(s) visible "
            "(need --gres=gpu:2)"
        )
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _load_executor(cfg: RuntimeConfig, device: torch.device, *, use_cuda_graph: bool,
                   max_seq_len: int) -> Qwen2Executor:
    weights, _ = load_weights_on_gpu(cfg, batch=1, device=device)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=max_seq_len, device=device)
    return Qwen2Executor(cfg, weights, buffers, use_cuda_graph=use_cuda_graph)


def _target_case(
    comm,
    *,
    target: Qwen2Executor,
    rng: random.Random,
    prompt_tokens: list[int],
    gamma: int,
    steps: int,
    draft_vocab: int,
    log_endpoints: bool = True,
) -> tuple[list[int], list[int]]:
    prompt = torch.tensor([prompt_tokens], dtype=torch.int64, device=target.device)
    target.prefill(prompt)

    committed = list(prompt_tokens)
    accepted_counts: list[int] = []
    for step in range(steps):
        payload = recv_draft_payload(comm, source=DRAFT_RANK, gamma=gamma, vocab=draft_vocab)
        draft_ids = torch.tensor(payload.draft_token_ids, dtype=torch.int64, device=target.device)
        draft_logits = torch.from_numpy(payload.draft_logits)  # CPU fp16 [γ+1, vocab]

        result = target_speculative_step(target, draft_ids, draft_logits, rng)
        send_target_result(
            comm,
            TargetResult(result.n_accepted, result.bonus_token,
                         result.prefix_len, result.cache_pos_after),
            dest=DRAFT_RANK,
        )
        committed += payload.draft_token_ids[: result.n_accepted] + [result.bonus_token]
        accepted_counts.append(result.n_accepted)
        if log_endpoints and step in (0, steps - 1):
            print(f"[target step {step}] accepted={result.n_accepted}/{gamma} "
                  f"bonus={result.bonus_token} cache_pos={target.cache_pos}", flush=True)

    flush_pending_bonus(target)
    return committed, accepted_counts


def _draft_case(
    comm,
    *,
    draft: DraftRunner,
    prompt_tokens: list[int],
    gamma: int,
    steps: int,
    greedy: bool,
    log_endpoints: bool = True,
) -> list[int]:
    prompt = torch.tensor([prompt_tokens], dtype=torch.int64, device=draft.executor.device)
    draft.prefill(prompt)

    committed = list(prompt_tokens)
    for step in range(steps):
        draft_ids_t, q_logits_t = draft.generate_drafts(gamma, greedy=greedy)
        ids = draft_ids_t[0].cpu().numpy().astype(np.int64).tolist()
        q = q_logits_t.detach().to("cpu", torch.float16).numpy()
        send_draft_payload(comm, DraftPayload(draft_token_ids=ids, draft_logits=q), dest=TARGET_RANK)

        result = recv_target_result(comm, source=TARGET_RANK)
        committed += ids[: result.n_accepted] + [result.bonus_token]
        draft.apply_target_feedback(result.n_accepted, result.bonus_token)
        if log_endpoints and step in (0, steps - 1):
            print(f"[draft step {step}] accepted={result.n_accepted}/{gamma} "
                  f"bonus={result.bonus_token} prefix_len={draft.prefix_len}", flush=True)

    return committed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2-rank MPI speculative decoding (7B target + 0.5B draft)")
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--steps", type=int, default=16, help="speculative iterations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq", type=int, default=512, dest="max_seq")
    p.add_argument("--greedy", action="store_true", help="argmax draft sampling")
    p.add_argument("--benchmark-json", action="store_true", help="Emit machine-readable benchmark summary on rank 0")
    p.add_argument(
        "--benchmark-suite-json",
        type=str,
        default="",
        help="Path to benchmark suite JSON (multiple prompt/trial cases in one persistent MPI run)",
    )
    p.add_argument("--prompt", type=int, nargs="*", default=DEFAULT_PROMPT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from mpi4py import MPI  # lazy: importing this module must not init MPI

    args = _parse_args(argv)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        if rank == 0:
            print(f"error: expected 2 MPI ranks, got {size}", file=sys.stderr)
        return 1
    if not (1 <= args.gamma <= 7):
        if rank == 0:
            print("error: --gamma must be in [1, 7]", file=sys.stderr)
        return 1

    draft_vocab = RuntimeConfig.from_yaml(CONFIG_05B).vocab_size
    target = None
    draft = None
    if rank == TARGET_RANK:
        target = _load_executor(
            RuntimeConfig.from_yaml(CONFIG_7B),
            _bind_device(TARGET_RANK),
            use_cuda_graph=True,
            max_seq_len=args.max_seq,
        )
    else:
        draft_ex = _load_executor(
            RuntimeConfig.from_yaml(CONFIG_05B),
            _bind_device(DRAFT_RANK),
            use_cuda_graph=True,
            max_seq_len=args.max_seq,
        )
        draft = DraftRunner(draft_ex, seed=args.seed + DRAFT_RANK)

    if args.benchmark_suite_json:
        with open(args.benchmark_suite_json, "r", encoding="utf-8") as f:
            suite = json.loads(f.read())
        cases = suite.get("cases", [])
        if not cases:
            if rank == TARGET_RANK:
                print("error: benchmark suite has no cases", file=sys.stderr)
            return 1

        for idx, case in enumerate(cases):
            prompt_tokens = [int(x) for x in case["prompt"]]
            steps = int(case["steps"])
            seed = int(case["seed"])
            t0 = MPI.Wtime()
            if rank == TARGET_RANK:
                committed, accepted_counts = _target_case(
                    comm,
                    target=target,
                    rng=random.Random(seed + TARGET_RANK),
                    prompt_tokens=prompt_tokens,
                    gamma=args.gamma,
                    steps=steps,
                    draft_vocab=draft_vocab,
                    log_endpoints=False,
                )
            else:
                committed = _draft_case(
                    comm,
                    draft=draft,
                    prompt_tokens=prompt_tokens,
                    gamma=args.gamma,
                    steps=steps,
                    greedy=args.greedy,
                    log_endpoints=False,
                )
            comm.Barrier()
            t1 = MPI.Wtime()
            all_committed = comm.gather(committed, root=TARGET_RANK)
            if rank == TARGET_RANK:
                if committed != all_committed[DRAFT_RANK]:
                    print(
                        f"error: committed-sequence mismatch in suite case {idx}",
                        file=sys.stderr,
                    )
                    return 2
                n_new = len(committed) - len(prompt_tokens)
                elapsed_s = max(t1 - t0, 1e-9)
                accepted_total = sum(accepted_counts)
                accept_per_iter = accepted_total / steps if steps else 0.0
                accept_ratio = accept_per_iter / args.gamma if args.gamma else 0.0
                tokens_per_s = n_new / elapsed_s
                print(
                    f"OK(case {idx}): steps={steps}, γ={args.gamma}, n_new={n_new}, "
                    f"tok/s={tokens_per_s:.2f}, acc={accept_per_iter:.2f}/{args.gamma}",
                    flush=True,
                )
                if args.benchmark_json:
                    print(
                        "BENCHMARK_JSON: "
                        + json.dumps(
                            {
                                "case_index": idx,
                                "prompt_id": case.get("prompt_id", f"case_{idx}"),
                                "category": case.get("category", "unknown"),
                                "prompt_len": len(prompt_tokens),
                                "trial_idx": int(case.get("trial_idx", 0)),
                                "steps": steps,
                                "gamma": args.gamma,
                                "n_generated": n_new,
                                "elapsed_s": elapsed_s,
                                "tokens_per_s": tokens_per_s,
                                "ms_per_token": 1000.0 * elapsed_s / max(n_new, 1),
                                "accepted_total": accepted_total,
                                "accept_per_iter": accept_per_iter,
                                "accept_ratio": accept_ratio,
                                "tokens_per_iter": (n_new / steps) if steps else 0.0,
                                "iters_per_token": (steps / n_new) if n_new else 0.0,
                                "target_cuda_graph": True,
                                "draft_cuda_graph": True,
                            }
                        ),
                        flush=True,
                    )
    else:
        t0 = MPI.Wtime()
        if rank == TARGET_RANK:
            committed, accepted_counts = _target_case(
                comm,
                target=target,
                rng=random.Random(args.seed + TARGET_RANK),
                prompt_tokens=args.prompt,
                gamma=args.gamma,
                steps=args.steps,
                draft_vocab=draft_vocab,
                log_endpoints=True,
            )
        else:
            committed = _draft_case(
                comm,
                draft=draft,
                prompt_tokens=args.prompt,
                gamma=args.gamma,
                steps=args.steps,
                greedy=args.greedy,
                log_endpoints=True,
            )

        comm.Barrier()
        t1 = MPI.Wtime()
        all_committed = comm.gather(committed, root=TARGET_RANK)
        if rank == TARGET_RANK:
            target_seq, draft_seq = committed, all_committed[DRAFT_RANK]
            if target_seq != draft_seq:
                print(f"error: committed-sequence mismatch\n  target={target_seq}\n  draft={draft_seq}",
                      file=sys.stderr)
                return 2
            n_new = len(committed) - len(args.prompt)
            elapsed_s = max(t1 - t0, 1e-9)
            accepted_total = sum(accepted_counts)
            accept_per_iter = accepted_total / args.steps if args.steps else 0.0
            accept_ratio = accept_per_iter / args.gamma if args.gamma else 0.0
            tokens_per_s = n_new / elapsed_s
            print(f"OK: {args.steps} iters, γ={args.gamma}, generated {n_new} tokens "
                  f"({n_new / args.steps:.2f}/iter), draft+target sequences match. "
                  f"tok/s={tokens_per_s:.2f}, acc={accept_per_iter:.2f}/{args.gamma}",
                  flush=True)
            if args.benchmark_json:
                print(
                    "BENCHMARK_JSON: "
                    + json.dumps(
                        {
                            "steps": args.steps,
                            "gamma": args.gamma,
                            "n_generated": n_new,
                            "elapsed_s": elapsed_s,
                            "tokens_per_s": tokens_per_s,
                            "ms_per_token": 1000.0 * elapsed_s / max(n_new, 1),
                            "accepted_total": accepted_total,
                            "accept_per_iter": accept_per_iter,
                            "accept_ratio": accept_ratio,
                            "tokens_per_iter": (n_new / args.steps) if args.steps else 0.0,
                            "iters_per_token": (args.steps / n_new) if n_new else 0.0,
                            "target_cuda_graph": True,
                            "draft_cuda_graph": True,
                        }
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
