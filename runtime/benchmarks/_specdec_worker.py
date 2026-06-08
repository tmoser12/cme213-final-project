#!/usr/bin/env python3
"""
runtime/benchmarks/_specdec_worker.py — GPU worker for `specdec_bench.py`.

Two modes, both driven by a JSON job file produced by the orchestrator:

  --mode single   (plain `python -m ...`)        one GPU, in-process:
      * baseline: target-only stochastic decode (the control)
      * single-GPU speculative decode for each γ  (draft graph ON)
  --mode mpi      (`mpirun -np 2 python -m ...`)  two GPUs, point-to-point:
      * rank 0 = 7B target on cuda:0, rank 1 = 0.5B draft on cuda:1 (draft graph ON)
      * speculative decode for each γ

The worker exists as a separate process so the single-GPU phase fully releases its
GPU memory/CUDA context (on process exit) before the orchestrator launches the
2-rank `mpirun` phase — no context coexistence on cuda:0.

Not meant to be run by hand; `specdec_bench.py` writes the job file and parses the
result file. Job/result schema is documented in `specdec_bench.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.buffers import allocate_buffers
from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights_on_gpu
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.sampler import logits_to_probs, sample_from_probs
from runtime.speculative.spec_decode import speculative_generate


def _load_executor(cfg: RuntimeConfig, device, *, use_cuda_graph: bool, max_seq_len: int):
    weights, _ = load_weights_on_gpu(cfg, batch=1, device=device)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=max_seq_len, device=device)
    return Qwen2Executor(cfg, weights, buffers, use_cuda_graph=use_cuda_graph)


def _timed(fn, *, warmup: int, trials: int):
    """Run `fn` warmup times (discarded) then `trials` timed; return (median_sec, last_result)."""
    for _ in range(warmup):
        fn()
    samples, result = [], None
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return median(samples), result


# --------------------------------------------------------------------------- single GPU

def _baseline_decode(target: Qwen2Executor, input_ids: torch.Tensor, n_new: int,
                     rng: random.Random) -> int:
    """Target-only ancestral (stochastic, temp=1) decode of n_new tokens. The control."""
    device = input_ids.device
    target.reset_kv_cache()
    logits = target.prefill(input_ids)
    nxt = sample_from_probs(logits_to_probs(logits[0, -1]), rng)
    for _ in range(n_new - 1):
        tok = torch.tensor([nxt], dtype=torch.int64, device=device)
        logits = target.decode_step(tok)  # eager: target is memory-bound, graph is 1.00x
        nxt = sample_from_probs(logits_to_probs(logits[0, -1]), rng)
    return n_new


def run_single(spec: dict, out_path: str) -> None:
    device = torch.device("cuda")
    n_new, seed, max_seq = spec["n_new"], spec["seed"], spec["max_seq"]
    warmup, trials, gammas = spec["warmup"], spec["trials"], spec["gammas"]

    cfg_t = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    cfg_d = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
    target = _load_executor(cfg_t, device, use_cuda_graph=False, max_seq_len=max_seq)
    draft_ex = _load_executor(cfg_d, device, use_cuda_graph=True, max_seq_len=max_seq)
    draft = DraftRunner(draft_ex, seed=seed)

    results = []
    for p in spec["prompts"]:
        ids = torch.tensor([p["ids"]], dtype=torch.int64, device=device)

        med, _ = _timed(lambda: _baseline_decode(target, ids, n_new, random.Random(seed)),
                        warmup=warmup, trials=trials)
        results.append({"id": p["id"], "category": p["category"], "config": "baseline",
                        "gamma": None, "tok_s": n_new / med, "accept_rate": None, "n_iters": None})
        print(f"[single] {p['id']:>10} baseline      : {n_new / med:6.1f} tok/s", flush=True)

        for g in gammas:
            med, r = _timed(
                lambda g=g: speculative_generate(target, draft, ids, n_new, g,
                                                 greedy=False, rng=random.Random(seed)),
                warmup=warmup, trials=trials)
            results.append({"id": p["id"], "category": p["category"], "config": "single",
                            "gamma": g, "tok_s": n_new / med,
                            "accept_rate": r.accept_rate, "n_iters": r.n_iters})
            print(f"[single] {p['id']:>10} spec   γ={g}   : {n_new / med:6.1f} tok/s "
                  f"(accept {r.accept_rate:.2f}/{g}, {r.n_iters} iters)", flush=True)

    Path(out_path).write_text(json.dumps({"results": results}))


# --------------------------------------------------------------------------- dual GPU (MPI)

def _mpi_run_job(comm, rank, role, prompt_ids, gamma, n_new, draft_vocab, rng, MPI):
    """One timed speculative generation across the two ranks. Returns (elapsed_or_None, accepted)."""
    from runtime.speculative.mpi_protocol import (
        DRAFT_RANK, TARGET_RANK, DraftPayload, TargetResult,
        recv_draft_payload, recv_target_result, send_draft_payload, send_target_result,
    )
    from runtime.speculative.target_step import flush_pending_bonus, target_speculative_step

    plen = len(prompt_ids)
    comm.Barrier()
    t0 = MPI.Wtime()
    accepted: list[int] = []

    if rank == TARGET_RANK:
        target = role
        prompt_t = torch.tensor([prompt_ids], dtype=torch.int64, device=target.device)
        target.reset_kv_cache()
        target.prefill(prompt_t)
        clen = plen
        while clen - plen < n_new:
            payload = recv_draft_payload(comm, source=DRAFT_RANK, gamma=gamma, vocab=draft_vocab)
            dids = torch.tensor([payload.draft_token_ids], dtype=torch.int64, device=target.device)
            dlog = torch.from_numpy(payload.draft_logits)  # CPU fp16 [γ+1, draft_vocab]
            res = target_speculative_step(target, dids, dlog, rng)
            send_target_result(comm, TargetResult(res.n_accepted, res.bonus_token,
                                                  res.prefix_len, res.cache_pos_after), dest=DRAFT_RANK)
            clen += res.n_accepted + 1
            accepted.append(res.n_accepted)
        flush_pending_bonus(target)
    else:
        draft = role
        prompt_t = torch.tensor([prompt_ids], dtype=torch.int64, device=draft._device)
        draft.prefill(prompt_t)
        clen = plen
        while clen - plen < n_new:
            dids_t, q_logits_t = draft.generate_drafts(gamma, greedy=False)
            ids = dids_t[0].cpu().numpy().astype(np.int64).tolist()
            q = q_logits_t.detach().to("cpu", torch.float16).numpy()
            send_draft_payload(comm, DraftPayload(draft_token_ids=ids, draft_logits=q), dest=TARGET_RANK)
            res = recv_target_result(comm, source=TARGET_RANK)
            clen += res.n_accepted + 1
            accepted.append(res.n_accepted)
            draft.apply_target_feedback(res.n_accepted, res.bonus_token)

    comm.Barrier()
    t1 = MPI.Wtime()
    return (t1 - t0) if rank == TARGET_RANK else None, accepted


def run_mpi(spec: dict, out_path: str) -> int:
    from mpi4py import MPI
    from runtime.speculative.mpi_protocol import DRAFT_RANK, TARGET_RANK

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        if rank == 0:
            print(f"error: --mode mpi needs exactly 2 ranks, got {size}", file=sys.stderr)
        return 1

    n_new, seed, max_seq = spec["n_new"], spec["seed"], spec["max_seq"]
    warmup, trials, gammas = spec["warmup"], spec["trials"], spec["gammas"]

    if rank >= torch.cuda.device_count():
        print(f"error: rank {rank} wants cuda:{rank}, only {torch.cuda.device_count()} GPU(s) "
              "visible (need --gres=gpu:2)", file=sys.stderr)
        return 1
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    draft_vocab = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT).vocab_size
    if rank == TARGET_RANK:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        role = _load_executor(cfg, device, use_cuda_graph=False, max_seq_len=max_seq)
    else:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        draft_ex = _load_executor(cfg, device, use_cuda_graph=True, max_seq_len=max_seq)
        role = DraftRunner(draft_ex, seed=seed + DRAFT_RANK)
    rng = random.Random(seed)

    results = []
    for p in spec["prompts"]:
        for g in gammas:
            for _ in range(warmup):
                _mpi_run_job(comm, rank, role, p["ids"], g, n_new, draft_vocab, rng, MPI)
            samples, last_acc = [], []
            for _ in range(trials):
                el, acc = _mpi_run_job(comm, rank, role, p["ids"], g, n_new, draft_vocab, rng, MPI)
                if el is not None:
                    samples.append(el)
                last_acc = acc
            if rank == TARGET_RANK:
                med = median(samples)
                rate = (sum(last_acc) / len(last_acc)) if last_acc else 0.0
                results.append({"id": p["id"], "category": p["category"], "config": "mpi",
                                "gamma": g, "tok_s": n_new / med,
                                "accept_rate": rate, "n_iters": len(last_acc)})
                print(f"[mpi]    {p['id']:>10} spec   γ={g}   : {n_new / med:6.1f} tok/s "
                      f"(accept {rate:.2f}/{g}, {len(last_acc)} iters)", flush=True)

    if rank == TARGET_RANK:
        Path(out_path).write_text(json.dumps({"results": results}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "mpi"], required=True)
    ap.add_argument("jobfile")
    ap.add_argument("resultfile")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("error: no CUDA device (run on a GPU node)", file=sys.stderr)
        return 2

    spec = json.loads(Path(args.jobfile).read_text())
    if args.mode == "single":
        run_single(spec, args.resultfile)
        return 0
    return run_mpi(spec, args.resultfile)


if __name__ == "__main__":
    raise SystemExit(main())
