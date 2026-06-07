#!/usr/bin/env python3
"""
runtime/benchmarks/spec_decode_bench.py — end-to-end speculative decoding throughput.

Compares, on one GPU (7B target + 0.5B draft):
  * target-only greedy decode (the baseline)
  * speculative decode (greedy), draft graph OFF
  * speculative decode (greedy), draft graph ON   <- the draft-graph win in context

Reports tokens/sec, accept rate, and iterations. Spec-decode throughput depends on
the accept rate (draft/target agreement), so this is reported alongside.

Run: bash slurm/run_python.sh runtime/benchmarks/spec_decode_bench.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.spec_decode import speculative_generate

DEVICE = "cuda"
MAX_SEQ = 512
N_NEW = 96
GAMMA = 4
PROMPT = [[151643, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 220]]


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.")
        return 2

    cfg_t = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    cfg_d = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    wt, _ = load_weights_on_gpu(cfg_t, batch=1, device=DEVICE)
    bt = allocate_buffers(cfg_t, batch=1, max_seq_len=MAX_SEQ, device=DEVICE)
    target = Qwen2Executor(cfg_t, wt, bt)
    wd, _ = load_weights_on_gpu(cfg_d, batch=1, device=DEVICE)
    bd = allocate_buffers(cfg_d, batch=1, max_seq_len=MAX_SEQ, device=DEVICE)
    draft_ex = Qwen2Executor(cfg_d, wd, bd, use_cuda_graph=True)
    draft = DraftRunner(draft_ex, seed=0)
    prompt = torch.tensor(PROMPT, device=DEVICE)
    plen = prompt.shape[1]

    def _time(fn):
        fn()  # warmup (also triggers draft graph capture)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, r

    # Target-only greedy
    dt_t, _ = _time(lambda: target.greedy_extend(prompt, N_NEW))
    tps_t = N_NEW / dt_t

    # Spec decode, draft graph OFF
    draft_ex.use_cuda_graph = False
    dt_so, r_so = _time(lambda: speculative_generate(target, draft, prompt, N_NEW, GAMMA, greedy=True))
    tps_so = N_NEW / dt_so

    # Spec decode, draft graph ON
    draft_ex.use_cuda_graph = True
    dt_sg, r_sg = _time(lambda: speculative_generate(target, draft, prompt, N_NEW, GAMMA, greedy=True))
    tps_sg = N_NEW / dt_sg

    print("\n================ end-to-end decode tokens/sec ================")
    print(f"  target-only greedy        : {1000*dt_t/N_NEW:6.2f} ms/tok  {tps_t:6.1f} tok/s   (1.00x)")
    print(f"  spec-decode (draft graph OFF): {1000*dt_so/N_NEW:6.2f} ms/tok  {tps_so:6.1f} tok/s   {tps_so/tps_t:.2f}x")
    print(f"  spec-decode (draft graph ON ): {1000*dt_sg/N_NEW:6.2f} ms/tok  {tps_sg:6.1f} tok/s   {tps_sg/tps_t:.2f}x")
    print(f"\n  accept rate: {r_sg.accept_rate:.2f}/{GAMMA} per iter over {r_sg.n_iters} iters")
    print(f"  draft-graph speedup within spec: {tps_sg/tps_so:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
