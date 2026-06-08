#!/usr/bin/env python3
"""Diagnose the split-KV decode NaN.

(A) Isolate the draft decode_attn kernel vs PyTorch SDPA across cur_len — the
    split/combine math is bounded, so its OUTPUT must be finite and match SDPA.
    This proves whether the kernel itself emits NaN/Inf.

(B) Reproduce the benchmark's eager-vs-graph decode and report isnan/isinf for
    eager and graph logits SEPARATELY, on a RANDOM prompt (as in
    draft_graph_decode.py) and a REAL prompt. If both paths overflow to Inf on
    the random prompt (eager is single-block at this length, so that is NOT the
    split kernel) and the real prompt is clean, the benchmark's `max|Δ| = nan`
    is fp16 logit overflow (inf-inf), not a kernel bug.

Run: bash slurm/run_python.sh runtime/benchmarks/diag_split_decode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.core.config import RuntimeConfig, CONFIG_05B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor
from runtime.production_kernels.draft.attention import ops as dattn

DEVICE = "cuda"
MAX_SEQ = 512


def _num_splits(cur_len: int, h_q: int, B: int) -> int:
    """Mirror choose_num_splits() so we can label which path each cur_len takes."""
    heads = h_q * B
    desired = (144 + heads - 1) // heads
    mbl = (cur_len + 255) // 256
    return max(1, min(desired, mbl, 32))


def isolate() -> bool:
    B, hq, hkv, D = 1, 14, 2, 64
    scale = 1.0 / (D ** 0.5)
    print("=== (A) draft decode_attn_forward vs SDPA (no RoPE), GQA 14/2, D=64, max_seq=512 ===")
    print(f"{'cur_len':>8} {'splits':>7} {'out.nan':>8} {'out.inf':>8} {'max|Δ| vs SDPA':>16}  verdict")
    ok = True
    for cur_len in [33, 64, 100, 256, 400, 511]:
        torch.manual_seed(cur_len)
        q = torch.randn(B, hq, 1, D, device=DEVICE, dtype=torch.float16)
        ck = torch.randn(B, hkv, MAX_SEQ, D, device=DEVICE, dtype=torch.float16)
        cv = torch.randn(B, hkv, MAX_SEQ, D, device=DEVICE, dtype=torch.float16)
        out = dattn.decode_attn_forward(q, ck, cv, cur_len, scale)  # [B,hq,1,D]
        rep = hq // hkv
        k = ck[:, :, :cur_len].repeat_interleave(rep, dim=1).float()
        v = cv[:, :, :cur_len].repeat_interleave(rep, dim=1).float()
        ref = F.scaled_dot_product_attention(q.float(), k, v, is_causal=False)  # 1 query sees all
        nan = bool(torch.isnan(out).any())
        inf = bool(torch.isinf(out).any())
        diff = (out.float() - ref).abs()
        md = diff[torch.isfinite(diff)].max().item() if torch.isfinite(diff).any() else float("nan")
        good = (not nan) and (not inf) and (md < 5e-2)
        ok = ok and good
        print(f"{cur_len:>8} {_num_splits(cur_len, hq, B):>7} {str(nan):>8} {str(inf):>8} "
              f"{md:>16.5f}  {'OK' if good else 'FAIL'}")
    print(f"  -> kernel isolation: {'PASS (finite, matches SDPA)' if ok else 'FAIL (real kernel bug)'}")
    return ok


def _logit_stats(x: torch.Tensor) -> str:
    xf = x.float()
    n_nan = int(torch.isnan(xf).sum())
    n_inf = int(torch.isinf(xf).sum())
    finite = xf[torch.isfinite(xf)]
    amax = finite.abs().max().item() if finite.numel() else float("nan")
    am = int(xf.nan_to_num(0.0, 0.0, 0.0).argmax())
    return f"nan={n_nan:>5} inf={n_inf:>5} |max_finite|={amax:9.1f} argmax={am}"


def reproduce(prompt: list[int], label: str) -> None:
    cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
    weights, _ = load_weights_on_gpu(cfg, batch=1, device=DEVICE)
    buffers = allocate_buffers(cfg, batch=1, max_seq_len=MAX_SEQ, device=DEVICE)
    ex = Qwen2Executor(cfg, weights, buffers)
    p = torch.tensor([prompt], dtype=torch.int64, device=DEVICE)

    logits = ex.prefill(p)
    tok = int(logits[0, -1].argmax())
    lg_e = ex.decode_step(torch.tensor([tok], device=DEVICE)).clone()   # eager: single-block at len 33

    logits = ex.prefill(p)
    tok = int(logits[0, -1].argmax())
    lg_g = ex.decode_step_graph(torch.tensor([tok], device=DEVICE)).clone()  # graph: split path

    cur_len = len(prompt) + 1
    d = (lg_e.float() - lg_g.float()).abs()
    md_finite = d[torch.isfinite(d)].max().item() if torch.isfinite(d).any() else float("nan")
    print(f"\n=== (B) eager vs graph decode logits — {label} "
          f"(prompt_len={len(prompt)}, cur_len={cur_len}, eager splits={_num_splits(cur_len, 14, 1)}, "
          f"graph splits={_num_splits(MAX_SEQ, 14, 1)}) ===")
    print(f"  eager (single-block) : {_logit_stats(lg_e)}")
    print(f"  graph (split-KV)     : {_logit_stats(lg_g)}")
    print(f"  argmax match: {int(lg_e.float().nan_to_num().argmax()) == int(lg_g.float().nan_to_num().argmax())}"
          f"   max|Δ| over FINITE entries = {md_finite:.5f}")


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. Run via slurm/run_python.sh.")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    isolate()
    torch.manual_seed(0)
    rand_prompt = torch.randint(0, 151936, (32,)).tolist()   # matches draft_graph_decode.py
    reproduce(rand_prompt, "RANDOM prompt (as benchmark)")
    reproduce([151643, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 220],
              "REAL prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
