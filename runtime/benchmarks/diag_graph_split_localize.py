#!/usr/bin/env python3
"""Localize the split-KV-under-graph NaN to the split path's in-capture scratch.

Capture a CUDA graph around a SINGLE decode_attn_forward_dev call and check the
output for NaN, vs the eager call. Control = single-block (max_seq=128 -> 1 split,
the validated path, allocates only `o`). Suspect = split (max_seq=512 -> 2 splits,
also allocates partial_o/m/l scratch that is freed inside the capture).

Run: bash slurm/run_python.sh runtime/benchmarks/diag_graph_split_localize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.production_kernels.draft.attention import ops as A

dev = "cuda"
B, hq, hkv, D = 1, 14, 2, 64
scale = 1.0 / (D ** 0.5)


def _num_splits(max_seq: int) -> int:
    heads = hq * B
    desired = (144 + heads - 1) // heads
    return max(1, min(desired, (max_seq + 255) // 256, 32))


def run(max_seq: int, cur_len: int, label: str) -> None:
    torch.manual_seed(0)
    q = torch.randn(B, hq, 1, D, device=dev, dtype=torch.float16)
    ck = torch.randn(B, hkv, max_seq, D, device=dev, dtype=torch.float16)
    cv = torch.randn(B, hkv, max_seq, D, device=dev, dtype=torch.float16)
    cl = torch.tensor(cur_len, dtype=torch.int64, device=dev)

    o_eager = A.decode_attn_forward_dev(q, ck, cv, cl, scale).clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            A.decode_attn_forward_dev(q, ck, cv, cl, scale)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_graph = A.decode_attn_forward_dev(q, ck, cv, cl, scale)
    g.replay()
    torch.cuda.synchronize()

    en = bool(torch.isnan(o_eager).any())
    gn = bool(torch.isnan(o_graph).any())
    md = (o_eager.float() - o_graph.float()).abs()
    md = md[torch.isfinite(md)].max().item() if torch.isfinite(md).any() else float("nan")
    print(f"  {label:42s} splits={_num_splits(max_seq)}  "
          f"eager.nan={en!s:5}  graph.nan={gn!s:5}  max|Δ|(finite)={md:.5f}")


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print("graph captured around ONE decode_attn_forward_dev call:")
    run(128, 33, "control: single-block (max_seq=128)")
    run(512, 33, "suspect: split-KV   (max_seq=512)")
    run(512, 400, "suspect: split-KV   (max_seq=512, len=400)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
