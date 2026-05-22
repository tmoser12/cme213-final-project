"""Profiling driver for src/kernels/attention/kernel.cu :: flash_attention_kernel.

This file is meant to be launched under Nsight Systems (timeline) and
Nsight Compute (per-kernel metrics) by run_profile_flash_attn.sh. It does
the bare minimum needed to get one clean capture of the flash attention
kernel:

  1. Build q + KV-cache tensors once.
  2. Run NUM_WARMUP launches OUTSIDE the cuda profiler region so neither
     ncu nor nsys captures them.
  3. Call cudaProfilerStart, run NUM_PROFILE launches each wrapped in an
     NVTX range, then cudaProfilerStop.

Pair with:
  - nsys profile --capture-range=cudaProfilerApi
  - ncu --profile-from-start off
so that warmup is excluded from the capture without --launch-skip math.
"""

import math
import torch

from src.kernels.attention.jit import load_attention_ops


# ============================================================
# Workload knobs — edit these to profile a different shape.
# The run_profile_flash_attn.sh script reads them from here
# (via `python -c`) to name the output files.
# ============================================================
BATCH_SIZE = 1
SEQ_LEN    = 2048
# ============================================================

# Qwen2.5-7B attention shape (fixed).
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 28, 4, 128
MAX_SEQ = 2048
SCALE   = 1.0 / math.sqrt(HEAD_DIM)

# Warmup is outside the capture region; profile a small handful of launches
# inside it so nsys shows steady-state timing on the timeline. ncu defaults
# to --launch-count 1 so it only captures the first profiled launch.
NUM_WARMUP  = 50
NUM_PROFILE = 5


def make_inputs(B, S):
    """Q for S new tokens + KV cache pre-filled to cur_len = S (pure prefill)."""
    q = torch.randn(B, NUM_HEADS, S, HEAD_DIM,
                    dtype=torch.float16, device="cuda")
    cache_k = torch.zeros(B, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM,
                          dtype=torch.float16, device="cuda")
    cache_v = torch.zeros(B, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM,
                          dtype=torch.float16, device="cuda")
    cache_k[:, :, :S, :] = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM,
                                       dtype=torch.float16, device="cuda")
    cache_v[:, :, :S, :] = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM,
                                       dtype=torch.float16, device="cuda")
    return q, cache_k, cache_v


def main():
    custom_ops = load_attention_ops()

    print(f"[profile_flash_attn] B={BATCH_SIZE} S={SEQ_LEN} "
          f"NH={NUM_HEADS} NKV={NUM_KV_HEADS} D={HEAD_DIM} "
          f"warmup={NUM_WARMUP} profile={NUM_PROFILE}")

    q, cache_k, cache_v = make_inputs(BATCH_SIZE, SEQ_LEN)
    cur_len = SEQ_LEN

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
    torch.cuda.synchronize()

    # cudaProfilerStart/Stop scopes what nsys (--capture-range=cudaProfilerApi)
    # and ncu (--profile-from-start off) record. Warmup is already done.
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"flash_attn/B{BATCH_SIZE}_S{SEQ_LEN}")
    with torch.no_grad():
        for i in range(NUM_PROFILE):
            torch.cuda.nvtx.range_push(f"iter_{i}")
            custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
            torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("[profile_flash_attn] done.")


if __name__ == "__main__":
    main()
