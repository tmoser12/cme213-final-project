"""Micro-benchmark for the fused causal SDPA sub-op.

Eager:    repeat_kv (modeling_qwen2.py:280) + F.scaled_dot_product_attention,
          mirroring Qwen2SdpaAttention's dispatch path inside modeling_qwen2.
Compiled: torch.compile(eager).
Custom:   custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, scale).

The custom kernel reads K/V from the paged cache up to cur_len (= write_pos+S),
which mirrors how the real inference path will dispatch it.
"""

import argparse
import math
import torch
import torch.nn.functional as F
from pathlib import Path

from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from src.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()

# Qwen2.5-7B
NH, NKV, D = 28, 4, 128
GROUPS = NH // NKV
MAX_SEQ = 32768
SCALE = 1.0 / math.sqrt(D)

CONFIGS = [(1, 1), (1, 128), (1, 256), (1, 512), (1, 2048), (1, 16384)]


def make_inputs(batch_size, seq_len):
    """Build q for S new tokens, plus a cache pre-filled up to cur_len = seq_len.

    For the scaffold we set write_pos=0 so cur_len = seq_len, meaning the
    custom kernel attends to exactly the new tokens (pure-prefill regime).
    The cache is sized to max(MAX_SEQ, seq_len) so configs with seq_len >
    MAX_SEQ still fit.
    """
    cache_seq = max(MAX_SEQ, seq_len)
    q = torch.randn(batch_size, NH, seq_len, D, dtype=torch.float16, device="cuda")
    cache_k = torch.zeros(batch_size, NKV, cache_seq, D, dtype=torch.float16, device="cuda")
    cache_v = torch.zeros(batch_size, NKV, cache_seq, D, dtype=torch.float16, device="cuda")
    # Put random K/V into the live prefix [0:seq_len].
    cache_k[:, :, :seq_len, :] = torch.randn(batch_size, NKV, seq_len, D,
                                             dtype=torch.float16, device="cuda")
    cache_v[:, :, :seq_len, :] = torch.randn(batch_size, NKV, seq_len, D,
                                             dtype=torch.float16, device="cuda")
    return q, cache_k, cache_v


def eager_fused_attn(q, cache_k, cache_v, cur_len):
    k_live = cache_k[:, :, :cur_len, :]
    v_live = cache_v[:, :, :cur_len, :]
    k_rep = repeat_kv(k_live, GROUPS)
    v_rep = repeat_kv(v_live, GROUPS)
    return F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True, scale=SCALE)


def _time(fn, num_runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, compiled_attn):
    q, cache_k, cache_v = make_inputs(batch_size, seq_len)
    cur_len = seq_len

    with torch.no_grad():
        ref = eager_fused_attn(q, cache_k, cache_v, cur_len)
        out = custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2), \
            f"fused_attn mismatch at B={batch_size}, S={seq_len}"

    with torch.no_grad():
        for _ in range(20):
            eager_fused_attn(q, cache_k, cache_v, cur_len)
            compiled_attn(q, cache_k, cache_v, cur_len)
            custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)

    num_runs = 500  # SDPA at S=1024 is the slowest config; keep total time bounded.
    with torch.no_grad():
        eager_time  = _time(lambda: eager_fused_attn(q, cache_k, cache_v, cur_len), num_runs)
        comp_time   = _time(lambda: compiled_attn(q, cache_k, cache_v, cur_len), num_runs)
        custom_time = _time(lambda: custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE), num_runs)
    return eager_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    q, cache_k, cache_v = make_inputs(batch_size, seq_len)
    cur_len = seq_len
    with torch.no_grad():
        for _ in range(5):
            custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
    torch.cuda.synchronize()
    tag = f"fused_attn/B{batch_size}_S{seq_len}"
    with torch.no_grad():
        torch.cuda.nvtx.range_push(tag)
        custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    print("Initializing fused_attn benchmark...")
    compiled_attn = torch.compile(eager_fused_attn)

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len in CONFIGS:
        print(f"Benchmarking B={batch_size:2d}, S={seq_len:4d}...")
        eager_t, comp_t, cust_t = run_benchmark_for_config(batch_size, seq_len, compiled_attn)
        results.append({
            "batch_size": batch_size, "seq_len": seq_len,
            "eager_us": eager_t, "compiled_us": comp_t, "custom_us": cust_t,
            "speedup_vs_eager":    eager_t / cust_t,
            "speedup_vs_compiled": comp_t  / cust_t,
        })

    report = "Fused Attention Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch':<8}| {'Seq':<6}| {'Eager (us)':<12}| {'Compiled (us)':<15}| {'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += (f"{r['batch_size']:<8}| {r['seq_len']:<6}| "
                   f"{r['eager_us']:<12.2f}| {r['compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| "
                   f"{r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n")

    report_path = Path(__file__).resolve().parent / "benchmark_fused_attn_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report: {report_path}\n\n{report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    if p.parse_args().profile: profile_main()
    else: main()
