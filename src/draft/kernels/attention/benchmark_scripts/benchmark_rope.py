"""Micro-benchmark for the in-place RoPE sub-op.

Eager:    apply_rotary_pos_emb from transformers.models.qwen2.modeling_qwen2
          (returns rotated copies of q and k; modeling_qwen2.py:237).
Compiled: torch.compile(apply_rotary_pos_emb).
Custom:   in-place mutation of q and k via custom_ops.rope_forward.

Correctness compares custom in-place result to the eager copies.
"""

import argparse
import torch
from pathlib import Path

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from src.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()

# Qwen2.5-7B
NH, NKV, D = 28, 4, 128

CONFIGS = [(1, 1), (1, 128), (2, 128), (8, 128), (8, 512), (16, 1024)]


def make_inputs(batch_size, seq_len):
    q = torch.randn(batch_size, NH,  seq_len, D, dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, NKV, seq_len, D, dtype=torch.float16, device="cuda")
    cos_half = torch.randn(batch_size, seq_len, D // 2, dtype=torch.float16, device="cuda")
    sin_half = torch.randn(batch_size, seq_len, D // 2, dtype=torch.float16, device="cuda")
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return q, k, cos, sin


def _time(fn, num_runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, rope_compiled):
    q, k, cos, sin = make_inputs(batch_size, seq_len)

    q_ref, k_ref = apply_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
    q_custom, k_custom = q.clone(), k.clone()
    custom_ops.rope_forward(q_custom, k_custom, cos, sin)
    assert torch.allclose(q_ref, q_custom, atol=1e-2, rtol=1e-2), \
        f"rope q mismatch at B={batch_size}, S={seq_len}"
    assert torch.allclose(k_ref, k_custom, atol=1e-2, rtol=1e-2), \
        f"rope k mismatch at B={batch_size}, S={seq_len}"

    # Warmup all three paths.
    for _ in range(20):
        apply_rotary_pos_emb(q, k, cos, sin)
        rope_compiled(q, k, cos, sin)
        q_b, k_b = q.clone(), k.clone()
        custom_ops.rope_forward(q_b, k_b, cos, sin)

    num_runs = 2000
    # Eager / compiled are non-mutating -- can pass q, k directly each iter.
    eager_time = _time(lambda: apply_rotary_pos_emb(q, k, cos, sin), num_runs)
    comp_time  = _time(lambda: rope_compiled(q, k, cos, sin), num_runs)
    # Custom mutates in place -- timer overwrites the same pre-allocated clones.
    q_buf, k_buf = q.clone(), k.clone()
    custom_time = _time(lambda: custom_ops.rope_forward(q_buf, k_buf, cos, sin), num_runs)
    return eager_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    q, k, cos, sin = make_inputs(batch_size, seq_len)
    for _ in range(5):
        q_b, k_b = q.clone(), k.clone()
        custom_ops.rope_forward(q_b, k_b, cos, sin)
    torch.cuda.synchronize()
    q_b, k_b = q.clone(), k.clone()
    tag = f"rope/B{batch_size}_S{seq_len}"
    torch.cuda.nvtx.range_push(tag)
    custom_ops.rope_forward(q_b, k_b, cos, sin)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    print("Initializing rope benchmark...")
    rope_compiled = torch.compile(apply_rotary_pos_emb)

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len in CONFIGS:
        print(f"Benchmarking B={batch_size:2d}, S={seq_len:4d}...")
        eager_t, comp_t, cust_t = run_benchmark_for_config(batch_size, seq_len, rope_compiled)
        results.append({
            "batch_size": batch_size, "seq_len": seq_len,
            "eager_us": eager_t, "compiled_us": comp_t, "custom_us": cust_t,
            "speedup_vs_eager":    eager_t / cust_t,
            "speedup_vs_compiled": comp_t  / cust_t,
        })

    report = "RoPE Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch':<8}| {'Seq':<6}| {'Eager (us)':<12}| {'Compiled (us)':<15}| {'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += (f"{r['batch_size']:<8}| {r['seq_len']:<6}| "
                   f"{r['eager_us']:<12.2f}| {r['compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| "
                   f"{r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n")

    report_path = Path(__file__).resolve().parent / "benchmark_rope_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report: {report_path}\n\n{report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    if p.parse_args().profile: profile_main()
    else: main()
