"""Micro-benchmark for the KV-cache write sub-op.

Eager:    cache[:, :, write_pos:write_pos+S, :] = new (pure tensor slice assign;
          this is what HF's DynamicCache.update does internally for the
          fixed-size path -- see modeling_qwen2.py:365 for the call site).
Compiled: torch.compile(eager).
Custom:   custom_ops.kv_write_forward(new_k, new_v, cache_k, cache_v, write_pos).

Correctness uses torch.equal (this is a pure copy; bit-exact expected).
"""

import argparse
import torch
from pathlib import Path

from src.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()

# Qwen2.5-7B
NKV, D = 4, 128
MAX_SEQ = 2048

CONFIGS = [(1, 1), (1, 128), (2, 128), (8, 128), (8, 512), (16, 1024)]


def make_inputs(batch_size, seq_len, write_pos=0):
    new_k = torch.randn(batch_size, NKV, seq_len, D, dtype=torch.float16, device="cuda")
    new_v = torch.randn(batch_size, NKV, seq_len, D, dtype=torch.float16, device="cuda")
    cache_k = torch.zeros(batch_size, NKV, MAX_SEQ, D, dtype=torch.float16, device="cuda")
    cache_v = torch.zeros(batch_size, NKV, MAX_SEQ, D, dtype=torch.float16, device="cuda")
    return new_k, new_v, cache_k, cache_v, write_pos


def eager_write(new_k, new_v, cache_k, cache_v, write_pos):
    S = new_k.size(2)
    cache_k[:, :, write_pos:write_pos + S, :] = new_k
    cache_v[:, :, write_pos:write_pos + S, :] = new_v


def _time(fn, num_runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, compiled_write):
    new_k, new_v, cache_k_ref, cache_v_ref, wp = make_inputs(batch_size, seq_len)
    cache_k_cust = cache_k_ref.clone()
    cache_v_cust = cache_v_ref.clone()

    eager_write(new_k, new_v, cache_k_ref, cache_v_ref, wp)
    custom_ops.kv_write_forward(new_k, new_v, cache_k_cust, cache_v_cust, wp)
    assert torch.equal(cache_k_ref, cache_k_cust), f"kv_write K mismatch B={batch_size}, S={seq_len}"
    assert torch.equal(cache_v_ref, cache_v_cust), f"kv_write V mismatch B={batch_size}, S={seq_len}"

    # Warmup all three. Each pass writes into the same caches (idempotent for
    # this op).
    cache_k_comp = cache_k_ref.clone()
    cache_v_comp = cache_v_ref.clone()
    for _ in range(20):
        eager_write(new_k, new_v, cache_k_ref, cache_v_ref, wp)
        compiled_write(new_k, new_v, cache_k_comp, cache_v_comp, wp)
        custom_ops.kv_write_forward(new_k, new_v, cache_k_cust, cache_v_cust, wp)

    num_runs = 2000
    eager_time  = _time(lambda: eager_write(new_k, new_v, cache_k_ref, cache_v_ref, wp), num_runs)
    comp_time   = _time(lambda: compiled_write(new_k, new_v, cache_k_comp, cache_v_comp, wp), num_runs)
    custom_time = _time(lambda: custom_ops.kv_write_forward(new_k, new_v, cache_k_cust, cache_v_cust, wp), num_runs)
    return eager_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    new_k, new_v, cache_k, cache_v, wp = make_inputs(batch_size, seq_len)
    for _ in range(5):
        custom_ops.kv_write_forward(new_k, new_v, cache_k, cache_v, wp)
    torch.cuda.synchronize()
    tag = f"kv_write/B{batch_size}_S{seq_len}"
    torch.cuda.nvtx.range_push(tag)
    custom_ops.kv_write_forward(new_k, new_v, cache_k, cache_v, wp)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    print("Initializing kv_write benchmark...")
    compiled_write = torch.compile(eager_write)

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len in CONFIGS:
        print(f"Benchmarking B={batch_size:2d}, S={seq_len:4d}...")
        eager_t, comp_t, cust_t = run_benchmark_for_config(batch_size, seq_len, compiled_write)
        results.append({
            "batch_size": batch_size, "seq_len": seq_len,
            "eager_us": eager_t, "compiled_us": comp_t, "custom_us": cust_t,
            "speedup_vs_eager":    eager_t / cust_t,
            "speedup_vs_compiled": comp_t  / cust_t,
        })

    report = "KV Write Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch':<8}| {'Seq':<6}| {'Eager (us)':<12}| {'Compiled (us)':<15}| {'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += (f"{r['batch_size']:<8}| {r['seq_len']:<6}| "
                   f"{r['eager_us']:<12.2f}| {r['compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| "
                   f"{r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n")

    report_path = Path(__file__).resolve().parent / "benchmark_kv_write_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report: {report_path}\n\n{report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    if p.parse_args().profile: profile_main()
    else: main()
