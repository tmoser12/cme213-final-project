"""Micro-benchmark for the output projection sub-op.

Eager:    a single nn.Linear(HQ, H, bias=False) -- mirrors Qwen2Attention.o_proj
          (modeling_qwen2.py:327, 391).
Compiled: torch.compile(eager).
Custom:   custom_ops.o_proj_forward(x, W_o).
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()

# Qwen2.5-7B
H, NH, D = 3584, 28, 128
HQ = NH * D  # == H for Qwen2 (square o_proj)

CONFIGS = [(1, 1), (1, 128), (2, 128), (8, 128), (8, 512), (16, 1024)]


class EagerOProj(nn.Module):
    """Single nn.Linear, bias=False -- mirrors Qwen2Attention.o_proj exactly."""

    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(HQ, H, bias=False)
        # Override Kaiming-uniform init with N(0,1) so output magnitudes are
        # large enough that the atol=0.5 correctness check sits well below the
        # fp16-GEMM noise floor (see notes in benchmark_qkv_proj.py).
        with torch.no_grad():
            self.o_proj.weight.normal_()

    def forward(self, x):
        return self.o_proj(x)


def make_input(batch_size, seq_len):
    return torch.randn(batch_size * seq_len, HQ, dtype=torch.float16, device="cuda")


def _time(fn, num_runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, eager, compiled, W_o):
    x = make_input(batch_size, seq_len)

    with torch.no_grad():
        ref = eager(x)
        out = custom_ops.o_proj_forward(x, W_o)
        assert torch.allclose(ref, out, atol=0.5, rtol=0), \
            f"o_proj mismatch at B={batch_size}, S={seq_len}"

    with torch.no_grad():
        for _ in range(20):
            eager(x)
            compiled(x)
            custom_ops.o_proj_forward(x, W_o)

    num_runs = 2000
    with torch.no_grad():
        eager_time  = _time(lambda: eager(x), num_runs)
        comp_time   = _time(lambda: compiled(x), num_runs)
        custom_time = _time(lambda: custom_ops.o_proj_forward(x, W_o), num_runs)
    return eager_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    eager = EagerOProj().cuda().half()
    W_o = eager.o_proj.weight.detach()
    x = make_input(batch_size, seq_len)
    with torch.no_grad():
        for _ in range(5):
            custom_ops.o_proj_forward(x, W_o)
    torch.cuda.synchronize()
    tag = f"o_proj/B{batch_size}_S{seq_len}"
    with torch.no_grad():
        torch.cuda.nvtx.range_push(tag)
        custom_ops.o_proj_forward(x, W_o)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    print("Initializing o_proj benchmark...")
    eager = EagerOProj().cuda().half()
    compiled = torch.compile(eager)
    W_o = eager.o_proj.weight.detach()  # weight is shared by reference with the eager call

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len in CONFIGS:
        print(f"Benchmarking B={batch_size:2d}, S={seq_len:4d}...")
        eager_t, comp_t, cust_t = run_benchmark_for_config(batch_size, seq_len, eager, compiled, W_o)
        results.append({
            "batch_size": batch_size, "seq_len": seq_len,
            "eager_us": eager_t, "compiled_us": comp_t, "custom_us": cust_t,
            "speedup_vs_eager":    eager_t / cust_t,
            "speedup_vs_compiled": comp_t  / cust_t,
        })

    report = "O Projection Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch':<8}| {'Seq':<6}| {'Eager (us)':<12}| {'Compiled (us)':<15}| {'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += (f"{r['batch_size']:<8}| {r['seq_len']:<6}| "
                   f"{r['eager_us']:<12.2f}| {r['compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| "
                   f"{r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n")

    report_path = Path(__file__).resolve().parent / "benchmark_o_proj_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report: {report_path}\n\n{report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    if p.parse_args().profile: profile_main()
    else: main()
