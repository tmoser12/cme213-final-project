"""End-to-end correctness + timing for CustomQwenMLP vs HF Qwen2MLP.

Mirrors src/kernels/rmsnorm/benchmark.py: builds a real HF Qwen2MLP, checks
torch.allclose(hf, custom) inline at each config, then times the custom fused
SwiGLU MLP against HF eager and torch.compile, writing benchmark_report.txt.

    bash src/kernels/swiglu/run_benchmark.sh             # this file
    bash src/kernels/swiglu/run_benchmark.sh --profile   # + nsys/ncu capture
"""

import argparse
import torch
from pathlib import Path
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from src.target.kernels.swiglu.wrapper import CustomQwenMLP

# Qwen2.5-7B-Instruct MLP dims.
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 18944

# Shape sweep used by both timing benchmarks and the ncu profile path.
CONFIGS = [
    (1, 1),      # Auto-regressive decoding phase
    (1, 128),    # Short prompt
    (2, 128),    # Batched short prompt
    (8, 128),
    (8, 512),    # Medium prompt
    (16, 1024),  # Long batched prompt
    (1, 1024),
]


def make_qwen7b_config():
    return Qwen2Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="silu",
    )


def _time(fn, num_runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, hf_baseline, hf_compiled, custom_module):
    x = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

    # Correctness check (fused op -> looser tolerance than a pure gather).
    with torch.no_grad():
        out_hf = hf_baseline(x)
        out_custom = custom_module(x)
        assert torch.allclose(out_hf, out_custom, atol=1e-2, rtol=1e-2), \
            f"Numerical mismatch at B={batch_size}, S={seq_len}!"

    with torch.no_grad():
        for _ in range(20):
            hf_baseline(x)
            hf_compiled(x)
            custom_module(x)

    # The MLP is compute-heavy (three big GEMMs); fewer iters than the
    # memory-bound rmsnorm kernel keeps the sweep tractable at S=1024.
    num_runs = 200
    hf_time = _time(lambda: hf_baseline(x), num_runs)
    comp_time = _time(lambda: hf_compiled(x), num_runs)
    custom_time = _time(lambda: custom_module(x), num_runs)
    return hf_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    """Minimal single-launch workload for Nsight (nsys/ncu) capture."""
    cfg = make_qwen7b_config()
    hf_baseline = Qwen2MLP(cfg).cuda().half()
    custom_module = CustomQwenMLP(hf_baseline).cuda()
    x = torch.randn(batch_size, seq_len, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        for _ in range(5):
            custom_module(x)
    torch.cuda.synchronize()

    tag = f"swiglu/B{batch_size}_S{seq_len}"
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(tag)
    with torch.no_grad():
        custom_module(x)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("✅ Profile workload done.")


def main():
    cfg = make_qwen7b_config()
    print("Initializing models and triggering JIT compilations...")
    hf_baseline = Qwen2MLP(cfg).cuda().half()
    hf_compiled = torch.compile(hf_baseline)
    custom_module = CustomQwenMLP(hf_baseline).cuda()

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len in CONFIGS:
        print(f"Benchmarking Batch Size = {batch_size:2d}, Sequence Length = {seq_len:4d}...")
        hf_time, comp_time, custom_time = run_benchmark_for_config(
            batch_size, seq_len, hf_baseline, hf_compiled, custom_module
        )
        results.append({
            "batch_size": batch_size, "seq_len": seq_len,
            "hf_eager_us": hf_time, "hf_compiled_us": comp_time, "custom_us": custom_time,
            "speedup_vs_eager": hf_time / custom_time,
            "speedup_vs_compiled": comp_time / custom_time,
        })

    report = "SwiGLU Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch Size':<12} | {'Seq Len':<10} | {'Eager (us)':<12} | {'Compiled (us)':<15} | {'Custom (us)':<12} | {'Eager Speedup':<15} | {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += f"{r['batch_size']:<12} | {r['seq_len']:<10} | {r['hf_eager_us']:<12.2f} | {r['hf_compiled_us']:<15.2f} | {r['custom_us']:<12.2f} | {r['speedup_vs_eager']:<14.2f}x | {r['speedup_vs_compiled']:<14.2f}x\n"

    from src.profiling import report_file

    report_path = report_file(__file__, "swiglu")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Benchmark complete! Report saved to: {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true",
                        help="Single-launch NVTX-tagged workload for ncu/nsys capture.")
    args = parser.parse_args()
    if args.profile:
        profile_main()
    else:
        main()
