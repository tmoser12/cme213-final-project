"""Correctness oracle + micro-benchmark for the residual_ops extension.

Two ops share this dir, so this file checks and times both:

  * residual_add  -- out = a + b. Oracle is torch's fp16 add; __hadd2 is the
    same round-to-nearest fp16 add, so the check is bit-exact (atol=0).
  * lm_head       -- logits = hidden @ weight^T. Oracle is nn.Linear(H, vocab,
    bias=False) (what Qwen2ForCausalLM.lm_head is); cuBLAS fp32-accum vs torch's
    GEMM agree to a small tolerance.

    bash src/target/kernels/residual_ops/run_benchmark.sh             # this file
    bash src/target/kernels/residual_ops/run_benchmark.sh --profile   # + nsys/ncu
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.target.kernels.residual_ops.wrapper import custom_ops, CustomQwenLMHead

# Qwen2.5-7B-Instruct dims.
HIDDEN_SIZE = 3584
VOCAB_SIZE = 152064

# Row counts (M = batch*seq). Decode is M=1; the rest are prefill/verify-ish.
ADD_CONFIGS = [1, 128, 256, 1024, 8192, 16384]
# The lm_head GEMM writes [M, 152064] -- keep M modest so the sweep is tractable.
LMHEAD_CONFIGS = [1, 5, 128, 512, 2048]


def _time(fn, num_runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


# ---------------------------------------------------------------------------
# residual_add
# ---------------------------------------------------------------------------
def bench_residual_add():
    rows = []
    print("\n[residual_add] correctness + timing")
    for M in ADD_CONFIGS:
        a = torch.randn(M, HIDDEN_SIZE, dtype=torch.float16, device="cuda")
        b = torch.randn(M, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

        with torch.no_grad():
            ref = a + b
            custom = custom_ops.residual_add_forward(a, b)
            assert torch.equal(ref, custom), f"residual_add mismatch at M={M}!"

        eager = lambda: a + b
        compiled = torch.compile(lambda x, y: x + y)
        with torch.no_grad():
            for _ in range(20):
                eager(); compiled(a, b); custom_ops.residual_add_forward(a, b)

        n = 500
        t_eager = _time(eager, n)
        t_comp = _time(lambda: compiled(a, b), n)
        t_custom = _time(lambda: custom_ops.residual_add_forward(a, b), n)
        rows.append((M, t_eager, t_comp, t_custom))
        print(f"  M={M:<6} eager={t_eager:7.2f}us  compiled={t_comp:7.2f}us  "
              f"custom={t_custom:7.2f}us  ({t_eager / t_custom:.2f}x eager)")
    return rows


# ---------------------------------------------------------------------------
# lm_head
# ---------------------------------------------------------------------------
def bench_lm_head():
    rows = []
    print("\n[lm_head] correctness + timing")
    hf = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False).cuda().half()
    custom = CustomQwenLMHead(hf).cuda()
    compiled = torch.compile(hf)

    for M in LMHEAD_CONFIGS:
        x = torch.randn(M, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

        with torch.no_grad():
            ref = hf(x)
            out = custom(x)
            assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2), \
                f"lm_head mismatch at M={M}!"

        with torch.no_grad():
            for _ in range(10):
                hf(x); compiled(x); custom(x)

        n = 100
        t_eager = _time(lambda: hf(x), n)
        t_comp = _time(lambda: compiled(x), n)
        t_custom = _time(lambda: custom(x), n)
        rows.append((M, t_eager, t_comp, t_custom))
        print(f"  M={M:<6} eager={t_eager:8.2f}us  compiled={t_comp:8.2f}us  "
              f"custom={t_custom:8.2f}us  ({t_eager / t_custom:.2f}x eager)")
    return rows


def profile_main(M=2048):
    """Single-launch workload for Nsight (nsys/ncu) capture: one add + one GEMM."""
    a = torch.randn(M, HIDDEN_SIZE, dtype=torch.float16, device="cuda")
    b = torch.randn(M, HIDDEN_SIZE, dtype=torch.float16, device="cuda")
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float16, device="cuda")

    with torch.no_grad():
        for _ in range(5):
            custom_ops.residual_add_forward(a, b)
            custom_ops.lm_head_forward(a, weight)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"residual_ops/M{M}")
    with torch.no_grad():
        custom_ops.residual_add_forward(a, b)
        custom_ops.lm_head_forward(a, weight)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("✅ Profile workload done.")


def main():
    print("Initializing + triggering JIT compilation...")
    add_rows = bench_residual_add()
    lm_rows = bench_lm_head()

    report = "Residual-Ops Micro-Benchmark Report\n" + "=" * 78 + "\n\n"
    report += "residual_add  (out = a + b, fp16, bit-exact vs torch)\n"
    report += f"{'M':<8} | {'Eager (us)':<12} | {'Compiled (us)':<14} | {'Custom (us)':<12} | {'Eager x':<8}\n"
    report += "-" * 78 + "\n"
    for M, e, c, cu in add_rows:
        report += f"{M:<8} | {e:<12.2f} | {c:<14.2f} | {cu:<12.2f} | {e / cu:<7.2f}x\n"

    report += "\nlm_head  (logits = hidden @ weight^T, [M,3584]@[152064,3584])\n"
    report += f"{'M':<8} | {'Eager (us)':<12} | {'Compiled (us)':<14} | {'Custom (us)':<12} | {'Eager x':<8}\n"
    report += "-" * 78 + "\n"
    for M, e, c, cu in lm_rows:
        report += f"{M:<8} | {e:<12.2f} | {c:<14.2f} | {cu:<12.2f} | {e / cu:<7.2f}x\n"

    report_path = Path(__file__).resolve().parent / "benchmark_report.txt"
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
