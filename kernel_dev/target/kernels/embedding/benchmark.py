import torch
import torch.nn as nn
from pathlib import Path
from kernel_dev.target.kernels.embedding.wrapper import CustomEmbedding

def run_benchmark_for_config(batch_size, seq_len, vocab_size, hidden_size, hf_baseline, hf_compiled, custom_module):
    # Fake token indices -- the only requirement is 0 <= id < vocab_size
    input_ids = torch.randint(
        low=0, high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.int64, device="cuda"
    )

    # Correctness Check (pure gather -> bit-exact)
    with torch.no_grad():
        out_hf = hf_baseline(input_ids)
        out_custom = custom_module(input_ids)
        assert torch.equal(out_hf, out_custom), f"Numerical mismatch at B={batch_size}, S={seq_len}!"

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            hf_baseline(input_ids)
            hf_compiled(input_ids)
            custom_module(input_ids)

    num_runs = 5000

    # Eager
    start_hf = torch.cuda.Event(enable_timing=True)
    end_hf = torch.cuda.Event(enable_timing=True)
    start_hf.record()
    for _ in range(num_runs):
        hf_baseline(input_ids)
    end_hf.record()
    torch.cuda.synchronize()
    hf_time = start_hf.elapsed_time(end_hf) / num_runs * 1000  # Convert ms to us

    # Compiled
    start_comp = torch.cuda.Event(enable_timing=True)
    end_comp = torch.cuda.Event(enable_timing=True)
    start_comp.record()
    for _ in range(num_runs):
        hf_compiled(input_ids)
    end_comp.record()
    torch.cuda.synchronize()
    comp_time = start_comp.elapsed_time(end_comp) / num_runs * 1000

    # Custom
    start_custom = torch.cuda.Event(enable_timing=True)
    end_custom = torch.cuda.Event(enable_timing=True)
    start_custom.record()
    for _ in range(num_runs):
        custom_module(input_ids)
    end_custom.record()
    torch.cuda.synchronize()
    custom_time = start_custom.elapsed_time(end_custom) / num_runs * 1000

    return hf_time, comp_time, custom_time

def main():
    print("Initializing models and triggering JIT compilations...")
    # Qwen2.5-7B-Instruct config: vocab_size=152064, hidden_size=3584
    vocab_size = 152064
    hidden_size = 3584

    hf_baseline = nn.Embedding(vocab_size, hidden_size).cuda().half()
    hf_compiled = torch.compile(hf_baseline)
    custom_module = CustomEmbedding(hf_baseline).cuda()

    configs = [
        (1, 1),      # Auto-regressive decoding phase
        (1, 128),    # Short prompt
        (2, 128),    # Batched short prompt
        (8, 128),
        (8, 512),    # Medium prompt
        (16, 1024),  # Long batched prompt
        (1, 1024),
    ]

    results = []

    print("\nStarting benchmarks...")
    for batch_size, seq_len in configs:
        print(f"Benchmarking Batch Size = {batch_size:2d}, Sequence Length = {seq_len:4d}...")
        hf_time, comp_time, custom_time = run_benchmark_for_config(
            batch_size, seq_len, vocab_size, hidden_size, hf_baseline, hf_compiled, custom_module
        )

        results.append({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hf_eager_us": hf_time,
            "hf_compiled_us": comp_time,
            "custom_us": custom_time,
            "speedup_vs_eager": hf_time / custom_time,
            "speedup_vs_compiled": comp_time / custom_time
        })

    # Generate Text Report
    report = "Embedding Micro-Benchmark Report\n"
    report += "=" * 110 + "\n"
    report += f"{'Batch Size':<12} | {'Seq Len':<10} | {'Eager (us)':<12} | {'Compiled (us)':<15} | {'Custom (us)':<12} | {'Eager Speedup':<15} | {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += f"{r['batch_size']:<12} | {r['seq_len']:<10} | {r['hf_eager_us']:<12.2f} | {r['hf_compiled_us']:<15.2f} | {r['custom_us']:<12.2f} | {r['speedup_vs_eager']:<14.2f}x | {r['speedup_vs_compiled']:<14.2f}x\n"

    from kernel_dev.profiling import report_file

    report_path = report_file(__file__, "embedding")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n✅ Benchmark complete! Report saved to: {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    main()
