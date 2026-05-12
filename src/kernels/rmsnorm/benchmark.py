import torch
from pathlib import Path
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from src.kernels.rmsnorm.wrapper import CustomQwenRMSNorm

def run_benchmark_for_config(batch_size, seq_len, hidden_size, hf_baseline, hf_compiled, custom_module):
    # Dummy input
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
    
    # Correctness Check
    with torch.no_grad():
        out_hf = hf_baseline(x)
        out_custom = custom_module(x)
        assert torch.allclose(out_hf, out_custom, atol=1e-3, rtol=1e-3), f"Numerical mismatch at B={batch_size}, S={seq_len}!"
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            hf_baseline(x)
            hf_compiled(x)
            custom_module(x)
            
    num_runs = 5000
    
    # Eager
    start_hf = torch.cuda.Event(enable_timing=True)
    end_hf = torch.cuda.Event(enable_timing=True)
    start_hf.record()
    for _ in range(num_runs):
        hf_baseline(x)
    end_hf.record()
    torch.cuda.synchronize()
    hf_time = start_hf.elapsed_time(end_hf) / num_runs * 1000  # Convert ms to us
    
    # Compiled
    start_comp = torch.cuda.Event(enable_timing=True)
    end_comp = torch.cuda.Event(enable_timing=True)
    start_comp.record()
    for _ in range(num_runs):
        hf_compiled(x)
    end_comp.record()
    torch.cuda.synchronize()
    comp_time = start_comp.elapsed_time(end_comp) / num_runs * 1000
    
    # Custom
    start_custom = torch.cuda.Event(enable_timing=True)
    end_custom = torch.cuda.Event(enable_timing=True)
    start_custom.record()
    for _ in range(num_runs):
        custom_module(x)
    end_custom.record()
    torch.cuda.synchronize()
    custom_time = start_custom.elapsed_time(end_custom) / num_runs * 1000
    
    return hf_time, comp_time, custom_time

def main():
    print("Initializing models and triggering JIT compilations...")
    hidden_size = 3584
    
    hf_baseline = Qwen2RMSNorm(hidden_size=hidden_size, eps=1e-6).cuda().half()
    hf_compiled = torch.compile(hf_baseline)
    custom_module = CustomQwenRMSNorm(hf_baseline).cuda()
    
    configs = [
        (1, 1),      # Auto-regressive decoding phase
        (1, 128),    # Short prompt
        (2, 128),    # Batched short prompt
        (8, 128),
        (8, 512),    # Medium prompt
        (16, 1024),  # Long batched prompt
    ]
    
    results = []
    
    print("\nStarting benchmarks...")
    for batch_size, seq_len in configs:
        print(f"Benchmarking Batch Size = {batch_size:2d}, Sequence Length = {seq_len:4d}...")
        hf_time, comp_time, custom_time = run_benchmark_for_config(
            batch_size, seq_len, hidden_size, hf_baseline, hf_compiled, custom_module
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
    report = "RMSNorm Micro-Benchmark Report\n"
    report += "=" * 110 + "\n"
    report += f"{'Batch Size':<12} | {'Seq Len':<10} | {'Eager (us)':<12} | {'Compiled (us)':<15} | {'Custom (us)':<12} | {'Eager Speedup':<15} | {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += f"{r['batch_size']:<12} | {r['seq_len']:<10} | {r['hf_eager_us']:<12.2f} | {r['hf_compiled_us']:<15.2f} | {r['custom_us']:<12.2f} | {r['speedup_vs_eager']:<14.2f}x | {r['speedup_vs_compiled']:<14.2f}x\n"
        
    report_path = Path(__file__).resolve().parent / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"\n✅ Benchmark complete! Report saved to: {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    main()
