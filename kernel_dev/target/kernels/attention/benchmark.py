"""End-to-end correctness + timing for CustomQwen2Attention vs HF Qwen2Attention.

This is the umbrella benchmark for the attention extension. While the
sub-op kernels are under development, prefer the per-sub-op files
(benchmark_qkv_proj.py, benchmark_rope.py, ...). This file becomes the
integrated regression once all five launchers in kernel.cu are implemented
and CustomQwen2Attention.forward no longer raises NotImplementedError.
"""

import argparse
import torch
from pathlib import Path
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

from kernel_dev.target.kernels.attention.wrapper import CustomQwen2Attention

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
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        intermediate_size=18944,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    )


def build_inputs(batch_size, seq_len, cfg, rotary_emb):
    x = torch.randn(batch_size, seq_len, cfg.hidden_size, dtype=torch.float16, device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    cos, sin = rotary_emb(x, position_ids)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float16, device="cuda")
    mask = torch.triu(mask, diagonal=1).view(1, 1, seq_len, seq_len)
    return x, (cos, sin), mask


def _time(fn, num_runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, cfg, hf_baseline, hf_compiled, custom_module, rotary_emb):
    x, pos_emb, mask = build_inputs(batch_size, seq_len, cfg, rotary_emb)

    with torch.no_grad():
        out_hf, _, _ = hf_baseline(x, attention_mask=mask, position_embeddings=pos_emb)
        out_custom = custom_module(x, position_embeddings=pos_emb, attention_mask=mask)
        assert torch.allclose(out_hf, out_custom, atol=1e-2, rtol=1e-2), \
            f"Numerical mismatch at B={batch_size}, S={seq_len}!"

    with torch.no_grad():
        for _ in range(20):
            hf_baseline(x, attention_mask=mask, position_embeddings=pos_emb)
            hf_compiled(x, attention_mask=mask, position_embeddings=pos_emb)
            custom_module(x, position_embeddings=pos_emb, attention_mask=mask)

    # Attention is heavier than rmsnorm -- fewer iters keeps total time reasonable
    # at S=1024.
    num_runs = 1000

    hf_time = _time(lambda: hf_baseline(x, attention_mask=mask, position_embeddings=pos_emb), num_runs)
    comp_time = _time(lambda: hf_compiled(x, attention_mask=mask, position_embeddings=pos_emb), num_runs)
    custom_time = _time(lambda: custom_module(x, position_embeddings=pos_emb, attention_mask=mask), num_runs)
    return hf_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    cfg = make_qwen7b_config()
    hf_baseline = Qwen2Attention(cfg, layer_idx=0).cuda().half()
    rotary_emb = Qwen2RotaryEmbedding(config=cfg).cuda()
    custom_module = CustomQwen2Attention(hf_baseline).cuda()
    x, pos_emb, mask = build_inputs(batch_size, seq_len, cfg, rotary_emb)

    with torch.no_grad():
        for _ in range(5):
            custom_module(x, position_embeddings=pos_emb, attention_mask=mask)
    torch.cuda.synchronize()

    # cudaProfilerStart/Stop scopes what nsys (--capture-range=cudaProfilerApi)
    # records so warmup stays out of the capture. Mirrors
    # benchmark_scripts/profile_flash_attn.py.
    tag = f"attention/B{batch_size}_S{seq_len}"
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(tag)
    with torch.no_grad():
        custom_module(x, position_embeddings=pos_emb, attention_mask=mask)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def main():
    cfg = make_qwen7b_config()
    print("Initializing models and triggering JIT compilations...")
    hf_baseline = Qwen2Attention(cfg, layer_idx=0).cuda().half()
    hf_compiled = torch.compile(hf_baseline)
    rotary_emb = Qwen2RotaryEmbedding(config=cfg).cuda()
    custom_module = CustomQwen2Attention(hf_baseline).cuda()

    # Probe: while kernels are scaffolds, CustomQwen2Attention.forward raises
    # NotImplementedError. Print a clear message and exit instead of running
    # the full sweep just to fail at every config.
    try:
        x, pos_emb, mask = build_inputs(1, 1, cfg, rotary_emb)
        with torch.no_grad():
            custom_module(x, position_embeddings=pos_emb, attention_mask=mask)
    except NotImplementedError as e:
        print(f"\n⚠️  CustomQwen2Attention is scaffold-only: {e}")
        print("    Fill in kernel_dev/target/kernels/attention/kernel.cu + wrapper.py first.")
        return

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len in CONFIGS:
        print(f"Benchmarking B={batch_size:2d}, S={seq_len:4d}...")
        hf_time, comp_time, custom_time = run_benchmark_for_config(
            batch_size, seq_len, cfg, hf_baseline, hf_compiled, custom_module, rotary_emb
        )
        results.append({
            "batch_size": batch_size, "seq_len": seq_len,
            "hf_eager_us": hf_time, "hf_compiled_us": comp_time, "custom_us": custom_time,
            "speedup_vs_eager": hf_time / custom_time,
            "speedup_vs_compiled": comp_time / custom_time,
        })

    report = "Attention Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch':<8}| {'Seq':<6}| {'Eager (us)':<12}| {'Compiled (us)':<15}| {'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += f"{r['batch_size']:<8}| {r['seq_len']:<6}| {r['hf_eager_us']:<12.2f}| {r['hf_compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| {r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n"

    from kernel_dev.profiling import report_file

    report_path = report_file(__file__, "attention")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Benchmark complete! Report saved to: {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true",
                        help="Single-launch NVTX-tagged workload for ncu capture.")
    args = parser.parse_args()
    if args.profile:
        profile_main()
    else:
        main()
