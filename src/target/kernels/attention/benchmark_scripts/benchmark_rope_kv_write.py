"""Correctness + micro-benchmark for the RoPE-fused KV-write sub-op.

This is the fused replacement for the rope_forward(K) + kv_write_forward pair:
it rotates K (HF rotate_half) as it scatters K/V into the paged cache, so K
never transits global memory rotated. V is copied verbatim.

Oracle (the un-fused baseline it replaces):
  k_rot = apply_rotary_pos_emb(., k, cos, sin)        # transformers, modeling_qwen2
  cache_k[:, :, wp:wp+S, :] = k_rot                   # DynamicCache.update slice
  cache_v[:, :, wp:wp+S, :] = v
Custom:
  custom_ops.rope_kv_write_forward(k, v, cache_k, cache_v, wp, cos, sin)

K uses allclose (rope adds fp16 error); V uses torch.equal (pure copy).
"""

import argparse
import torch
from pathlib import Path

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from src.target.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()

# Qwen2.5-7B
NKV, D = 4, 128
MAX_SEQ = 2048

# (batch, seq_len, write_pos). write_pos > 0 exercises the destination offset.
CONFIGS = [(1, 1, 0), (1, 1, 37), (1, 128, 0), (2, 128, 64), (8, 128, 0),
           (8, 512, 256), (16, 1024, 0)]


def make_inputs(batch_size, seq_len):
    new_k = torch.randn(batch_size, NKV, seq_len, D, dtype=torch.float16, device="cuda")
    new_v = torch.randn(batch_size, NKV, seq_len, D, dtype=torch.float16, device="cuda")
    cache_k = torch.zeros(batch_size, NKV, MAX_SEQ, D, dtype=torch.float16, device="cuda")
    cache_v = torch.zeros(batch_size, NKV, MAX_SEQ, D, dtype=torch.float16, device="cuda")
    # RoPE tables: upper D/2 duplicates lower D/2, indexed by local row s.
    cos_half = torch.randn(batch_size, seq_len, D // 2, dtype=torch.float16, device="cuda")
    sin_half = torch.randn(batch_size, seq_len, D // 2, dtype=torch.float16, device="cuda")
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return new_k, new_v, cache_k, cache_v, cos, sin


def eager_rope_write(new_k, new_v, cache_k, cache_v, write_pos, cos, sin):
    S = new_k.size(2)
    # apply_rotary_pos_emb rotates both args; we only consume the K result.
    _, k_rot = apply_rotary_pos_emb(new_k, new_k, cos, sin)
    cache_k[:, :, write_pos:write_pos + S, :] = k_rot
    cache_v[:, :, write_pos:write_pos + S, :] = new_v


def _time(fn, num_runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, write_pos, compiled_write):
    new_k, new_v, cache_k_ref, cache_v_ref, cos, sin = make_inputs(batch_size, seq_len)
    cache_k_cust = cache_k_ref.clone()
    cache_v_cust = cache_v_ref.clone()

    with torch.no_grad():
        eager_rope_write(new_k, new_v, cache_k_ref, cache_v_ref, write_pos, cos, sin)
        custom_ops.rope_kv_write_forward(new_k, new_v, cache_k_cust, cache_v_cust, write_pos, cos, sin)
    assert torch.allclose(cache_k_ref, cache_k_cust, atol=1e-2, rtol=1e-2), \
        f"rope_kv_write K mismatch B={batch_size}, S={seq_len}, wp={write_pos}"
    assert torch.equal(cache_v_ref, cache_v_cust), \
        f"rope_kv_write V mismatch B={batch_size}, S={seq_len}, wp={write_pos}"

    cache_k_comp = cache_k_ref.clone()
    cache_v_comp = cache_v_ref.clone()
    with torch.no_grad():
        for _ in range(20):
            eager_rope_write(new_k, new_v, cache_k_ref, cache_v_ref, write_pos, cos, sin)
            compiled_write(new_k, new_v, cache_k_comp, cache_v_comp, write_pos, cos, sin)
            custom_ops.rope_kv_write_forward(new_k, new_v, cache_k_cust, cache_v_cust, write_pos, cos, sin)

    num_runs = 2000
    with torch.no_grad():
        eager_time  = _time(lambda: eager_rope_write(new_k, new_v, cache_k_ref, cache_v_ref, write_pos, cos, sin), num_runs)
        comp_time   = _time(lambda: compiled_write(new_k, new_v, cache_k_comp, cache_v_comp, write_pos, cos, sin), num_runs)
        custom_time = _time(lambda: custom_ops.rope_kv_write_forward(new_k, new_v, cache_k_cust, cache_v_cust, write_pos, cos, sin), num_runs)
    return eager_time, comp_time, custom_time


def profile_main(batch_size=8, seq_len=128):
    new_k, new_v, cache_k, cache_v, cos, sin = make_inputs(batch_size, seq_len)
    for _ in range(5):
        custom_ops.rope_kv_write_forward(new_k, new_v, cache_k, cache_v, 0, cos, sin)
    torch.cuda.synchronize()
    tag = f"rope_kv_write/B{batch_size}_S{seq_len}"
    torch.cuda.nvtx.range_push(tag)
    custom_ops.rope_kv_write_forward(new_k, new_v, cache_k, cache_v, 0, cos, sin)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    print("Initializing rope_kv_write benchmark...")
    compiled_write = torch.compile(eager_rope_write)

    results = []
    print("\nStarting benchmarks...")
    for batch_size, seq_len, write_pos in CONFIGS:
        print(f"Benchmarking B={batch_size:2d}, S={seq_len:4d}, wp={write_pos}...")
        eager_t, comp_t, cust_t = run_benchmark_for_config(batch_size, seq_len, write_pos, compiled_write)
        results.append({
            "batch_size": batch_size, "seq_len": seq_len, "write_pos": write_pos,
            "eager_us": eager_t, "compiled_us": comp_t, "custom_us": cust_t,
            "speedup_vs_eager":    eager_t / cust_t,
            "speedup_vs_compiled": comp_t  / cust_t,
        })

    report = "RoPE-fused KV Write Micro-Benchmark Report\n" + "=" * 110 + "\n"
    report += f"{'Batch':<8}| {'Seq':<6}| {'Eager (us)':<12}| {'Compiled (us)':<15}| {'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n"
    report += "-" * 110 + "\n"
    for r in results:
        report += (f"{r['batch_size']:<8}| {r['seq_len']:<6}| "
                   f"{r['eager_us']:<12.2f}| {r['compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| "
                   f"{r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n")

    from src.profiling import report_file

    report_path = report_file(__file__, "rope_kv_write")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ Report: {report_path}\n\n{report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    if p.parse_args().profile: profile_main()
    else: main()
