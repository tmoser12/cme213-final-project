"""Correctness + micro-benchmark for the decode-phase attention sub-ops.

Covers both decode entry points against a **populated** KV cache (history > 0,
cur_len > S) — the regime the prefill benchmark (benchmark_fused_attn.py) never
exercises:
  - S == 1            -> custom_ops.decode_attn_forward   (true single-token decode)
  - S in [2, 8]       -> custom_ops.small_q_attn_forward  (spec-decode verify of gamma+1)

Every config is checked against TWO independent oracles before timing:
  1. HF: repeat_kv + F.scaled_dot_product_attention with an EXPLICIT causal
     mask. We do not use is_causal=True here: that flag is only well-defined
     when q_len == k_len, and for q_len = S < k_len = cur_len PyTorch's math
     fallback top-left-aligns the triangle (query 0 -> key 0 only), which is NOT
     decode semantics. The explicit mask encodes key_pos <= query_abs_pos
     (query row i is at absolute position cur_len - S + i), matching the kernel.
  2. The existing prefill kernel (fused_attn_forward), which already handles
     cur_len > S. An independent custom path catches mask/index bugs the HF
     comparison might share.

The cache is allocated with PAD extra trailing rows filled with random values
that are NOT part of the live prefix [0:cur_len]. If the kernel's tail masking
(kpos >= cur_len) is wrong, those rows leak into the result and the oracle
checks fail — so the padding is a deliberate masking test, not just slack.
"""

import argparse
import math
import torch
import torch.nn.functional as F
from pathlib import Path

from transformers.models.qwen2.modeling_qwen2 import repeat_kv, apply_rotary_pos_emb
from kernel_dev.target.kernels.attention.jit import load_attention_ops

custom_ops = load_attention_ops()

# Qwen2.5-7B attention shape.
NH, NKV, D = 28, 4, 128
GROUPS = NH // NKV
SCALE = 1.0 / math.sqrt(D)

# Unwritten trailing cache rows (filled with random garbage) used to verify the
# kpos >= cur_len tail masking.
PAD = 16

# (batch, S, history). S=1 hits decode_attn_forward; S>1 hits small_q_attn_forward.
# Non-tile-aligned histories (127, 16383) cross KV_TILE boundaries and stress the
# online-softmax combine + tail masking. The B=2 rows exercise the batch axis.
CONFIGS = [
    (1, 1, 0), (1, 1, 127), (1, 1, 1024), (1, 1, 4096), (1, 1, 16383),
    (1, 2, 1024), (1, 4, 127), (1, 5, 4096), (1, 8, 16383), (1, 8, 0),
    (2, 1, 1024), (2, 5, 1024),
]


def make_inputs(batch_size, seq_len, history):
    """Random q for S new tokens + a cache pre-filled to cur_len = history + S.

    The cache is sized cur_len + PAD; the [cur_len:] tail is random and must be
    masked out by the kernel. Returns (q, cache_k, cache_v, cur_len).
    """
    cur_len = history + seq_len
    cache_seq = cur_len + PAD
    q = torch.randn(batch_size, NH, seq_len, D, dtype=torch.float16, device="cuda")
    cache_k = torch.randn(batch_size, NKV, cache_seq, D, dtype=torch.float16, device="cuda")
    cache_v = torch.randn(batch_size, NKV, cache_seq, D, dtype=torch.float16, device="cuda")
    return q, cache_k, cache_v, cur_len


def build_causal_mask(seq_len, cur_len, device):
    """Boolean attend-mask [S, cur_len]; True where query row i (abs pos
    cur_len - S + i) may attend to key j, i.e. j <= cur_len - S + i. Broadcasts
    over [B, NH, S, cur_len]."""
    q_pos = torch.arange(cur_len - seq_len, cur_len, device=device).view(seq_len, 1)
    k_pos = torch.arange(cur_len, device=device).view(1, cur_len)
    return k_pos <= q_pos


def eager_attn(q, cache_k, cache_v, cur_len, attn_mask):
    """HF oracle: attend over the live cache prefix [0:cur_len] under attn_mask."""
    k = repeat_kv(cache_k[:, :, :cur_len, :], GROUPS)
    v = repeat_kv(cache_v[:, :, :cur_len, :], GROUPS)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=SCALE)


def custom_attn(q, cache_k, cache_v, cur_len, cos=None, sin=None):
    """Dispatch on S: S==1 -> decode, S>1 -> small-q verify. cos/sin (optional)
    enable the fused rope-on-Q path; None leaves Q un-rotated."""
    if q.shape[2] == 1:
        return custom_ops.decode_attn_forward(q, cache_k, cache_v, cur_len, SCALE, cos, sin)
    return custom_ops.small_q_attn_forward(q, cache_k, cache_v, cur_len, SCALE, cos, sin)


def _time(fn, num_runs):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_runs * 1000  # us


def run_benchmark_for_config(batch_size, seq_len, history, compiled_attn):
    q, cache_k, cache_v, cur_len = make_inputs(batch_size, seq_len, history)
    mask = build_causal_mask(seq_len, cur_len, q.device)

    with torch.no_grad():
        ref     = eager_attn(q, cache_k, cache_v, cur_len, mask)
        prefill = custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
        out     = custom_attn(q, cache_k, cache_v, cur_len)
        assert torch.allclose(ref, out, atol=1e-2, rtol=1e-2), \
            f"decode vs HF mismatch at B={batch_size}, S={seq_len}, hist={history}"
        assert torch.allclose(prefill, out, atol=1e-2, rtol=1e-2), \
            f"decode vs prefill-kernel mismatch at B={batch_size}, S={seq_len}, hist={history}"

        # Fused rope-on-Q path, cross-checked against BOTH oracles (HF explicit
        # mask + the prefill kernel), mirroring the no-rope checks above. cos/sin
        # are [B, S, D] (upper D/2 duplicates lower D/2), one row per query token;
        # the same cos is applied to the same local rows in oracle and kernel.
        cos_half = torch.randn(batch_size, seq_len, D // 2, dtype=torch.float16, device=q.device)
        sin_half = torch.randn(batch_size, seq_len, D // 2, dtype=torch.float16, device=q.device)
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        q_rot, _    = apply_rotary_pos_emb(q, q, cos, sin)
        ref_rope     = eager_attn(q_rot, cache_k, cache_v, cur_len, mask)
        prefill_rope = custom_ops.fused_attn_forward(q, cache_k, cache_v, cur_len, SCALE, cos, sin)
        out_rope     = custom_attn(q, cache_k, cache_v, cur_len, cos, sin)
        assert torch.allclose(ref_rope, out_rope, atol=1e-2, rtol=1e-2), \
            f"decode rope-Q vs HF mismatch at B={batch_size}, S={seq_len}, hist={history}"
        assert torch.allclose(prefill_rope, out_rope, atol=1e-2, rtol=1e-2), \
            f"decode rope-Q vs prefill-kernel mismatch at B={batch_size}, S={seq_len}, hist={history}"

    with torch.no_grad():
        for _ in range(20):
            eager_attn(q, cache_k, cache_v, cur_len, mask)
            compiled_attn(q, cache_k, cache_v, cur_len, mask)
            custom_attn(q, cache_k, cache_v, cur_len)

    num_runs = 500
    with torch.no_grad():
        eager_time  = _time(lambda: eager_attn(q, cache_k, cache_v, cur_len, mask), num_runs)
        comp_time   = _time(lambda: compiled_attn(q, cache_k, cache_v, cur_len, mask), num_runs)
        custom_time = _time(lambda: custom_attn(q, cache_k, cache_v, cur_len), num_runs)
    return eager_time, comp_time, custom_time


def profile_main(batch_size=1, seq_len=1, history=4096):
    q, cache_k, cache_v, cur_len = make_inputs(batch_size, seq_len, history)
    with torch.no_grad():
        for _ in range(5):
            custom_attn(q, cache_k, cache_v, cur_len)
    torch.cuda.synchronize()
    with torch.no_grad():
        torch.cuda.nvtx.range_push(f"decode_attn/B{batch_size}_S{seq_len}_H{history}")
        custom_attn(q, cache_k, cache_v, cur_len)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def main():
    print("Initializing decode_attn benchmark...")
    compiled_attn = torch.compile(eager_attn)

    results = []
    print("\nStarting benchmarks (each config is correctness-checked first)...")
    for batch_size, seq_len, history in CONFIGS:
        print(f"Benchmarking B={batch_size}, S={seq_len}, hist={history:5d}...")
        eager_t, comp_t, cust_t = run_benchmark_for_config(batch_size, seq_len, history, compiled_attn)
        results.append({
            "batch_size": batch_size, "seq_len": seq_len, "history": history,
            "eager_us": eager_t, "compiled_us": comp_t, "custom_us": cust_t,
            "speedup_vs_eager":    eager_t / cust_t,
            "speedup_vs_compiled": comp_t  / cust_t,
        })

    report = "Decode Attention Micro-Benchmark Report (populated KV cache)\n" + "=" * 118 + "\n"
    report += (f"{'Batch':<6}| {'S':<4}| {'History':<8}| {'Eager (us)':<12}| {'Compiled (us)':<15}| "
               f"{'Custom (us)':<12}| {'Eager Speedup':<15}| {'Compiled Speedup':<15}\n")
    report += "-" * 118 + "\n"
    for r in results:
        report += (f"{r['batch_size']:<6}| {r['seq_len']:<4}| {r['history']:<8}| "
                   f"{r['eager_us']:<12.2f}| {r['compiled_us']:<15.2f}| {r['custom_us']:<12.2f}| "
                   f"{r['speedup_vs_eager']:<14.2f}x| {r['speedup_vs_compiled']:<14.2f}x\n")

    from kernel_dev.profiling import report_file

    report_path = report_file(__file__, "decode_attn")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✅ All configs passed correctness. Report: {report_path}\n\n{report}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    if p.parse_args().profile: profile_main()
    else: main()
