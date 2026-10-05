"""Regression coverage for padded scalar attention and tiled RoPE/cache writes."""
import importlib
import math
import unittest

import torch
import torch.nn.functional as F


def inputs(batch, heads, seq, dim):
    return torch.randn(batch, heads, seq, dim, device='cuda', dtype=torch.float16) * .3


def tables(batch, seq, dim):
    angles = torch.randn(batch, seq, dim // 2, device='cuda')
    return angles.cos().repeat(1, 1, 2).half(), angles.sin().repeat(1, 1, 2).half()


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1).float()
    return (x.float() * cos[:, None].float() + rotated * sin[:, None].float()).half()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TestAttentionLayouts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(215)

    def test_rope_tiles_offsets_and_sentinels(self):
        for role, heads, dim in [('target', 4, 128), ('draft', 2, 64)]:
            ops = importlib.import_module(f'runtime.production_kernels.{role}.attention.ops')
            for batch in [1, 2]:
                for seq in [0, 1, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1742]:
                    for position in [3, 2048 - seq]:
                        with self.subTest(role=role, batch=batch, seq=seq, position=position):
                            k, v = inputs(batch, heads, seq, dim), inputs(batch, heads, seq, dim)
                            cos, sin = tables(batch, seq, dim)
                            ref_k = torch.full((batch, heads, 2048, dim), 7., device='cuda', dtype=torch.float16)
                            ref_v = torch.full_like(ref_k, -7.)
                            ref_k[:, :, position:position + seq] = rotate(k, cos, sin)
                            ref_v[:, :, position:position + seq] = v
                            for device_scalar in [False, True]:
                                ck, cv = torch.full_like(ref_k, 7.), torch.full_like(ref_v, -7.)
                                fn = ops.rope_kv_write_forward_dev if device_scalar else ops.rope_kv_write_forward
                                pos = torch.tensor(position, device='cuda') if device_scalar else position
                                fn(k, v, ck, cv, pos, cos, sin)
                                torch.testing.assert_close(ck, ref_k, atol=.002, rtol=.002)
                                torch.testing.assert_close(cv, ref_v, atol=0, rtol=0)
                                # Untouched rows must remain exact, not merely within the RoPE tolerance.
                                for got, expected in [(ck, ref_k), (cv, ref_v)]:
                                    torch.testing.assert_close(got[:, :, :position], expected[:, :, :position], atol=0, rtol=0)
                                    torch.testing.assert_close(got[:, :, position + seq:], expected[:, :, position + seq:], atol=0, rtol=0)

    def test_rope_graph_replay_updates_position(self):
        for role, heads, dim in [('target', 4, 128), ('draft', 2, 64)]:
            ops = importlib.import_module(f'runtime.production_kernels.{role}.attention.ops')
            batch, seq, capacity = 2, 33, 128
            k, v = inputs(batch, heads, seq, dim), inputs(batch, heads, seq, dim)
            cos, sin = tables(batch, seq, dim)
            ck = torch.full((batch, heads, capacity, dim), 7., device='cuda', dtype=torch.float16)
            cv = torch.full_like(ck, -7.)
            pos = torch.tensor(3, device='cuda')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ops.rope_kv_write_forward_dev(k, v, ck, cv, pos, cos, sin)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ops.rope_kv_write_forward_dev(k, v, ck, cv, pos, cos, sin)
            for position in [3, 19, capacity - seq]:
                ck.fill_(7.); cv.fill_(-7.); pos.fill_(position)
                graph.replay()
                expected_k, expected_v = torch.full_like(ck, 7.), torch.full_like(cv, -7.)
                expected_k[:, :, position:position + seq] = rotate(k, cos, sin)
                expected_v[:, :, position:position + seq] = v
                torch.testing.assert_close(ck, expected_k, atol=.002, rtol=.002)
                torch.testing.assert_close(cv, expected_v, atol=0, rtol=0)

    def test_scalar_attention_all_query_sizes_and_split_boundary(self):
        for role, heads, kv_heads, dim in [('target', 28, 4, 128), ('draft', 14, 2, 64)]:
            ops = importlib.import_module(f'runtime.production_kernels.{role}.attention.ops')
            for batch in [1, 2]:
                for seq in range(1, 10):
                    for length in [seq, 255, 256, 257, 1742]:
                        with self.subTest(role=role, batch=batch, seq=seq, length=length):
                            q = inputs(batch, heads, seq, dim)
                            k, v = inputs(batch, kv_heads, 2048, dim), inputs(batch, kv_heads, 2048, dim)
                            cos, sin = tables(batch, seq, dim)
                            mask = torch.arange(length, device='cuda')[None] <= (length-seq+torch.arange(seq, device='cuda')[:, None])
                            ref = F.scaled_dot_product_attention(rotate(q, cos, sin).float(),
                                k[:, :, :length].repeat_interleave(heads//kv_heads, 1).float(),
                                v[:, :, :length].repeat_interleave(heads//kv_heads, 1).float(), attn_mask=mask).half()
                            actual = ops.small_q_attn_forward(q, k, v, length, 1/math.sqrt(dim), cos, sin)
                            dev = ops.small_q_attn_forward_dev(q, k, v, torch.tensor(length, device='cuda'), 1/math.sqrt(dim), cos, sin)
                            torch.testing.assert_close(actual, ref, atol=.004, rtol=.04)
                            torch.testing.assert_close(dev, ref, atol=.004, rtol=.04)

if __name__ == '__main__':
    unittest.main()
