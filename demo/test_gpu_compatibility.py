"""Architecture checks for attention's fragment indexing and partial tiles.

Run after compiling on the selected GPU:
    python -m unittest demo.test_gpu_compatibility -v
"""

import math
import unittest

import torch

from runtime.production_kernels.target.attention import (
    fused_attn_forward, small_q_attn_forward, small_q_attn_forward_dev,
)


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA and the target AOT build")
class TestAttentionPortability(unittest.TestCase):
    @staticmethod
    def inputs(seq, length):
        q = torch.randn(1, 28, seq, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 4, length + 16, 128, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        angles = torch.randn(1, seq, 64, device="cuda")
        cos = torch.cat([angles.cos()] * 2, dim=-1).half()
        sin = torch.cat([angles.sin()] * 2, dim=-1).half()
        return q, k, v, cos, sin

    @staticmethod
    def reference(q, k, v, length, cos, sin):
        # FP32 rotation rounded to FP16, as in the fused Q rotation.
        x = q.float()
        rotated = torch.cat([-x[..., 64:], x[..., :64]], dim=-1)
        q = (x * cos[:, None].float() + rotated * sin[:, None].float()).half().float()
        k = k[:, :, :length].repeat_interleave(7, dim=1).float()
        v = v[:, :, :length].repeat_interleave(7, dim=1).float()
        positions = torch.arange(length - q.size(2), length, device="cuda")
        mask = torch.arange(length, device="cuda")[None, :] <= positions[:, None]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(128)
        return (scores.masked_fill(~mask, -torch.inf).softmax(dim=-1) @ v).half()

    @torch.inference_mode()
    def test_prefill_partial_tiles_with_rope(self):
        torch.manual_seed(213)
        for seq in (1, 31, 63, 64, 65, 127, 129):
            with self.subTest(seq=seq):
                q, k, v, cos, sin = self.inputs(seq, seq)
                expected = self.reference(q, k, v, seq, cos, sin)
                actual = fused_attn_forward(q, k, v, seq, 1 / math.sqrt(128), cos, sin)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @torch.inference_mode()
    def test_decode_split_kv_and_device_lengths(self):
        torch.manual_seed(214)
        for seq, length in ((1, 1), (1, 257), (1, 2048), (4, 1025), (9, 258)):
            with self.subTest(seq=seq, length=length):
                q, k, v, cos, sin = self.inputs(seq, length)
                expected = self.reference(q, k, v, length, cos, sin)
                actual = small_q_attn_forward(q, k, v, length, 1 / math.sqrt(128), cos, sin)
                device_length = torch.tensor(length, dtype=torch.int64, device="cuda")
                actual_dev = small_q_attn_forward_dev(q, k, v, device_length, 1 / math.sqrt(128), cos, sin)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(actual_dev, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
