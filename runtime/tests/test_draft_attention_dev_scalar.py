"""Phase D2 parity: draft device-scalar attention ops vs host-int ops (bit-exact).

Mirrors test_attention_dev_scalar but for the DRAFT kernels (head_dim=64,
Qwen2.5-0.5B GQA: 14 q-heads / 2 kv-heads). The `_dev` variants read
write_pos/cur_len from a 0-d int64 CUDA tensor; outside a graph they must compute
exactly the same thing as the host-int path.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_draft_attention_dev_scalar
"""

from __future__ import annotations

import math
import unittest

import torch

from runtime.production_kernels.draft.attention import (
    decode_attn_forward,
    decode_attn_forward_dev,
    rope_kv_write_forward,
    rope_kv_write_forward_dev,
    small_q_attn_forward,
    small_q_attn_forward_dev,
)

GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"

# 0.5B draft attention dims (kernels templated on head_dim=64).
NH, NKV, D = 14, 2, 64
MAX_SEQ = 96
SCALE = 1.0 / math.sqrt(D)
DEV = "cuda"
DT = torch.float16


def _rand(*shape):
    return (torch.randn(*shape, device=DEV, dtype=torch.float32) * 0.1).to(DT)


def _scalar(value: int) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int64, device=DEV)


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestDraftAttentionDevScalar(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _identity_rope(self, S):
        cos = torch.ones((1, S, D), device=DEV, dtype=DT)
        sin = torch.zeros((1, S, D), device=DEV, dtype=DT)
        return cos, sin

    def test_decode_attn_dev_matches_host(self):
        q = _rand(1, NH, 1, D)
        cache_k = _rand(1, NKV, MAX_SEQ, D)
        cache_v = _rand(1, NKV, MAX_SEQ, D)
        cos, sin = self._identity_rope(1)
        for cur_len in (1, 5, 17, 64, MAX_SEQ):
            out_int = decode_attn_forward(q, cache_k, cache_v, cur_len, SCALE, cos, sin)
            out_dev = decode_attn_forward_dev(q, cache_k, cache_v, _scalar(cur_len), SCALE, cos, sin)
            self.assertTrue(torch.equal(out_int, out_dev),
                            f"decode_attn dev != host at cur_len={cur_len}")

    def test_decode_attn_dev_matches_host_no_rope(self):
        q = _rand(1, NH, 1, D)
        cache_k = _rand(1, NKV, MAX_SEQ, D)
        cache_v = _rand(1, NKV, MAX_SEQ, D)
        for cur_len in (1, 33, MAX_SEQ):
            out_int = decode_attn_forward(q, cache_k, cache_v, cur_len, SCALE)
            out_dev = decode_attn_forward_dev(q, cache_k, cache_v, _scalar(cur_len), SCALE)
            self.assertTrue(torch.equal(out_int, out_dev),
                            f"decode_attn (no rope) dev != host at cur_len={cur_len}")

    def test_small_q_attn_dev_matches_host(self):
        cache_k = _rand(1, NKV, MAX_SEQ, D)
        cache_v = _rand(1, NKV, MAX_SEQ, D)
        for S in (2, 4, 5):
            q = _rand(1, NH, S, D)
            cos, sin = self._identity_rope(S)
            for cur_len in (S, 20, MAX_SEQ):
                out_int = small_q_attn_forward(q, cache_k, cache_v, cur_len, SCALE, cos, sin)
                out_dev = small_q_attn_forward_dev(q, cache_k, cache_v, _scalar(cur_len), SCALE, cos, sin)
                self.assertTrue(torch.equal(out_int, out_dev),
                                f"small_q dev != host at S={S}, cur_len={cur_len}")

    def test_rope_kv_write_dev_matches_host(self):
        S = 1
        for write_pos in (0, 1, 30, MAX_SEQ - S):
            new_k = _rand(1, NKV, S, D)
            new_v = _rand(1, NKV, S, D)
            cos, sin = self._identity_rope(S)
            ck_int = torch.zeros((1, NKV, MAX_SEQ, D), device=DEV, dtype=DT)
            cv_int = torch.zeros_like(ck_int)
            ck_dev = torch.zeros_like(ck_int)
            cv_dev = torch.zeros_like(ck_int)
            rope_kv_write_forward(new_k, new_v, ck_int, cv_int, write_pos, cos, sin)
            rope_kv_write_forward_dev(new_k, new_v, ck_dev, cv_dev, _scalar(write_pos), cos, sin)
            self.assertTrue(torch.equal(ck_int, ck_dev), f"K cache mismatch at write_pos={write_pos}")
            self.assertTrue(torch.equal(cv_int, cv_dev), f"V cache mismatch at write_pos={write_pos}")


if __name__ == "__main__":
    unittest.main()
