"""Tests for target attention AOT kernel integration (Phase 3).

Attention fused/decode kernels are templated on head_dim=128 (7B). Tests use
CONFIG_7B dimensions throughout.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from runtime.core.config import CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights
from runtime.production_kernels.target.attention import (
    decode_attn_forward,
    fused_attn_forward,
    o_proj_forward,
    qkv_proj_forward,
    rope_kv_write_forward,
    small_q_attn_forward,
)
from runtime.production_kernels.target.attention.ops import EXTENSION_MODULE

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
)
REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


def _qwen7b_attn_dims(cfg: RuntimeConfig):
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    d = cfg.head_dim
    h = cfg.hidden_size
    hq = nh * d
    hkv = nkv * d
    scale = 1.0 / math.sqrt(d)
    groups = nh // nkv
    return h, nh, nkv, d, hq, hkv, scale, groups


class TestAttentionAot(unittest.TestCase):
    def test_prebuilt_extension_importable(self) -> None:
        ext = importlib.import_module(EXTENSION_MODULE)
        for name in (
            "qkv_proj_forward",
            "rope_kv_write_forward",
            "fused_attn_forward",
            "decode_attn_forward",
            "small_q_attn_forward",
            "o_proj_forward",
        ):
            self.assertTrue(hasattr(ext, name), msg=name)
        self.assertNotIn("torch_extensions", ext.__file__)
        self.assertIn(
            "production_kernels/target/attention", ext.__file__.replace("\\", "/")
        )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestQkvProjGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        cls.h, cls.nh, cls.nkv, cls.d, cls.hq, cls.hkv, _, _ = _qwen7b_attn_dims(cls.cfg)

    def test_matches_eager_qkv(self) -> None:
        q_proj = nn.Linear(self.h, self.hq, bias=True).cuda().half()
        k_proj = nn.Linear(self.h, self.hkv, bias=True).cuda().half()
        v_proj = nn.Linear(self.h, self.hkv, bias=True).cuda().half()
        w_qkv = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0).contiguous()
        b_qkv = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0).contiguous()

        for batch, seq in [(1, 1), (1, 128), (4, 512)]:
            x = torch.randn(batch * seq, self.h, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = torch.cat([q_proj(x), k_proj(x), v_proj(x)], dim=-1)
                actual = qkv_proj_forward(x, w_qkv, b_qkv)
            self.assertTrue(
                torch.allclose(expected, actual, atol=0.5, rtol=0),
                msg=f"B={batch} S={seq}",
            )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestRopeKvWriteGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        _, _, cls.nkv, cls.d, _, _, _, _ = _qwen7b_attn_dims(cls.cfg)

    def _make_inputs(self, batch, seq_len):
        max_seq = 2048
        new_k = torch.randn(batch, self.nkv, seq_len, self.d, dtype=torch.float16, device="cuda")
        new_v = torch.randn(batch, self.nkv, seq_len, self.d, dtype=torch.float16, device="cuda")
        cache_k = torch.zeros(batch, self.nkv, max_seq, self.d, dtype=torch.float16, device="cuda")
        cache_v = torch.zeros(batch, self.nkv, max_seq, self.d, dtype=torch.float16, device="cuda")
        cos_half = torch.randn(batch, seq_len, self.d // 2, dtype=torch.float16, device="cuda")
        sin_half = torch.randn(batch, seq_len, self.d // 2, dtype=torch.float16, device="cuda")
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        return new_k, new_v, cache_k, cache_v, cos, sin

    def test_rope_kv_write(self) -> None:
        for batch, seq, write_pos in [(1, 1, 0), (1, 128, 64), (2, 128, 0)]:
            new_k, new_v, cache_k_ref, cache_v_ref, cos, sin = self._make_inputs(batch, seq)
            cache_k_cust = cache_k_ref.clone()
            cache_v_cust = cache_v_ref.clone()
            with torch.no_grad():
                _, k_rot = apply_rotary_pos_emb(new_k, new_k, cos, sin)
                cache_k_ref[:, :, write_pos : write_pos + seq, :] = k_rot
                cache_v_ref[:, :, write_pos : write_pos + seq, :] = new_v
                rope_kv_write_forward(
                    new_k, new_v, cache_k_cust, cache_v_cust, write_pos, cos, sin
                )
            self.assertTrue(
                torch.allclose(cache_k_ref, cache_k_cust, atol=1e-2, rtol=1e-2),
                msg=f"K B={batch} S={seq} wp={write_pos}",
            )
            self.assertTrue(
                torch.equal(cache_v_ref, cache_v_cust),
                msg=f"V B={batch} S={seq} wp={write_pos}",
            )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestFusedAttnGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        _, cls.nh, cls.nkv, cls.d, _, _, cls.scale, cls.groups = _qwen7b_attn_dims(cls.cfg)

    def _eager_prefill(self, q, cache_k, cache_v, cur_len):
        k_live = cache_k[:, :, :cur_len, :]
        v_live = cache_v[:, :, :cur_len, :]
        k_rep = repeat_kv(k_live, self.groups)
        v_rep = repeat_kv(v_live, self.groups)
        return F.scaled_dot_product_attention(
            q, k_rep, v_rep, is_causal=True, scale=self.scale
        )

    def test_fused_attn_prefill(self) -> None:
        for batch, seq in [(1, 1), (1, 128), (2, 128)]:
            q = torch.randn(batch, self.nh, seq, self.d, dtype=torch.float16, device="cuda")
            cache_k = torch.zeros(batch, self.nkv, seq, self.d, dtype=torch.float16, device="cuda")
            cache_v = torch.zeros(batch, self.nkv, seq, self.d, dtype=torch.float16, device="cuda")
            cache_k[:, :, :seq, :] = torch.randn_like(cache_k[:, :, :seq, :])
            cache_v[:, :, :seq, :] = torch.randn_like(cache_v[:, :, :seq, :])
            with torch.no_grad():
                expected = self._eager_prefill(q, cache_k, cache_v, seq)
                actual = fused_attn_forward(q, cache_k, cache_v, seq, self.scale)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
                msg=f"B={batch} S={seq}",
            )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestDecodeAttnGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        _, cls.nh, cls.nkv, cls.d, _, _, cls.scale, cls.groups = _qwen7b_attn_dims(cls.cfg)

    def _causal_mask(self, seq_len, cur_len, device):
        q_pos = torch.arange(cur_len - seq_len, cur_len, device=device).view(seq_len, 1)
        k_pos = torch.arange(cur_len, device=device).view(1, cur_len)
        return k_pos <= q_pos

    def _eager_decode(self, q, cache_k, cache_v, cur_len):
        mask = self._causal_mask(q.shape[2], cur_len, q.device)
        k = repeat_kv(cache_k[:, :, :cur_len, :], self.groups)
        v = repeat_kv(cache_v[:, :, :cur_len, :], self.groups)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

    def test_decode_single_token(self) -> None:
        for batch, history in [(1, 0), (1, 127), (1, 1024)]:
            seq = 1
            cur_len = history + seq
            pad = 16
            q = torch.randn(batch, self.nh, seq, self.d, dtype=torch.float16, device="cuda")
            cache_k = torch.randn(batch, self.nkv, cur_len + pad, self.d, dtype=torch.float16, device="cuda")
            cache_v = torch.randn(batch, self.nkv, cur_len + pad, self.d, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = self._eager_decode(q, cache_k, cache_v, cur_len)
                actual = decode_attn_forward(q, cache_k, cache_v, cur_len, self.scale)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
                msg=f"B={batch} history={history}",
            )

    def test_small_q_verify(self) -> None:
        batch, seq, history = 1, 4, 1024
        cur_len = history + seq
        pad = 16
        q = torch.randn(batch, self.nh, seq, self.d, dtype=torch.float16, device="cuda")
        cache_k = torch.randn(batch, self.nkv, cur_len + pad, self.d, dtype=torch.float16, device="cuda")
        cache_v = torch.randn(batch, self.nkv, cur_len + pad, self.d, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = self._eager_decode(q, cache_k, cache_v, cur_len)
            actual = small_q_attn_forward(q, cache_k, cache_v, cur_len, self.scale)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-2, rtol=1e-2))


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestOProjGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        cls.h, _, _, _, cls.hq, _, _, _ = _qwen7b_attn_dims(cls.cfg)

    def test_o_proj(self) -> None:
        linear = nn.Linear(self.hq, self.h, bias=False).cuda().half()
        for batch, seq in [(1, 1), (1, 128), (4, 512)]:
            x = torch.randn(batch * seq, self.hq, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = linear(x)
                actual = o_proj_forward(x, linear.weight)
            self.assertTrue(
                torch.allclose(expected, actual, atol=0.5, rtol=0),
                msg=f"B={batch} S={seq}",
            )

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
        ),
        "7B weights not on disk",
    )
    def test_o_proj_loaded_weights(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        weights = load_weights(cfg, device="cuda")
        w_o = weights["model.layers.0.self_attn.o_proj.weight"]
        h, _, _, _, hq, _, _, _ = _qwen7b_attn_dims(cfg)
        x = torch.randn(8, hq, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = F.linear(x, w_o)
            actual = o_proj_forward(x, w_o)
        self.assertTrue(torch.allclose(expected, actual, atol=0.5, rtol=0))


if __name__ == "__main__":
    unittest.main()
