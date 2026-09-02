"""Tests for Phase 4 buffer allocation."""

from __future__ import annotations

import os
import unittest

import torch

from runtime.buffers import allocate_buffers, build_rope_tables, buffer_fits_vram_budget
from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.memory import plan_memory
from runtime.core.weights import load_weights_on_gpu
from runtime.tests._support import PROJECT_ROOT, load_05b, load_7b

REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


class TestBufferAllocationCpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg_7b = load_7b()
        cls.cfg_05b = load_05b()

    def test_shapes_match_plan_memory_7b(self) -> None:
        cfg = self.cfg_7b
        batch, seq = 1, 512
        plan = plan_memory(cfg, batch=batch, max_seq_len=seq)
        device = "cuda" if not REQUIRES_GPU else "cpu"
        buffers = allocate_buffers(cfg, batch=batch, max_seq_len=seq, device=device)

        self.assertEqual(tuple(buffers.hidden_a.shape), plan["buffers"]["hidden_ping"])
        self.assertEqual(tuple(buffers.hidden_b.shape), plan["buffers"]["hidden_pong"])
        self.assertEqual(tuple(buffers.q_states.shape), plan["buffers"]["q_states"])
        self.assertEqual(tuple(buffers.kv_cache_k.shape), plan["buffers"]["kv_cache_keys"])
        self.assertEqual(tuple(buffers.logits.shape), plan["buffers"]["logits"])
        self.assertEqual(buffers.rope_cos.shape, (seq, cfg.head_dim))
        self.assertEqual(buffers.cache_position.shape, ())
        self.assertEqual(buffers.cache_position.dtype, torch.int64)

    def test_byte_counts_match_plan_05b_cpu(self) -> None:
        cfg = self.cfg_05b
        plan = plan_memory(cfg, batch=1, max_seq_len=128)
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=128, device="cpu")
        report = buffers.memory_report()
        for key, expected in plan["buffer_bytes"].items():
            self.assertEqual(report[key], expected, msg=key)
        self.assertEqual(buffers.nbytes(), sum(report.values()))

    def test_kv_cache_row_16_byte_aligned(self) -> None:
        cfg = self.cfg_7b
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=64, device="cpu")
        row_bytes = buffers.kv_head_dim * cfg.dtype_bytes
        self.assertEqual(row_bytes % 16, 0)

    def test_rope_tables_match_hf(self) -> None:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        from transformers import Qwen2Config

        cfg = self.cfg_7b
        hf_cfg = Qwen2Config(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
        )
        seq = 32
        rotary = Qwen2RotaryEmbedding(config=hf_cfg)
        x = torch.zeros(1, seq, cfg.hidden_size)
        pos = torch.arange(seq).unsqueeze(0)
        cos_hf, sin_hf = rotary(x, pos)

        cos, sin = build_rope_tables(cfg, seq, torch.device("cpu"))
        self.assertTrue(
            torch.allclose(cos_hf[0].float(), cos[:seq].float(), atol=1e-3, rtol=1e-3)
        )
        self.assertTrue(
            torch.allclose(sin_hf[0].float(), sin[:seq].float(), atol=1e-3, rtol=1e-3)
        )

    def test_rope_embeddings_expand_batch(self) -> None:
        buffers = allocate_buffers(self.cfg_05b, batch=2, max_seq_len=64, device="cpu")
        cos, sin = buffers.rope_embeddings(start=4, length=8)
        self.assertEqual(cos.shape, (2, 8, self.cfg_05b.head_dim))
        self.assertEqual(sin.shape, (2, 8, self.cfg_05b.head_dim))

    def test_swap_hidden_exchanges_tensors(self) -> None:
        buffers = allocate_buffers(self.cfg_05b, batch=1, max_seq_len=8, device="cpu")
        a_id = id(buffers.hidden_a)
        b_id = id(buffers.hidden_b)
        buffers.swap_hidden()
        self.assertEqual(id(buffers.hidden_a), b_id)
        self.assertEqual(id(buffers.hidden_b), a_id)

    def test_kv_layer_views(self) -> None:
        buffers = allocate_buffers(self.cfg_05b, batch=1, max_seq_len=16, device="cpu")
        k0 = buffers.kv_cache_k_layer(0)
        self.assertEqual(
            tuple(k0.shape),
            (1, self.cfg_05b.num_key_value_heads, 16, self.cfg_05b.head_dim),
        )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestBufferAllocationGpu(unittest.TestCase):
    def test_allocate_on_cuda_7b(self) -> None:
        cfg = load_7b()
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=512, device="cuda")
        self.assertEqual(buffers.hidden_a.device.type, "cuda")
        self.assertEqual(buffers.cache_position.device.type, "cuda")

    @unittest.skipUnless(
        os.path.isdir(os.path.join(PROJECT_ROOT, "models/Qwen2.5-0.5B-Instruct")),
        "0.5B weights not on disk",
    )
    def test_buffers_fit_after_05b_weights(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        weights, budget = load_weights_on_gpu(cfg, batch=1, reserve_mib=512)
        max_seq = min(512, budget["max_seq_len"])
        self.assertGreater(max_seq, 0)
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=max_seq, device="cuda")
        plan = plan_memory(cfg, batch=1, max_seq_len=max_seq)
        self.assertEqual(buffers.nbytes(), sum(plan["buffer_bytes"].values()))
        check = buffer_fits_vram_budget(cfg, weights, buffers, reserve_mib=512)
        self.assertTrue(check["fits"])

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
        ),
        "7B weights not on disk",
    )
    def test_buffers_fit_after_7b_weights(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        weights, budget = load_weights_on_gpu(cfg, batch=1, reserve_mib=512)
        max_seq = budget["max_seq_len"]
        self.assertGreater(max_seq, 0)
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=max_seq, device="cuda")
        self.assertLessEqual(
            buffers.nbytes(),
            budget["buffer_budget_mib"] * 1024 * 1024 + 1024,
        )


if __name__ == "__main__":
    unittest.main()
