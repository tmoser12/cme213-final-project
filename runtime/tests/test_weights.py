"""Tests for GPU weight loading and VRAM budget.

VRAM budget tests target a single model per GPU. Production sizing uses 7B only
(TestGpuLoad7B). The 0.5B GPU tests are a fast dev smoke path — never load both
models on the same device.
"""

import os
import unittest

import torch

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.memory import max_seq_len_after_weights, max_seq_len_for_budget, plan_memory, runtime_bytes_per_seq
from runtime.core.weights import (
    expected_keys,
    gpu_memory_snapshot,
    load_weights,
    load_weights_on_gpu,
    memory_report,
    vram_budget,
)

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", "/home/cme213/tobiascm/cme213-final-project"
)
REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP_REASON = "CUDA not available — run: bash slurm/run_tests_gpu.sh"


class TestExpectedKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg_7b = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        cls.cfg_05b = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)

    def test_7b_key_count(self) -> None:
        keys = expected_keys(self.cfg_7b)
        self.assertEqual(len(keys), 3 + 12 * 28)
        self.assertIn("lm_head.weight", keys)

    def test_05b_key_count_no_lm_head(self) -> None:
        keys = expected_keys(self.cfg_05b)
        self.assertEqual(len(keys), 2 + 12 * 24)
        self.assertNotIn("lm_head.weight", keys)


class TestMaxSeqLenMath(unittest.TestCase):
    """CPU-only math tests for buffer budgeting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg_7b = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        cls.cfg_05b = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)

    def test_runtime_scales_linearly_with_seq(self) -> None:
        cfg = self.cfg_05b
        p1 = plan_memory(cfg, batch=1, max_seq_len=1)["runtime_bytes"]
        p128 = plan_memory(cfg, batch=1, max_seq_len=128)["runtime_bytes"]
        self.assertEqual(p128, p1 * 128)

    def test_max_seq_len_for_budget(self) -> None:
        cfg = self.cfg_05b
        per_seq = runtime_bytes_per_seq(cfg, batch=1)
        budget = per_seq * 512
        self.assertEqual(max_seq_len_for_budget(cfg, budget, batch=1), 512)

    def test_max_seq_len_capped_by_yaml(self) -> None:
        cfg = self.cfg_7b
        huge_budget = 100 * 1024**3
        self.assertEqual(
            max_seq_len_for_budget(cfg, huge_budget, batch=1),
            cfg.max_position_embeddings,
        )

    def test_max_seq_len_after_7b_weights_on_24gb_gpu(self) -> None:
        """Canonical budget: 7B weights only, ~9.5 GiB left for buffers on 24GB Turing."""
        cfg = self.cfg_7b
        weight_mib = plan_memory(cfg, batch=1, max_seq_len=1)["weight_mib"]
        total_gib = 24 * 1024**3
        # free HBM after 7B weights only (single model on GPU)
        free_after_7b = total_gib - int(weight_mib * 1024 * 1024)
        info = max_seq_len_after_weights(
            cfg, free_vram_bytes=free_after_7b, batch=1, reserve_bytes=512 * 1024**2
        )
        self.assertGreater(info["max_seq_len"], 1000)
        plan = info["plan_at_max_seq_len"]
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertLessEqual(plan["runtime_bytes"], info["buffer_budget_bytes"])


def _has_05b_weights() -> bool:
    return os.path.isdir(os.path.join(PROJECT_ROOT, "models/Qwen2.5-0.5B-Instruct"))


def _has_7b_weights() -> bool:
    return os.path.isfile(
        os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
    )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP_REASON)
@unittest.skipUnless(_has_05b_weights(), "0.5B weights not on disk")
class TestGpuLoad05B(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @classmethod
    def tearDownClass(cls) -> None:
        torch.cuda.empty_cache()

    def test_load_all_weights_on_gpu_hbm(self) -> None:
        weights, budget = load_weights_on_gpu(self.cfg, batch=1)
        self.assertEqual(len(weights), len(expected_keys(self.cfg)))
        for name, t in weights.items():
            self.assertEqual(t.device.type, "cuda", msg=name)
            self.assertEqual(t.dtype, torch.float16, msg=name)
        self.assertGreater(budget["weights"]["total_mib"], 900)
        self.assertLess(budget["weights"]["total_mib"], 1100)

    def test_vram_free_and_max_seq_len(self) -> None:
        weights, budget = load_weights_on_gpu(self.cfg, batch=1, reserve_mib=512)
        gpu = budget["gpu"]
        self.assertGreater(gpu["total_mib"], 20000)  # Quadro RTX 6000 ~24GB
        self.assertGreater(gpu["free_mib"], 10000)   # plenty left after ~1GB weights
        self.assertGreater(budget["max_seq_len"], 4096)
        self.assertLessEqual(budget["max_seq_len"], self.cfg.max_position_embeddings)
        # budgeted runtime fits in free VRAM
        plan = plan_memory(self.cfg, batch=1, max_seq_len=budget["max_seq_len"])
        self.assertLessEqual(
            plan["runtime_bytes"],
            gpu["free_bytes"] - int(512 * 1024 * 1024),
        )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP_REASON)
@unittest.skipUnless(_has_7b_weights(), "7B weights not on disk")
class TestGpuLoad7B(unittest.TestCase):
    """Canonical GPU test: 7B only on HBM, then compute buffer budget."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @classmethod
    def tearDownClass(cls) -> None:
        torch.cuda.empty_cache()

    def test_load_all_weights_on_gpu_hbm(self) -> None:
        weights, budget = load_weights_on_gpu(self.cfg, batch=1)
        self.assertEqual(len(weights), len(expected_keys(self.cfg)))
        for name, t in weights.items():
            self.assertEqual(t.device.type, "cuda", msg=name)
            self.assertEqual(t.dtype, torch.float16, msg=name)
        self.assertIn("lm_head.weight", weights)
        self.assertGreater(budget["weights"]["total_mib"], 14000)
        self.assertLess(budget["weights"]["total_mib"], 15000)

    def test_vram_free_and_max_seq_len(self) -> None:
        """Remaining HBM is for buffers only — 7B weights are the sole occupant."""
        weights, budget = load_weights_on_gpu(self.cfg, batch=1, reserve_mib=512)
        self.assertTrue(budget["single_model_on_gpu"])
        self.assertEqual(budget["model"], "qwen2.5-7b-instruct")

        gpu = budget["gpu"]
        w_mib = budget["weights"]["total_mib"]
        # weights + reserved ≈ total (single model, no second model loaded)
        self.assertGreater(w_mib, 14000)
        self.assertLess(w_mib, 15000)
        self.assertGreater(gpu["free_mib"], 500)
        self.assertLess(gpu["reserved_mib"], gpu["total_mib"])
        self.assertGreater(budget["max_seq_len"], 128)
        self.assertLessEqual(budget["max_seq_len"], self.cfg.max_position_embeddings)

        plan = plan_memory(self.cfg, batch=1, max_seq_len=budget["max_seq_len"])
        self.assertLessEqual(
            plan["runtime_bytes"],
            gpu["free_bytes"] - int(512 * 1024 * 1024),
        )

    def test_only_7b_weights_on_gpu_not_both_models(self) -> None:
        """Confirm reserved VRAM matches ~7B weights alone, not 7B+0.5B."""
        _, budget = load_weights_on_gpu(self.cfg, batch=1)
        reserved_mib = budget["gpu"]["reserved_mib"]
        w_mib = budget["weights"]["total_mib"]
        # reserved should be close to weight bytes (+ small CUDA context), not 2x
        self.assertLess(reserved_mib, w_mib + 1024)
        self.assertGreater(reserved_mib, w_mib * 0.9)

    def test_gpu_memory_snapshot(self) -> None:
        snap = gpu_memory_snapshot(0)
        self.assertIn("Quadro", snap["device_name"])
        self.assertGreater(snap["total_mib"], 20000)


if __name__ == "__main__":
    unittest.main()
