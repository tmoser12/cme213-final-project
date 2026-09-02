"""Tests for target residual_ops AOT kernel integration (Phase 3)."""

import importlib
import os
from pathlib import Path
import unittest

import torch
import torch.nn as nn

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights
from runtime.production_kernels.target.residual_ops import (
    lm_head_forward,
    residual_add_forward,
)
from runtime.production_kernels.target.residual_ops.ops import EXTENSION_MODULE

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
)
REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


class TestResidualOpsAot(unittest.TestCase):
    def test_prebuilt_extension_importable(self) -> None:
        ext = importlib.import_module(EXTENSION_MODULE)
        self.assertTrue(hasattr(ext, "residual_add_forward"))
        self.assertTrue(hasattr(ext, "lm_head_forward"))
        self.assertNotIn("torch_extensions", ext.__file__)
        self.assertIn(
            "production_kernels/target/residual_ops", ext.__file__.replace("\\", "/")
        )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestResidualOpsGpu(unittest.TestCase):
    def test_residual_add_bit_exact_7b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        for rows in [1, 128, 512]:
            a = torch.randn(rows, cfg.hidden_size, dtype=torch.float16, device="cuda")
            b = torch.randn(rows, cfg.hidden_size, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = a + b
                actual = residual_add_forward(a, b)
            self.assertTrue(torch.equal(expected, actual), msg=f"rows={rows}")

    def test_lm_head_matches_linear_05b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).cuda().half()
        for batch, seq in [(1, 1), (2, 64)]:
            hidden = torch.randn(batch, seq, cfg.hidden_size, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = head(hidden)
                actual = lm_head_forward(hidden, head.weight)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
                msg=f"B={batch} S={seq}",
            )

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
        ),
        "7B weights not on disk",
    )
    def test_lm_head_with_loaded_weights(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        weights = load_weights(cfg, device="cuda")
        w = weights["lm_head.weight"]
        hidden = torch.randn(1, 8, cfg.hidden_size, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = torch.nn.functional.linear(hidden, w)
            actual = lm_head_forward(hidden, w)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
