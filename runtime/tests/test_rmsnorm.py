"""Tests for target RMSNorm AOT kernel integration (Phase 3)."""

import importlib
import os
from pathlib import Path
import unittest

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights
from runtime.production_kernels.target.rmsnorm import forward, init, workspace_bytes
from runtime.production_kernels.target.rmsnorm.ops import EXTENSION_MODULE

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
)
REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


class TestRmsnormAot(unittest.TestCase):
    def test_prebuilt_extension_importable(self) -> None:
        ext = importlib.import_module(EXTENSION_MODULE)
        self.assertTrue(hasattr(ext, "forward"))
        self.assertNotIn("torch_extensions", ext.__file__)
        self.assertIn("production_kernels/target/rmsnorm", ext.__file__.replace("\\", "/"))


class TestRmsnormInterface(unittest.TestCase):
    def test_workspace_bytes_is_zero(self) -> None:
        self.assertEqual(workspace_bytes(batch=1, seq_len=128, hidden_size=896), 0)


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestRmsnormGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        init()

    def test_forward_matches_hf_7b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        hf = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).cuda().half()

        for batch, seq in [(1, 1), (1, 128), (4, 512)]:
            x = torch.randn(batch, seq, cfg.hidden_size, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = hf(x)
                actual = forward(x, hf.weight, cfg.rms_norm_eps)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-3, rtol=1e-3),
                msg=f"batch={batch} seq={seq}",
            )

    def test_forward_matches_hf_05b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        hf = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).cuda().half()

        x = torch.randn(2, 64, cfg.hidden_size, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = hf(x)
            actual = forward(x, hf.weight, cfg.rms_norm_eps)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-3))

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
        ),
        "7B weights not on disk",
    )
    def test_forward_with_loaded_model_weights(self) -> None:
        """Uses real layer-0 input_layernorm weight from safetensors."""
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        weights = load_weights(cfg, device="cuda")
        w = weights["model.layers.0.input_layernorm.weight"]

        x = torch.randn(1, 32, cfg.hidden_size, dtype=torch.float16, device="cuda")
        hf = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).cuda().half()
        hf.weight.data.copy_(w)

        with torch.no_grad():
            expected = hf(x)
            actual = forward(x, w, cfg.rms_norm_eps)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
