"""Tests for target SwiGLU AOT kernel integration (Phase 3)."""

import importlib
import os
import unittest

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights
from runtime.production_kernels.target.swiglu import swiglu_forward
from runtime.production_kernels.target.swiglu.ops import EXTENSION_MODULE

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", "/home/cme213/tobiascm/cme213-final-project"
)
REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


class TestSwigluAot(unittest.TestCase):
    def test_prebuilt_extension_importable(self) -> None:
        ext = importlib.import_module(EXTENSION_MODULE)
        self.assertTrue(hasattr(ext, "swiglu_forward"))
        self.assertNotIn("torch_extensions", ext.__file__)
        self.assertIn("production_kernels/target/swiglu", ext.__file__.replace("\\", "/"))


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestSwigluGpu(unittest.TestCase):
    def _cfg_to_qwen2_config(self, cfg: RuntimeConfig):
        from transformers import Qwen2Config

        return Qwen2Config(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            hidden_act=cfg.hidden_act,
        )

    def test_forward_matches_hf_7b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        hf = Qwen2MLP(self._cfg_to_qwen2_config(cfg)).cuda().half()

        for batch, seq in [(1, 1), (1, 128), (4, 512)]:
            x = torch.randn(batch, seq, cfg.hidden_size, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                expected = hf(x)
                actual = swiglu_forward(
                    x, hf.gate_proj.weight, hf.up_proj.weight, hf.down_proj.weight
                )
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-2, rtol=1e-2),
                msg=f"B={batch} S={seq}",
            )

    def test_forward_matches_hf_05b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        hf = Qwen2MLP(self._cfg_to_qwen2_config(cfg)).cuda().half()
        x = torch.randn(2, 64, cfg.hidden_size, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = hf(x)
            actual = swiglu_forward(
                x, hf.gate_proj.weight, hf.up_proj.weight, hf.down_proj.weight
            )
        self.assertTrue(torch.allclose(expected, actual, atol=1e-2, rtol=1e-2))

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
        ),
        "7B weights not on disk",
    )
    def test_with_loaded_layer0_weights(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        weights = load_weights(cfg, device="cuda")
        prefix = "model.layers.0.mlp"
        w_gate = weights[f"{prefix}.gate_proj.weight"]
        w_up = weights[f"{prefix}.up_proj.weight"]
        w_down = weights[f"{prefix}.down_proj.weight"]

        hf = Qwen2MLP(self._cfg_to_qwen2_config(cfg)).cuda().half()
        hf.gate_proj.weight.data.copy_(w_gate)
        hf.up_proj.weight.data.copy_(w_up)
        hf.down_proj.weight.data.copy_(w_down)

        x = torch.randn(1, 32, cfg.hidden_size, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            expected = hf(x)
            actual = swiglu_forward(x, w_gate, w_up, w_down)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
