"""Tests for target embedding AOT kernel integration (Phase 3)."""

import importlib
import os
import unittest

import torch
import torch.nn as nn

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights
from runtime.production_kernels.target.embedding import embedding_forward
from runtime.production_kernels.target.embedding.ops import EXTENSION_MODULE

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", "/home/cme213/tobiascm/cme213-final-project"
)
REQUIRES_GPU = not torch.cuda.is_available()
GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"


class TestEmbeddingAot(unittest.TestCase):
    def test_prebuilt_extension_importable(self) -> None:
        ext = importlib.import_module(EXTENSION_MODULE)
        self.assertTrue(hasattr(ext, "embedding_forward"))
        self.assertNotIn("torch_extensions", ext.__file__)
        self.assertIn("production_kernels/target/embedding", ext.__file__.replace("\\", "/"))


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
class TestEmbeddingGpu(unittest.TestCase):
    def test_matches_hf_7b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        hf = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().half()

        for batch, seq in [(1, 1), (1, 128), (4, 512)]:
            input_ids = torch.randint(
                0, cfg.vocab_size, (batch, seq), dtype=torch.int64, device="cuda"
            )
            with torch.no_grad():
                expected = hf(input_ids)
                actual = embedding_forward(input_ids, hf.weight)
            self.assertTrue(torch.equal(expected, actual), msg=f"B={batch} S={seq}")

    def test_matches_hf_05b(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        hf = nn.Embedding(cfg.vocab_size, cfg.hidden_size).cuda().half()
        input_ids = torch.randint(
            0, cfg.vocab_size, (2, 64), dtype=torch.int64, device="cuda"
        )
        with torch.no_grad():
            expected = hf(input_ids)
            actual = embedding_forward(input_ids, hf.weight)
        self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(PROJECT_ROOT, "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors")
        ),
        "7B weights not on disk",
    )
    def test_with_loaded_model_weights(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
        weights = load_weights(cfg, device="cuda")
        w = weights["model.embed_tokens.weight"]
        input_ids = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.int64, device="cuda")
        with torch.no_grad():
            expected = torch.nn.functional.embedding(input_ids, w)
            actual = embedding_forward(input_ids, w)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
