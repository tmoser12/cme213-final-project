"""Tests for YAML config loading."""

import os
import unittest

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig


class TestRuntimeConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = os.environ.get(
            "PROJECT_ROOT", "/home/cme213/tobiascm/cme213-final-project"
        )

    def test_load_7b_yaml(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=self.root)
        cfg.validate()
        self.assertEqual(cfg.name, "qwen2.5-7b-instruct")
        self.assertEqual(cfg.hidden_size, 3584)
        self.assertEqual(cfg.num_hidden_layers, 28)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.num_kv_groups, 7)
        self.assertEqual(cfg.dtype, "fp16")
        self.assertFalse(cfg.tie_word_embeddings)
        self.assertTrue(cfg.model_path.endswith("Qwen2.5-7B-Instruct"))

    def test_load_05b_yaml(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=self.root)
        cfg.validate()
        self.assertEqual(cfg.hidden_size, 896)
        self.assertEqual(cfg.num_hidden_layers, 24)
        self.assertEqual(cfg.head_dim, 64)
        self.assertTrue(cfg.tie_word_embeddings)

    def test_layer_order_matches_reference(self) -> None:
        cfg = RuntimeConfig.from_yaml(CONFIG_7B, project_root=self.root)
        self.assertEqual(
            cfg.layer_order,
            (
                "input_rmsnorm",
                "attention",
                "residual_add",
                "post_attn_rmsnorm",
                "swiglu_mlp",
                "residual_add",
            ),
        )

    def test_custom_yaml_path(self) -> None:
        """Passing a path string works the same as the bundled constants."""
        cfg = RuntimeConfig.from_yaml(str(CONFIG_7B), project_root=self.root)
        cfg.validate()
        self.assertEqual(cfg.kv_cache_layout, "layer_major")


if __name__ == "__main__":
    unittest.main()
