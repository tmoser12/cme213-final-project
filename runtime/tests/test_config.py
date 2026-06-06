"""Setup tests: YAML config loads with correct architecture and policy."""

import unittest

from runtime.tests._support import LAYER_ORDER, PROJECT_ROOT, load_05b, load_7b
from runtime.core.config import CONFIG_7B, RuntimeConfig


class TestConfig7B(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_7b()

    def test_name_and_architecture(self) -> None:
        self.assertEqual(self.cfg.name, "qwen2.5-7b-instruct")
        self.assertEqual(self.cfg.hidden_size, 3584)
        self.assertEqual(self.cfg.intermediate_size, 18944)
        self.assertEqual(self.cfg.num_hidden_layers, 28)
        self.assertEqual(self.cfg.num_attention_heads, 28)
        self.assertEqual(self.cfg.num_key_value_heads, 4)
        self.assertEqual(self.cfg.vocab_size, 152064)

    def test_derived_dims(self) -> None:
        self.assertEqual(self.cfg.head_dim, 128)
        self.assertEqual(self.cfg.kv_dim, 512)
        self.assertEqual(self.cfg.num_kv_groups, 7)

    def test_runtime_policy(self) -> None:
        self.assertEqual(self.cfg.dtype, "fp16")
        self.assertEqual(self.cfg.dtype_bytes, 2)
        self.assertFalse(self.cfg.tie_word_embeddings)
        self.assertEqual(self.cfg.max_batch, 1)
        self.assertEqual(self.cfg.max_seq_len, 2048)
        self.assertAlmostEqual(self.cfg.rms_norm_eps, 1e-6)

    def test_model_path_resolved(self) -> None:
        self.assertTrue(self.cfg.model_path.endswith("Qwen2.5-7B-Instruct"))

    def test_layer_order(self) -> None:
        self.assertEqual(self.cfg.layer_order, LAYER_ORDER)

    def test_kv_cache_layout(self) -> None:
        self.assertEqual(self.cfg.kv_cache_layout, "layer_major")

    def test_custom_yaml_path_string(self) -> None:
        cfg = RuntimeConfig.from_yaml(str(CONFIG_7B), project_root=str(PROJECT_ROOT))
        cfg.validate()
        self.assertEqual(cfg.name, self.cfg.name)


class TestConfig05B(unittest.TestCase):
    def test_load_05b_yaml(self) -> None:
        cfg = load_05b()
        self.assertEqual(cfg.name, "qwen2.5-0.5b-instruct")
        self.assertEqual(cfg.hidden_size, 896)
        self.assertEqual(cfg.num_hidden_layers, 24)
        self.assertEqual(cfg.head_dim, 64)
        self.assertTrue(cfg.tie_word_embeddings)


if __name__ == "__main__":
    unittest.main()
