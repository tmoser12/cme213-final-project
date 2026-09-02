"""Tests for shape helpers and memory planning."""

import os
from pathlib import Path
import unittest

from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.memory import plan_memory
from runtime.core import shapes


class TestShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = os.environ.get(
            "PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
        )
        cls.cfg_7b = RuntimeConfig.from_yaml(CONFIG_7B, project_root=root)
        cls.cfg_05b = RuntimeConfig.from_yaml(CONFIG_05B, project_root=root)

    def test_7b_activation_shapes(self) -> None:
        cfg = self.cfg_7b
        self.assertEqual(shapes.hidden(2, 128, cfg), (2, 128, 3584))
        self.assertEqual(shapes.q_states(2, 128, cfg), (2, 28, 128, 128))
        self.assertEqual(shapes.kv_states(2, 128, cfg), (2, 4, 128, 128))
        self.assertEqual(shapes.kv_cache(1, 2048, cfg), (28, 1, 4, 2048, 128))

    def test_05b_activation_shapes(self) -> None:
        cfg = self.cfg_05b
        self.assertEqual(shapes.hidden(1, 64, cfg), (1, 64, 896))
        self.assertEqual(shapes.q_states(1, 64, cfg), (1, 14, 64, 64))

    def test_7b_weight_shapes(self) -> None:
        w = shapes.weight_shapes(self.cfg_7b)
        self.assertEqual(w["q_proj"], (3584, 3584))
        self.assertEqual(w["k_proj"], (512, 3584))
        self.assertEqual(w["gate_proj"], (18944, 3584))

    def test_weight_bytes_7b(self) -> None:
        mib = shapes.total_weight_bytes(self.cfg_7b) / (1024 * 1024)
        self.assertGreater(mib, 14000)
        self.assertLess(mib, 15000)

    def test_weight_bytes_05b(self) -> None:
        mib = shapes.total_weight_bytes(self.cfg_05b) / (1024 * 1024)
        self.assertGreater(mib, 900)
        self.assertLess(mib, 1100)

    def test_plan_memory_both_models(self) -> None:
        for cfg in (self.cfg_7b, self.cfg_05b):
            plan = plan_memory(cfg, batch=1, max_seq_len=512)
            self.assertIn("hidden_ping", plan["buffers"])
            self.assertIn("kv_cache_keys", plan["buffers"])
            self.assertGreater(plan["weight_bytes"], 0)
            self.assertGreater(plan["kv_cache_bytes"], 0)
            self.assertEqual(
                plan["buffer_bytes"]["hidden_ping"],
                shapes.nbytes(shapes.hidden(1, 512, cfg), cfg),
            )


if __name__ == "__main__":
    unittest.main()
