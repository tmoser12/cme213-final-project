"""Setup tests: memory planning from RuntimeConfig."""

import unittest

from runtime.core import shapes
from runtime.core.memory import max_seq_len_for_budget, plan_memory, runtime_bytes_per_seq
from runtime.tests._support import load_7b


class TestMemoryPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_7b()

    def test_plan_uses_requested_batch_and_seq(self) -> None:
        plan = plan_memory(self.cfg, batch=1, max_seq_len=512)
        self.assertEqual(plan["batch"], 1)
        self.assertEqual(plan["max_seq_len"], 512)

    def test_plan_defaults_from_config(self) -> None:
        plan = plan_memory(self.cfg)
        self.assertEqual(plan["batch"], self.cfg.max_batch)
        self.assertEqual(plan["max_seq_len"], self.cfg.max_seq_len)

    def test_weight_bytes_in_expected_range(self) -> None:
        plan = plan_memory(self.cfg, batch=1, max_seq_len=512)
        mib = plan["weight_bytes"] / (1024 * 1024)
        self.assertGreater(mib, 14_000)
        self.assertLess(mib, 15_000)

    def test_buffer_byte_counts(self) -> None:
        plan = plan_memory(self.cfg, batch=1, max_seq_len=512)
        hidden_bytes = 1 * 512 * self.cfg.hidden_size * self.cfg.dtype_bytes
        self.assertEqual(plan["buffer_bytes"]["hidden_ping"], hidden_bytes)
        self.assertEqual(plan["buffer_bytes"]["hidden_pong"], hidden_bytes)
        self.assertIn("kv_cache_keys", plan["buffer_bytes"])
        self.assertIn("logits", plan["buffer_bytes"])

    def test_runtime_bytes_sum(self) -> None:
        plan = plan_memory(self.cfg, batch=1, max_seq_len=128)
        self.assertEqual(
            plan["runtime_bytes"],
            plan["activation_bytes"] + plan["kv_cache_bytes"],
        )
        self.assertGreater(plan["activation_bytes"], 0)
        self.assertGreater(plan["kv_cache_bytes"], 0)

    def test_runtime_scales_linearly_with_seq(self) -> None:
        per_seq = runtime_bytes_per_seq(self.cfg, batch=1)
        plan_64 = plan_memory(self.cfg, batch=1, max_seq_len=64)
        plan_128 = plan_memory(self.cfg, batch=1, max_seq_len=128)
        fixed = shapes.cache_position_bytes()
        self.assertEqual(plan_64["runtime_bytes"], per_seq * 64 + fixed)
        self.assertEqual(plan_128["runtime_bytes"], per_seq * 128 + fixed)

    def test_max_seq_len_for_budget(self) -> None:
        budget = int(7.5 * 1024**3)
        max_s = max_seq_len_for_budget(self.cfg, budget, batch=1)
        self.assertGreater(max_s, 10_000)
        self.assertLessEqual(max_s, self.cfg.max_position_embeddings)


if __name__ == "__main__":
    unittest.main()
