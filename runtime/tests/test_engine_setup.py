"""Setup tests: config + memory plan form a valid engine-ready configuration."""

import unittest

from runtime.core.memory import plan_memory
from runtime.tests._support import load_7b


class TestEngineSetup(unittest.TestCase):
    """Checks the inputs an inference engine needs before any GPU kernels run."""

    def test_7b_default_setup(self) -> None:
        cfg = load_7b()
        plan = plan_memory(cfg, batch=1, max_seq_len=cfg.max_seq_len)

        self.assertEqual(plan["batch"], 1)
        self.assertEqual(plan["max_seq_len"], 2048)
        self.assertGreater(plan["weight_bytes"], 0)
        self.assertGreater(plan["runtime_bytes"], 0)
        self.assertEqual(len(cfg.layer_order), 6)

    def test_custom_batch_and_seq_override(self) -> None:
        cfg = load_7b()
        plan = plan_memory(cfg, batch=2, max_seq_len=256)

        self.assertEqual(plan["batch"], 2)
        self.assertEqual(plan["max_seq_len"], 256)
        expected_hidden = 2 * 256 * cfg.hidden_size * cfg.dtype_bytes
        self.assertEqual(plan["buffer_bytes"]["hidden_ping"], expected_hidden)

    def test_all_required_buffers_present(self) -> None:
        cfg = load_7b()
        plan = plan_memory(cfg, batch=1, max_seq_len=128)
        required = (
            "hidden_ping",
            "hidden_pong",
            "q_states",
            "k_states",
            "v_states",
            "mlp_gate",
            "logits",
            "kv_cache_keys",
            "kv_cache_values",
        )
        for name in required:
            with self.subTest(buffer=name):
                self.assertIn(name, plan["buffers"])
                self.assertIn(name, plan["buffer_bytes"])
                self.assertGreater(plan["buffer_bytes"][name], 0)

    def test_fp16_dtype_policy(self) -> None:
        cfg = load_7b()
        self.assertEqual(cfg.dtype, "fp16")
        self.assertEqual(cfg.accum_dtype, "fp32")
        self.assertEqual(cfg.cuda_arch, "sm_75")


if __name__ == "__main__":
    unittest.main()
