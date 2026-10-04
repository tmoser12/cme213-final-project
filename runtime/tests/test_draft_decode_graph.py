"""Correctness: CUDA-graph draft decode vs eager (bit-exact trajectory).

The draft (0.5B) reuses the same decode-graph machinery as the target. Replaying
the captured graph must reproduce eager `decode_step` exactly and stay correct as
the position advances. No HF needed — compares two of our own forward paths.

Run: .venv/bin/python -m unittest runtime.tests.test_draft_decode_graph
"""

from __future__ import annotations

import unittest

import torch

from runtime.core.config import RuntimeConfig, CONFIG_05B
from runtime.core.weights import load_weights
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor

GPU_SKIP = "CUDA not available — run on a CUDA GPU after bash demo/setup.sh"
PROMPT_LEN = 24
MAX_SEQ = 128
N_STEPS = 8


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestDraftDecodeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_05B)
        cls.weights = load_weights(cls.cfg, device="cuda")
        cls.buffers = allocate_buffers(cls.cfg, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.ex = Qwen2Executor(cls.cfg, cls.weights, cls.buffers)
        torch.manual_seed(0)
        cls.prompt = torch.randint(0, cls.cfg.vocab_size, (1, PROMPT_LEN),
                                   dtype=torch.int64, device="cuda")

    @classmethod
    def tearDownClass(cls):
        del cls.ex, cls.buffers, cls.weights, cls.prompt
        torch.cuda.empty_cache()

    def _tok(self, t: int) -> torch.Tensor:
        return torch.tensor([t], dtype=torch.int64, device="cuda")

    def test_graph_matches_eager_trajectory(self):
        ex = self.ex
        logits = ex.prefill(self.prompt)
        tok = int(logits[0, -1].argmax().item())
        eager = []
        for _ in range(N_STEPS):
            lg = ex.decode_step(self._tok(tok))
            eager.append(lg.clone())
            tok = int(lg[0, -1].argmax().item())

        logits = ex.prefill(self.prompt)
        tok = int(logits[0, -1].argmax().item())
        for i in range(N_STEPS):
            lg = ex.decode_step_graph(self._tok(tok))
            diff = (lg.float() - eager[i].float()).abs().max().item()
            self.assertTrue(torch.equal(lg, eager[i]),
                            f"draft graph != eager at step {i}; max|Δ|={diff}")
            tok = int(lg[0, -1].argmax().item())

    def test_graph_reused_across_prefills(self):
        ex = self.ex
        for _ in range(2):
            logits = ex.prefill(self.prompt)
            tok = int(logits[0, -1].argmax().item())
            lg = ex.decode_step_graph(self._tok(tok))
            self.assertEqual(tuple(lg.shape), (1, 1, self.cfg.vocab_size))
            self.assertFalse(torch.isnan(lg.float()).any())


if __name__ == "__main__":
    unittest.main()
