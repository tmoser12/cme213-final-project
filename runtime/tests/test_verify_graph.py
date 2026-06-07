"""Phase 6 correctness: CUDA-graph VERIFY vs eager verify_gamma (bit-exact).

`verify_gamma_graph` captures one graph per query length S (= γ without a leading
bonus, γ+1 with one) and replays it. Replay must reproduce eager `verify_gamma`
exactly. We check both the first-iter (no bonus, S=γ) and later-iter (bonus, S=γ+1)
paths, and that a second γ captures its own graph.

No HF needed — compares two of our own forward paths. Note: 7B verify is
compute-dense, so this is a correctness/availability feature, not a speedup.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_verify_graph
"""

from __future__ import annotations

import unittest

import torch

from runtime.core.config import RuntimeConfig, CONFIG_7B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor

GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"
PROMPT_LEN = 24
MAX_SEQ = 128


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestVerifyGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B)
        cls.weights, _ = load_weights_on_gpu(cls.cfg, batch=1, device="cuda")
        cls.buffers = allocate_buffers(cls.cfg, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.ex = Qwen2Executor(cls.cfg, cls.weights, cls.buffers)
        torch.manual_seed(0)
        cls.prompt = torch.randint(0, cls.cfg.vocab_size, (1, PROMPT_LEN),
                                   dtype=torch.int64, device="cuda")

    def _drafts(self, gamma: int) -> torch.Tensor:
        return torch.randint(0, self.cfg.vocab_size, (1, gamma),
                             dtype=torch.int64, device="cuda")

    def _check(self, gamma: int, leading_bonus):
        ex = self.ex
        drafts = self._drafts(gamma)
        ex.prefill(self.prompt)
        eager = ex.verify_gamma(drafts, leading_bonus=leading_bonus).clone()
        ex.prefill(self.prompt)
        graphed = ex.verify_gamma_graph(drafts, leading_bonus=leading_bonus)
        S = gamma + (1 if leading_bonus is not None else 0)
        self.assertEqual(tuple(graphed.shape), (1, S, self.cfg.vocab_size))
        diff = (graphed.float() - eager.float()).abs().max().item()
        self.assertTrue(torch.equal(graphed, eager),
                        f"verify graph != eager (γ={gamma}, bonus={leading_bonus}); max|Δ|={diff}")

    def test_no_bonus_first_iter(self):
        self._check(gamma=4, leading_bonus=None)        # S = γ = 4

    def test_with_leading_bonus(self):
        self._check(gamma=4, leading_bonus=777)         # S = γ+1 = 5

    def test_second_gamma_gets_its_own_graph(self):
        self._check(gamma=2, leading_bonus=None)        # S = 2, distinct graph
        self._check(gamma=4, leading_bonus=None)        # S = 4 reused from earlier
        self.assertIn(2, self.ex._verify_state)
        self.assertIn(4, self.ex._verify_state)


if __name__ == "__main__":
    unittest.main()
