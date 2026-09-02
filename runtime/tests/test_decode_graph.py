"""Phase 4/5 correctness: CUDA-graph decode vs eager decode (bit-exact trajectory).

`decode_step_graph` captures the static S=1 decode forward once and replays it per
token. Replaying must reproduce eager `decode_step` exactly, and must stay correct
as the position advances (the device-scalar positions are the whole point). We check
a multi-step greedy trajectory for bit-exact logits.

No HF needed — compares two of our own forward paths.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_decode_graph
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
N_STEPS = 8


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestDecodeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_7B)
        cls.weights, _ = load_weights_on_gpu(cls.cfg, batch=1, device="cuda")
        cls.buffers = allocate_buffers(cls.cfg, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.ex = Qwen2Executor(cls.cfg, cls.weights, cls.buffers)
        torch.manual_seed(0)
        cls.prompt = torch.randint(0, cls.cfg.vocab_size, (1, PROMPT_LEN),
                                   dtype=torch.int64, device="cuda")

    def _tok(self, t: int) -> torch.Tensor:
        return torch.tensor([t], dtype=torch.int64, device="cuda")

    def test_graph_matches_eager_trajectory(self):
        ex = self.ex

        # Eager reference trajectory.
        logits = ex.prefill(self.prompt)
        tok = int(logits[0, -1].argmax().item())
        eager = []
        for _ in range(N_STEPS):
            lg = ex.decode_step(self._tok(tok))
            eager.append(lg.clone())
            tok = int(lg[0, -1].argmax().item())

        # Graph trajectory from the identical post-prefill state. First call
        # captures (warmup + record), then every step is a single replay().
        logits = ex.prefill(self.prompt)
        tok = int(logits[0, -1].argmax().item())
        for i in range(N_STEPS):
            lg = ex.decode_step_graph(self._tok(tok))
            diff = (lg.float() - eager[i].float()).abs().max().item()
            self.assertTrue(torch.equal(lg, eager[i]),
                            f"graph != eager at step {i}; max|Δ|={diff}")
            tok = int(lg[0, -1].argmax().item())

    def test_graph_reused_across_prefills(self):
        # The captured graph must keep working after a reset + new prefill
        # (KV/buffer addresses are stable; positions come from device scalars).
        ex = self.ex
        for _ in range(2):
            logits = ex.prefill(self.prompt)
            tok = int(logits[0, -1].argmax().item())
            lg = ex.decode_step_graph(self._tok(tok))
            self.assertEqual(tuple(lg.shape), (1, 1, self.cfg.vocab_size))
            self.assertFalse(torch.isnan(lg.float()).any())


if __name__ == "__main__":
    unittest.main()
