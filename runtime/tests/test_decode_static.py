"""Phase 3 parity: static/`_dev` decode forward vs eager decode_step (no graph).

`decode_step_static` runs the S=1 decode through static buffers + device-scalar
attention ops — the exact region we capture into a CUDA graph in Phase 4. Eager,
it must be numerically identical to `decode_step`. We check a multi-step greedy
trajectory (catches position-advance bugs) for bit-exact logits.

No HF needed — this compares two of our own forward paths.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_decode_static
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
N_STEPS = 6


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestDecodeStatic(unittest.TestCase):
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

    def _eager_trajectory(self):
        logits = self.ex.prefill(self.prompt)
        tok = int(logits[0, -1].argmax().item())
        out = []
        for _ in range(N_STEPS):
            lg = self.ex.decode_step(self._tok(tok))
            out.append(lg.clone())
            tok = int(lg[0, -1].argmax().item())
        return out

    def test_static_matches_eager_trajectory(self):
        eager = self._eager_trajectory()

        # Re-prefill resets the KV cache + position to the identical post-prompt
        # state, so the static run starts from exactly where eager did.
        logits = self.ex.prefill(self.prompt)
        tok = int(logits[0, -1].argmax().item())
        for i in range(N_STEPS):
            lg = self.ex.decode_step_static(self._tok(tok))
            diff = (lg.float() - eager[i].float()).abs().max().item()
            self.assertTrue(torch.equal(lg, eager[i]),
                            f"static != eager at step {i}; max|Δ|={diff}")
            tok = int(lg[0, -1].argmax().item())

    def test_static_buffers_present(self):
        b = self.buffers
        self.assertEqual(tuple(b.static_input_ids.shape), (1, 1))
        self.assertEqual(b.static_input_ids.dtype, torch.int64)
        self.assertEqual(b.static_cur_len.shape, ())
        # RoPE is gathered in-graph from rope_arange + cache_position (no static_cos/sin).
        self.assertEqual(b.rope_arange.dtype, torch.int64)
        self.assertGreaterEqual(b.rope_arange.numel(), 8)


if __name__ == "__main__":
    unittest.main()
