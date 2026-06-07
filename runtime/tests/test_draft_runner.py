"""Phase D5: DraftRunner API mechanics (shapes, KV cursor, rollback, greedy).

No HF needed — exercises generate_drafts / apply_target_feedback on the 0.5B draft
executor and checks the bookkeeping the speculative loop relies on.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_draft_runner
"""

from __future__ import annotations

import unittest

import torch

from runtime.core.config import RuntimeConfig, CONFIG_05B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner

GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"
PROMPT_LEN = 16
MAX_SEQ = 128
GAMMA = 4


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestDraftRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_05B)
        cls.weights, _ = load_weights_on_gpu(cls.cfg, batch=1, device="cuda")
        cls.buffers = allocate_buffers(cls.cfg, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.ex = Qwen2Executor(cls.cfg, cls.weights, cls.buffers)
        torch.manual_seed(0)
        cls.prompt = torch.randint(0, cls.cfg.vocab_size, (1, PROMPT_LEN),
                                   dtype=torch.int64, device="cuda")

    def _runner(self, seed=0):
        return DraftRunner(self.ex, seed=seed)

    def test_generate_drafts_shapes_and_cursor(self):
        r = self._runner()
        r.prefill(self.prompt)
        self.assertEqual(r.prefix_len, PROMPT_LEN)
        ids, q = r.generate_drafts(GAMMA)
        self.assertEqual(tuple(ids.shape), (1, GAMMA))
        self.assertEqual(tuple(q.shape), (GAMMA + 1, self.cfg.vocab_size))
        self.assertEqual(ids.dtype, torch.int64)
        # γ drafts committed to the draft KV.
        self.assertEqual(self.ex.cache_pos, PROMPT_LEN + GAMMA)

    def test_apply_feedback_rolls_back_and_commits_bonus(self):
        r = self._runner()
        r.prefill(self.prompt)
        r.generate_drafts(GAMMA)
        n_accepted, bonus = 2, 123
        r.apply_target_feedback(n_accepted, bonus)
        # cursor = prefix + n_accepted + 1 (accepted drafts + bonus)
        self.assertEqual(self.ex.cache_pos, PROMPT_LEN + n_accepted + 1)
        self.assertEqual(r.prefix_len, PROMPT_LEN + n_accepted + 1)
        self.assertIsNotNone(r.last_logits)
        self.assertEqual(tuple(r.last_logits.shape), (self.cfg.vocab_size,))

    def test_greedy_drafts_match_greedy_extend(self):
        # generate_drafts(greedy=True) proposes the model's argmax trajectory,
        # which must equal the first γ tokens of greedy_extend from the same prompt.
        r = self._runner()
        r.prefill(self.prompt)
        ids, _ = r.generate_drafts(GAMMA, greedy=True)
        draft_list = ids[0].tolist()

        full = self.ex.greedy_extend(self.prompt, GAMMA)  # re-prefills internally
        ref = full[0, PROMPT_LEN:PROMPT_LEN + GAMMA].tolist()
        self.assertEqual(draft_list, ref)

    def test_multi_iteration_positions(self):
        r = self._runner()
        r.prefill(self.prompt)
        pos = PROMPT_LEN
        for n_accepted in (4, 1, 3):  # γ=4 each iter
            r.generate_drafts(GAMMA)
            self.assertEqual(self.ex.cache_pos, pos + GAMMA)
            r.apply_target_feedback(n_accepted, 50)
            pos = pos + n_accepted + 1
            self.assertEqual(r.prefix_len, pos)
            self.assertEqual(self.ex.cache_pos, pos)

    def test_bonus_outside_draft_vocab_raises(self):
        r = self._runner()
        r.prefill(self.prompt)
        r.generate_drafts(GAMMA)
        with self.assertRaises(ValueError):
            r.apply_target_feedback(2, self.cfg.vocab_size + 5)


if __name__ == "__main__":
    unittest.main()
