"""Rigorous audit of the draft-side bonus-token protocol (matches target_step.py).

The target protocol (target_step.py): each iteration the target accepts n of γ
drafts, samples a bonus t, and DEFERS t (bundled as the leading token of the next
verify). The draft must mirror this: keep the n accepted drafts, commit the bonus,
and continue generating from AFTER the bonus.

This test proves the draft's bookkeeping is exactly right by checking that the
draft's INCREMENTAL KV state — built from decode steps + rollback + bonus commit
across several iterations with varying accept counts — matches a FRESH PREFILL of
the committed token sequence (prompt + accepted drafts + bonuses). If the rollback
or bonus commit were off by even one position, the distributions would diverge.

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_draft_bonus_protocol
"""

from __future__ import annotations

import unittest

import torch

from runtime.core.config import RuntimeConfig, CONFIG_05B
from runtime.core.weights import load_weights_on_gpu
from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.tests.parity_support import logits_allclose

GPU_SKIP = "CUDA not available — run via slurm/run_tests_gpu.sh"
PROMPT_LEN = 16
MAX_SEQ = 128
GAMMA = 4


@unittest.skipIf(not torch.cuda.is_available(), GPU_SKIP)
class TestDraftBonusProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_05B)
        cls.weights, _ = load_weights_on_gpu(cls.cfg, batch=1, device="cuda")
        cls.buffers = allocate_buffers(cls.cfg, batch=1, max_seq_len=MAX_SEQ, device="cuda")
        cls.ex = Qwen2Executor(cls.cfg, cls.weights, cls.buffers)
        torch.manual_seed(0)
        cls.prompt = torch.randint(0, cls.cfg.vocab_size, (1, PROMPT_LEN),
                                   dtype=torch.int64, device="cuda")

    def test_incremental_state_matches_fresh_prefill(self):
        r = DraftRunner(self.ex, seed=0)
        committed = self.prompt[0].tolist()
        r.prefill(self.prompt)
        self.assertEqual(r.prefix_len, PROMPT_LEN)

        # Varying accept counts, including 0 (all rejected) and γ (all accepted).
        for n_accepted in (2, 0, GAMMA, 1):
            ids, _ = r.generate_drafts(GAMMA, greedy=True)
            drafts = ids[0].tolist()
            self.assertEqual(self.ex.cache_pos, len(committed) + GAMMA)

            # Simulate the target's feedback: keep n drafts, append a bonus token.
            bonus = (committed[-1] * 3 + 11) % self.cfg.vocab_size
            committed = committed + drafts[:n_accepted] + [bonus]
            r.apply_target_feedback(n_accepted, bonus)

            # Cursor and prefix length must track the committed sequence exactly.
            self.assertEqual(r.prefix_len, len(committed))
            self.assertEqual(self.ex.cache_pos, len(committed))

        last_incremental = r.last_logits.clone()

        # Gold standard: a fresh prefill of the committed sequence must yield the
        # same next-token distribution as the incrementally-built KV state.
        fresh = self.ex.prefill(torch.tensor([committed], dtype=torch.int64, device="cuda"))
        fresh_last = fresh[0, -1]
        diff = (fresh_last.float() - last_incremental.float()).abs().max().item()
        self.assertTrue(
            logits_allclose(fresh_last, last_incremental),
            msg=f"incremental bonus/rollback KV != fresh prefill of committed seq; max|Δ|={diff:.4f}",
        )

    def test_all_accepted_then_all_rejected(self):
        # Edge sequence: γ accepted (bonus extends), then 0 accepted (bonus replaces
        # the first draft slot). Both must leave a consistent prefix.
        r = DraftRunner(self.ex, seed=1)
        committed = self.prompt[0].tolist()
        r.prefill(self.prompt)
        for n_accepted in (GAMMA, 0):
            ids, _ = r.generate_drafts(GAMMA, greedy=True)
            drafts = ids[0].tolist()
            bonus = (committed[-1] + 5) % self.cfg.vocab_size
            committed = committed + drafts[:n_accepted] + [bonus]
            r.apply_target_feedback(n_accepted, bonus)
            self.assertEqual(r.prefix_len, len(committed))
        fresh = self.ex.prefill(torch.tensor([committed], dtype=torch.int64, device="cuda"))
        self.assertTrue(logits_allclose(fresh[0, -1], r.last_logits))


if __name__ == "__main__":
    unittest.main()
