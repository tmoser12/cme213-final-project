"""CPU tests for speculative decoding sampler (Phase 8a, dummy logits)."""

from __future__ import annotations

import random
import unittest

import torch

from runtime.speculative.sampler import (
    adjusted_distribution,
    logits_to_probs,
    speculative_acceptance,
)


class TestSpeculativeSamplerCpu(unittest.TestCase):
    def test_identical_p_q_accepts_all_gamma(self) -> None:
        vocab = 32
        gamma = 4
        draft = [3, 7, 11, 19]
        p = torch.randn(gamma + 1, vocab)
        q = p.clone()
        rng = random.Random(0)

        n, bonus = speculative_acceptance(p, q, draft, rng)
        self.assertEqual(n, gamma)
        self.assertGreaterEqual(bonus, 0)
        self.assertLess(bonus, vocab)

    def test_rejects_when_q_prob_negligible_on_draft_token(self) -> None:
        vocab = 8
        draft = [3]
        p = torch.full((2, vocab), -10.0)
        q = torch.full((2, vocab), -10.0)
        p[0, draft[0]] = 10.0
        q[0, draft[0]] = -100.0
        n, _ = speculative_acceptance(p, q, draft, random.Random(0))
        self.assertEqual(n, 0)

    def test_adjusted_distribution_subtracts_q_on_reject(self) -> None:
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.4, 0.4, 0.2])
        adj = adjusted_distribution(p, q, rejected=True)
        self.assertAlmostEqual(adj.sum().item(), 1.0, places=5)
        self.assertTrue((adj >= 0).all())

    def test_reproducible_with_fixed_seed(self) -> None:
        p = torch.randn(5, 64)
        q = torch.randn(5, 64)
        draft = [1, 5, 9, 12]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        self.assertEqual(
            speculative_acceptance(p, q, draft, rng1),
            speculative_acceptance(p, q, draft, rng2),
        )

    def test_logits_to_probs_rows_sum_to_one(self) -> None:
        probs = logits_to_probs(torch.randn(4, 128))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
