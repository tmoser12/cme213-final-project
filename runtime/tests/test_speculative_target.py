"""GPU tests for target speculative decoding (Phase 8a, no MPI)."""

from __future__ import annotations

import random
import unittest

import torch

from runtime.speculative.target_step import flush_pending_bonus, target_speculative_step
from runtime.speculative.types import MAX_VERIFY_GAMMA
from runtime.tests.parity_support import (
    GPU_SKIP,
    HAS_7B_WEIGHTS,
    REQUIRES_GPU,
    default_7b_cfg,
    load_hf_and_executor,
    logits_allclose,
)


def _dummy_q_logits_from_target(p_all: torch.Tensor) -> torch.Tensor:
    """Dummy draft logits identical to target — all γ tokens should accept."""
    return p_all.clone()


def _build_matching_q(
    executor,
    draft: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Build q=p logits for a step without consuming deferred bonus."""
    leading_bonus = executor.pending_bonus_token
    with torch.no_grad():
        if leading_bonus is not None:
            p_verify = executor.verify_gamma(draft, leading_bonus=leading_bonus)
            p_all = p_verify[0]
        else:
            p_verify = executor.verify_gamma(draft)
            p_all = torch.cat([executor.p1_logits.unsqueeze(0), p_verify[0]], dim=0)
        executor.rollback_cache(prefix_len)
    return _dummy_q_logits_from_target(p_all.cpu())


def _run_step_with_matching_q(
    executor,
    draft: torch.Tensor,
    seed: int,
):
    """Run one step using q=p logits (requires a throwaway verify to build q)."""
    prefix_len = executor.cache_pos
    draft_q = _build_matching_q(executor, draft, prefix_len)
    return target_speculative_step(
        executor,
        draft,
        draft_q,
        random.Random(seed),
    )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_7B_WEIGHTS, "7B weights not on disk")
class TestVerifyGammaGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = default_7b_cfg()
        cls.hf, cls.executor = load_hf_and_executor(cls.cfg, max_seq_len=64)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.hf, cls.executor
        torch.cuda.empty_cache()

    def test_verify_logits_match_hf(self) -> None:
        prompt = torch.tensor([[151643, 8948, 198]], device="cuda")
        draft = torch.tensor([[2610, 525, 374]], device="cuda")
        gamma = draft.shape[1]
        prefix_len = prompt.shape[1]

        with torch.no_grad():
            prefill_logits = self.executor.prefill(prompt)
            p1 = prefill_logits[0, -1, :]
            self.assertTrue(torch.equal(p1, self.executor.p1_logits))

            pos_before = self.executor.cache_pos
            p_verify = self.executor.verify_gamma(draft)
            self.assertEqual(self.executor.cache_pos, pos_before + gamma)

            full = torch.cat([prompt, draft], dim=1)
            hf_logits = self.hf(full).logits.float()
            hf_p1 = hf_logits[0, prefix_len - 1, :]
            self.assertTrue(logits_allclose(hf_p1.unsqueeze(0), p1.unsqueeze(0)))

            for i in range(gamma):
                hf_row = hf_logits[0, prefix_len + i, :]
                our_row = p_verify[0, i, :]
                self.assertTrue(
                    logits_allclose(hf_row.unsqueeze(0), our_row.unsqueeze(0)),
                    msg=f"verify row {i} (p_{i + 2})",
                )

    def test_rollback_cache_restores_cursor(self) -> None:
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        draft = torch.tensor([[198, 2610, 525]], device="cuda")
        with torch.no_grad():
            self.executor.prefill(prompt)
            prefix_len = self.executor.cache_pos
            self.executor.verify_gamma(draft)
            self.assertEqual(self.executor.cache_pos, prefix_len + draft.shape[1])
            self.executor.rollback_cache(prefix_len + 1)
            self.assertEqual(self.executor.cache_pos, prefix_len + 1)

    def test_verify_with_commit_bonus_matches_hf(self) -> None:
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        bonus = 198
        draft = torch.tensor([[2610, 525]], device="cuda")
        gamma = draft.shape[1]
        prefix_len = prompt.shape[1]

        with torch.no_grad():
            self.executor.prefill(prompt)
            self.executor.defer_bonus_token(bonus)
            pos_before = self.executor.cache_pos

            p_verify = self.executor.verify_gamma(draft, leading_bonus=bonus)
            self.assertEqual(p_verify.shape[1], gamma + 1)
            self.assertEqual(self.executor.cache_pos, pos_before + gamma + 1)

            full = torch.cat(
                [prompt, torch.tensor([[bonus]], device="cuda"), draft],
                dim=1,
            )
            hf_logits = self.hf(full).logits.float()
            for i in range(gamma + 1):
                hf_row = hf_logits[0, prefix_len + i, :]
                our_row = p_verify[0, i, :]
                self.assertTrue(
                    logits_allclose(hf_row.unsqueeze(0), our_row.unsqueeze(0)),
                    msg=f"bundled verify row {i} (p_{i + 1})",
                )


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_7B_WEIGHTS, "7B weights not on disk")
class TestTargetSpeculativeStepGpu(unittest.TestCase):
    """End-to-end target step with dummy draft logits (no MPI)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = default_7b_cfg()
        _, cls.executor = load_hf_and_executor(cls.cfg, max_seq_len=64)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.executor
        torch.cuda.empty_cache()

    def test_step_with_dummy_q_accepts_all(self) -> None:
        prompt = torch.tensor([[151643, 8948, 198]], device="cuda")
        draft = torch.tensor([[2610, 525]], device="cuda")
        gamma = draft.shape[1]

        with torch.no_grad():
            self.executor.prefill(prompt)
            prefix_len = self.executor.cache_pos
            result = _run_step_with_matching_q(self.executor, draft, seed=123)

        self.assertEqual(result.n_accepted, gamma)
        self.assertEqual(result.prefix_len, prefix_len)
        self.assertEqual(result.new_prefix_len, prefix_len + gamma + 1)
        self.assertEqual(result.cache_pos_after, prefix_len + gamma)
        self.assertEqual(self.executor.cache_pos, result.cache_pos_after)
        self.assertEqual(self.executor.pending_bonus_token, result.bonus_token)

        flush_pending_bonus(self.executor)
        self.assertEqual(self.executor.cache_pos, result.new_prefix_len)
        self.assertIsNone(self.executor.pending_bonus_token)

    def test_first_iter_has_no_leading_bonus(self) -> None:
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        draft = torch.tensor([[198, 2610]], device="cuda")

        with torch.no_grad():
            self.executor.prefill(prompt)
            result = _run_step_with_matching_q(self.executor, draft, seed=1)

        self.assertFalse(result.had_leading_bonus)
        self.assertIsNotNone(self.executor.pending_bonus_token)

    def test_second_iter_bundles_leading_bonus(self) -> None:
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        draft_a = torch.tensor([[198, 2610]], device="cuda")
        draft_b = torch.tensor([[525, 374]], device="cuda")

        with torch.no_grad():
            self.executor.prefill(prompt)
            _run_step_with_matching_q(self.executor, draft_a, seed=2)
            result_b = _run_step_with_matching_q(self.executor, draft_b, seed=3)

        self.assertTrue(result_b.had_leading_bonus)

    def test_deferred_bonus_bundled_into_next_verify(self) -> None:
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        draft_a = torch.tensor([[198, 2610]], device="cuda")
        draft_b = torch.tensor([[525, 374]], device="cuda")

        with torch.no_grad():
            self.executor.prefill(prompt)
            result_a = _run_step_with_matching_q(self.executor, draft_a, seed=7)

            self.assertEqual(self.executor.cache_pos, result_a.cache_pos_after)
            self.assertEqual(self.executor.pending_bonus_token, result_a.bonus_token)

            pos_before_b = self.executor.cache_pos
            result_b = _run_step_with_matching_q(self.executor, draft_b, seed=8)

        self.assertTrue(result_b.had_leading_bonus)
        self.assertGreater(self.executor.cache_pos, pos_before_b)
        self.assertEqual(self.executor.pending_bonus_token, result_b.bonus_token)
        self.assertEqual(self.executor.cache_pos, result_b.cache_pos_after)

        flush_pending_bonus(self.executor)
        self.assertIsNone(self.executor.pending_bonus_token)
        self.assertEqual(self.executor.cache_pos, result_b.new_prefix_len)

    def test_step_gamma_within_kernel_limit(self) -> None:
        self.assertGreaterEqual(MAX_VERIFY_GAMMA, 1)
        prompt = torch.tensor([[151643]], device="cuda")
        draft = torch.tensor([[8948, 198, 2610, 525]], device="cuda")
        with torch.no_grad():
            self.executor.prefill(prompt)
            p_verify = self.executor.verify_gamma(draft)
        self.assertEqual(p_verify.shape[1], draft.shape[1])


class TestNoMpiInPhase8a(unittest.TestCase):
    """Sanity: Phase 8a modules must not import mpi4py."""

    def test_speculative_package_has_no_mpi_import(self) -> None:
        import importlib
        import pkgutil
        import runtime.speculative as spec_pkg

        for mod_info in pkgutil.walk_packages(spec_pkg.__path__, spec_pkg.__name__ + "."):
            mod = importlib.import_module(mod_info.name)
            source_path = getattr(mod, "__file__", "") or ""
            if source_path.endswith(".py"):
                with open(source_path, encoding="utf-8") as f:
                    text = f.read()
                self.assertNotIn("mpi4py", text, msg=f"unexpected mpi4py in {mod_info.name}")
                self.assertNotIn("from mpi4py", text, msg=f"unexpected mpi import in {mod_info.name}")


if __name__ == "__main__":
    unittest.main()
