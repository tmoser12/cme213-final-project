"""Stage 4 parity: multi-token greedy decode trajectory vs HF (Phase 6)."""

from __future__ import annotations

import unittest

import torch

from runtime.tests.parity_support import (
    GPU_SKIP,
    HAS_7B_WEIGHTS,
    REQUIRES_GPU,
    default_7b_cfg,
    greedy_decode_executor,
    greedy_decode_hf,
    load_hf_and_executor,
)


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_7B_WEIGHTS, "7B weights not on disk")
class TestGreedyDecodeTrajectory(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = default_7b_cfg()
        cls.hf, cls.executor = load_hf_and_executor(cls.cfg, max_seq_len=128)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.hf, cls.executor
        torch.cuda.empty_cache()

    def _assert_trajectory(self, prompt: list[int], n_new: int) -> None:
        prompt_ids = torch.tensor([prompt], dtype=torch.int64, device="cuda")
        hf_tokens = greedy_decode_hf(self.hf, prompt_ids, n_new)
        our_tokens = greedy_decode_executor(self.executor, prompt_ids, n_new)
        self.assertEqual(
            our_tokens,
            hf_tokens,
            msg=f"first mismatch at step {next((i for i, (a, b) in enumerate(zip(our_tokens, hf_tokens)) if a != b), len(hf_tokens))}",
        )

    def test_eight_tokens_from_short_prompt(self) -> None:
        self._assert_trajectory([151643, 8948, 198], n_new=8)

    def test_sixteen_tokens_from_minimal_prompt(self) -> None:
        self._assert_trajectory([151643], n_new=16)


if __name__ == "__main__":
    unittest.main()
