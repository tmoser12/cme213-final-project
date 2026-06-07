"""Phase D3 parity: draft Qwen2Executor (kernel_set=draft) vs HF Qwen2.5-0.5B.

Confirms the generalized executor runs the 0.5B model (head_dim=64, draft kernels)
correctly: prefill + decode logits match HF, and the greedy trajectory matches
HF generate(do_sample=False).

Run: bash slurm/run_tests_gpu.sh runtime.tests.test_draft_executor
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch

from runtime.core.config import CONFIG_05B, RuntimeConfig
from runtime.tests._support import PROJECT_ROOT
from runtime.tests.parity_support import (
    GPU_SKIP,
    REQUIRES_GPU,
    greedy_decode_executor,
    greedy_decode_hf,
    load_hf_and_executor,
    logits_allclose,
)

HAS_05B_WEIGHTS = os.path.isdir(Path(PROJECT_ROOT) / "models/Qwen2.5-0.5B-Instruct")


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_05B_WEIGHTS, "0.5B weights not on disk")
class TestDraftExecutorGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
        cls.hf, cls.executor = load_hf_and_executor(cls.cfg, max_seq_len=64)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.hf, cls.executor
        torch.cuda.empty_cache()

    def test_executor_uses_draft_kernels(self) -> None:
        self.assertEqual(self.executor.kernel_set, "draft")
        self.assertEqual(self.cfg.head_dim, 64)

    def test_prefill_logits_match_hf(self) -> None:
        input_ids = torch.tensor([[151643, 8948, 198, 2610]], device="cuda")
        with torch.no_grad():
            hf_logits = self.hf(input_ids).logits
            actual = self.executor.prefill(input_ids)
        self.assertEqual(tuple(actual.shape), tuple(hf_logits.shape))
        self.assertTrue(
            logits_allclose(hf_logits, actual),
            msg=f"max diff {(hf_logits.float() - actual.float()).abs().max().item():.4f}",
        )
        self.assertEqual(self.executor._cache_pos, input_ids.shape[1])

    def test_decode_step_after_prefill(self) -> None:
        prompt = torch.tensor([[151643, 8948, 198]], device="cuda")
        with torch.no_grad():
            prefill_logits = self.executor.prefill(prompt)
            next_id = prefill_logits[:, -1].argmax(dim=-1)
            full = torch.cat([prompt, next_id.unsqueeze(0)], dim=1)
            hf_logits = self.hf(full).logits[:, -1:, :]
            actual = self.executor.decode_step(next_id)
        self.assertEqual(tuple(actual.shape), (1, 1, self.cfg.vocab_size))
        self.assertTrue(
            logits_allclose(hf_logits, actual),
            msg=f"max diff {(hf_logits.float() - actual.float()).abs().max().item():.4f}",
        )

    def test_greedy_trajectory_matches_hf(self) -> None:
        prompt = torch.tensor([[151643, 8948, 198]], device="cuda")
        hf_tokens = greedy_decode_hf(self.hf, prompt, n_new_tokens=4)
        our_tokens = greedy_decode_executor(self.executor, prompt, n_new_tokens=4)
        self.assertEqual(our_tokens, hf_tokens)


if __name__ == "__main__":
    unittest.main()
