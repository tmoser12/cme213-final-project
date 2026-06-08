"""Tests for Qwen2Executor (Phase 5) — stages 3–4 of parity harness."""

from __future__ import annotations

import unittest

import torch

from runtime.buffers import allocate_buffers
from runtime.executor import Qwen2Executor, _ATTN_HEAD_DIM
from runtime.tests._support import LAYER_ORDER, load_05b, load_7b
from runtime.tests.parity_support import (
    GPU_SKIP,
    HAS_7B_WEIGHTS,
    REQUIRES_GPU,
    default_7b_cfg,
    greedy_decode_executor,
    greedy_decode_hf,
    load_hf_and_executor,
    logits_allclose,
)


class TestExecutorStructure(unittest.TestCase):
    def test_layer_order_matches_support_constant(self) -> None:
        cfg = load_7b()
        self.assertEqual(cfg.layer_order, LAYER_ORDER)

    def test_kernel_set_head_dim_gate(self) -> None:
        # 0.5b head_dim=64 -> draft kernels; forcing kernel_set="target" (expects
        # head_dim 128) must raise. (0.5b with its default kernel_set=draft is now
        # supported — see TestDraftExecutorGpu.)
        cfg = load_05b()
        self.assertEqual(cfg.head_dim, _ATTN_HEAD_DIM["draft"])
        self.assertEqual(cfg.kernel_set, "draft")
        weights = {"model.embed_tokens.weight": torch.empty(1)}
        buffers = allocate_buffers(cfg, batch=1, max_seq_len=8, device="cpu")
        with self.assertRaises(ValueError):
            Qwen2Executor(cfg, weights, buffers, kernel_set="target")


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_7B_WEIGHTS, "7B weights not on disk")
class TestExecutorGpu(unittest.TestCase):
    """Stage 3: full-model prefill/decode logits vs HF."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = default_7b_cfg()
        cls.hf, cls.executor = load_hf_and_executor(cls.cfg, max_seq_len=64)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.hf, cls.executor
        torch.cuda.empty_cache()

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

    def test_greedy_token_matches_hf_argmax(self) -> None:
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        with torch.no_grad():
            prefill_logits = self.executor.prefill(prompt)
            hf_logits = self.hf(prompt).logits
            hf_next = hf_logits[0, -1].argmax()
            our_next = prefill_logits[0, -1].argmax()
        self.assertEqual(our_next.item(), hf_next.item())

    def test_greedy_three_token_trajectory(self) -> None:
        """Lightweight stage-4 check colocated with executor tests."""
        prompt = torch.tensor([[151643, 8948]], device="cuda")
        hf_tokens = greedy_decode_hf(self.hf, prompt, n_new_tokens=3)
        our_tokens = greedy_decode_executor(self.executor, prompt, n_new_tokens=3)
        self.assertEqual(our_tokens, hf_tokens)


if __name__ == "__main__":
    unittest.main()
