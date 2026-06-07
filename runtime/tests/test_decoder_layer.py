"""Stage 2 parity: single composed decoder layer vs HF (Phase 6).

Runs for both the 7B target and 0.5B draft model (config-driven executor).
"""

from __future__ import annotations

import unittest

import torch

from runtime.tests.parity_support import (
    GPU_SKIP,
    HAS_05B_WEIGHTS,
    HAS_7B_WEIGHTS,
    REQUIRES_GPU,
    capture_decoder_layer,
    default_05b_cfg,
    default_7b_cfg,
    hidden_allclose,
    load_hf_and_executor,
    reset_layer_kv,
)


class _DecoderLayerParityBase:
    """One HF model load; hook captures layer I/O for kernel-composed comparison."""

    CFG_LOADER: staticmethod

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = cls.CFG_LOADER()
        cls.hf, cls.executor = load_hf_and_executor(cls.cfg, max_seq_len=64)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.hf, cls.executor
        torch.cuda.empty_cache()

    def _assert_layer(self, layer_idx: int, input_ids: torch.Tensor) -> None:
        seq_len = input_ids.shape[1]
        hf_in, hf_out = capture_decoder_layer(self.hf, input_ids, layer_idx)
        reset_layer_kv(self.executor, layer_idx)
        with torch.no_grad():
            actual = self.executor.run_decoder_layer(hf_in, layer_idx, seq_len, decode=False)
        self.assertTrue(
            hidden_allclose(hf_out, actual),
            msg=(
                f"layer={layer_idx} seq={seq_len} "
                f"max_diff={(hf_out.float() - actual.float()).abs().max().item():.4f}"
            ),
        )

    def test_layer0_short_prompt(self) -> None:
        ids = torch.tensor([[151643, 8948, 198]], device="cuda")
        self._assert_layer(0, ids)

    def test_layer0_longer_prompt(self) -> None:
        ids = torch.tensor([[151643, 8948, 198, 2610, 525]], device="cuda")
        self._assert_layer(0, ids)

    def test_mid_layer(self) -> None:
        ids = torch.tensor([[151643, 8948, 198, 2610]], device="cuda")
        self._assert_layer(self.cfg.num_hidden_layers // 2, ids)

    def test_last_layer(self) -> None:
        ids = torch.tensor([[151643, 8948, 198]], device="cuda")
        self._assert_layer(self.cfg.num_hidden_layers - 1, ids)


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_7B_WEIGHTS, "7B weights not on disk")
class TestDecoderLayerParity(_DecoderLayerParityBase, unittest.TestCase):
    CFG_LOADER = staticmethod(default_7b_cfg)


@unittest.skipIf(REQUIRES_GPU, GPU_SKIP)
@unittest.skipUnless(HAS_05B_WEIGHTS, "0.5B weights not on disk")
class TestDecoderLayer05BParity(_DecoderLayerParityBase, unittest.TestCase):
    CFG_LOADER = staticmethod(default_05b_cfg)


if __name__ == "__main__":
    unittest.main()
