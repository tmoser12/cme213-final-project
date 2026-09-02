"""Wire-format roundtrip for the MPI speculative protocol (no MPI, no models)."""

from __future__ import annotations

import unittest

import numpy as np

from runtime.speculative.mpi_protocol import (
    DraftPayload,
    TargetResult,
    draft_logits_byte_count,
    pack_draft_payload,
    pack_target_result,
    unpack_target_result,
)


class TestMpiProtocol(unittest.TestCase):
    def test_target_result_roundtrip(self):
        r = TargetResult(n_accepted=3, bonus_token=151000, prefix_len=42, cache_pos_after=45)
        self.assertEqual(unpack_target_result(pack_target_result(r)), r)

    def test_draft_logits_uint8_roundtrip(self):
        gamma, vocab = 4, 257
        logits = (np.random.randn(gamma + 1, vocab) * 5).astype(np.float16)
        ids, packed = pack_draft_payload(DraftPayload([10, 20, 30, 40], logits))
        # Simulate the uint8-view wire transfer used by send/recv.
        wire = packed.view(np.uint8)
        self.assertEqual(wire.nbytes, draft_logits_byte_count(gamma, vocab))
        back = wire.view(np.float16).reshape(gamma + 1, vocab)
        self.assertTrue(np.array_equal(back, logits))
        self.assertEqual(ids.tolist(), [10, 20, 30, 40])

    def test_pack_rejects_wrong_logit_rows(self):
        with self.assertRaises(ValueError):
            pack_draft_payload(DraftPayload([1, 2, 3], np.zeros((3, 8), dtype=np.float16)))  # need γ+1=4


if __name__ == "__main__":
    unittest.main()
