#!/usr/bin/env python3
"""Single-process tests for mock spec-decode logic (no MPI)."""

from __future__ import annotations

import random
import unittest

import numpy as np

from mpi_prototype.mock_models import MockDraftModel, MockTargetModel, sync_draft_prefix
from mpi_prototype.protocol import (
    DraftPayload,
    TargetResult,
    draft_logits_byte_count,
    pack_draft_payload,
    pack_target_result,
    unpack_draft_payload,
    unpack_target_result,
)


class TestProtocol(unittest.TestCase):
    def test_draft_payload_roundtrip(self) -> None:
        payload = DraftPayload(
            draft_token_ids=[3, 7, 11],
            draft_logits=np.random.randn(4, 32).astype(np.float16),
        )
        ids, logits = pack_draft_payload(payload)
        back = unpack_draft_payload(ids, logits)
        self.assertEqual(back.draft_token_ids, payload.draft_token_ids)
        np.testing.assert_array_equal(back.draft_logits, payload.draft_logits)

    def test_target_result_roundtrip(self) -> None:
        result = TargetResult(
            n_accepted=2, bonus_token=99, prefix_len=10, cache_pos_after=12
        )
        back = unpack_target_result(pack_target_result(result))
        self.assertEqual(back, result)

    def test_fp16_byte_count(self) -> None:
        self.assertEqual(draft_logits_byte_count(4, 64), 5 * 64 * 2)


class TestMockSpecDecodeStep(unittest.TestCase):
    def test_draft_target_stay_in_sync_without_mpi(self) -> None:
        """Mirror one main-loop iteration in-process."""
        vocab = 48
        gamma = 4
        prefix = [1, 2, 3]
        seed = 123

        draft = MockDraftModel(vocab_size=vocab, prefix=list(prefix))
        target = MockTargetModel(vocab_size=vocab, prefix=list(prefix))
        draft_rng = random.Random(seed + 1)
        target_rng = random.Random(seed + 0)

        for _ in range(8):
            payload = draft.draft_gamma(gamma, draft_rng)
            ids, logits = pack_draft_payload(payload)
            payload_wire = unpack_draft_payload(ids, logits)

            result = target.verify_and_accept(payload_wire, target_rng)
            sync_draft_prefix(draft, payload.draft_token_ids, result)

        target.flush_pending_bonus()
        self.assertEqual(draft.prefix, target.prefix)

    def test_prefix_len_is_pre_step_length(self) -> None:
        draft = MockDraftModel(vocab_size=32, prefix=[5, 6])
        target = MockTargetModel(vocab_size=32, prefix=[5, 6])
        payload = draft.draft_gamma(3, random.Random(0))
        result = target.verify_and_accept(payload, random.Random(0))
        self.assertEqual(result.prefix_len, 2)
        sync_draft_prefix(draft, payload.draft_token_ids, result)
        self.assertEqual(len(draft.prefix), result.cache_pos_after + 1)


if __name__ == "__main__":
    unittest.main()
