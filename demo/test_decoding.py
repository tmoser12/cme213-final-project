"""CPU tests for serving boundaries; the real verifier has GPU parity tests."""
import unittest
from demo.decoding import speculative_chunks


class FakeIds:
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, index):
        return FakeIds(self.ids[index[1]])

    def tolist(self):
        return self.ids


class FakeExecutor:
    def __init__(self):
        self.resets = 0
        self.cache_pos = 0

    def prefill(self, prompt):
        self.cache_pos = len(prompt)

    def reset_kv_cache(self):
        self.resets += 1
        self.cache_pos = 0


class FakeDraft:
    def __init__(self):
        self.executor = FakeExecutor()
        self.proposals = []
        self.feedback = []

    def prefill(self, prompt):
        self.executor.prefill(prompt)

    def generate_drafts(self, gamma):
        self.proposals.append(gamma)
        return FakeIds([10, 11, 12, 13][:gamma])

    def apply_target_feedback(self, accepted, bonus):
        self.feedback.append((accepted, bonus))


class TestSpeculativeStream(unittest.TestCase):
    def run_stream(self, limit, verify, eos=()):
        target, draft = FakeExecutor(), FakeDraft()
        chunks = list(speculative_chunks(target, draft, [1, 2], limit, 4, eos, verify=verify))
        self.assertEqual(target.resets, 1)
        self.assertEqual(draft.executor.resets, 1)
        return chunks, draft

    def test_final_round_reduces_gamma_and_discards_extra_bonus(self):
        chunks, draft = self.run_stream(7, lambda t, ids: (len(ids.ids), 20))
        self.assertEqual([c['ids'] for c in chunks], [[10, 11, 12, 13, 20], [10, 11]])
        self.assertEqual(draft.proposals, [4, 2])
        self.assertEqual(draft.feedback, [(4, 20)])

    def test_rejection_only_emits_accepted_prefix_and_bonus(self):
        chunks, draft = self.run_stream(4, lambda t, ids: (1, 99))
        self.assertEqual([c['ids'] for c in chunks], [[10, 99], [10, 99]])
        self.assertEqual(draft.feedback, [(1, 99)])

    def test_eos_inside_accepted_chunk_stops_without_feedback(self):
        chunks, draft = self.run_stream(20, lambda t, ids: (4, 20), eos={11})
        self.assertEqual([c['ids'] for c in chunks], [[10, 11]])
        self.assertFalse(draft.feedback)

    def test_eos_bonus_stops(self):
        chunks, draft = self.run_stream(20, lambda t, ids: (0, 99), eos={99})
        self.assertEqual(chunks[0]['ids'], [99])
        self.assertFalse(draft.feedback)

    def test_disconnect_closes_both_caches_before_next_round(self):
        target, draft = FakeExecutor(), FakeDraft()
        stream = speculative_chunks(target, draft, [1], 20, 4, (), verify=lambda t, ids: (1, 99))
        next(stream)
        stream.close()
        self.assertEqual(target.resets, 1)
        self.assertEqual(draft.executor.resets, 1)
        self.assertFalse(draft.feedback)

    def test_verification_failure_cleans_both_caches(self):
        target, draft = FakeExecutor(), FakeDraft()
        def fail(*args):
            raise ValueError('verification failed')
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            list(speculative_chunks(target, draft, [1], 20, 4, (), verify=fail))
        self.assertEqual(target.resets, 1)
        self.assertEqual(draft.executor.resets, 1)
