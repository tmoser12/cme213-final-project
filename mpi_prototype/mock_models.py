"""Deterministic mock draft/target models — no neural nets, no GPU, no torch."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from mpi_prototype.protocol import DraftPayload, TargetResult
from mpi_prototype.sampler import sample_from_probs, speculative_acceptance


def _seed_from_prefix(prefix: Sequence[int], position: int, role: str) -> int:
    h = hash((tuple(prefix), position, role))
    return h & 0xFFFFFFFF


def mock_logits(prefix: Sequence[int], position: int, role: str, vocab_size: int) -> np.ndarray:
    """Return float32 logits for one forward position."""
    rng = np.random.default_rng(_seed_from_prefix(prefix, position, role))
    return rng.standard_normal(vocab_size, dtype=np.float32)


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    exp = np.exp(x)
    return (exp / exp.sum()).astype(np.float32)


class MockDraftModel:
    """Approximation model M_q — sequential gamma-token draft generation."""

    def __init__(self, *, vocab_size: int, prefix: list[int] | None = None) -> None:
        self.vocab_size = vocab_size
        self.prefix = list(prefix or [])

    def draft_gamma(self, gamma: int, rng: random.Random) -> DraftPayload:
        if gamma < 1:
            raise ValueError("gamma must be >= 1")

        draft_ids: list[int] = []
        logits_rows: list[np.ndarray] = []

        logits_rows.append(mock_logits(self.prefix, 0, "draft", self.vocab_size))

        for i in range(gamma):
            extended = self.prefix + draft_ids
            q_logits = mock_logits(extended, i + 1, "draft", self.vocab_size)
            logits_rows.append(q_logits)
            token = sample_from_probs(_softmax_1d(q_logits), rng)
            draft_ids.append(token)

        stacked = np.stack(logits_rows, axis=0).astype(np.float16)
        return DraftPayload(draft_token_ids=draft_ids, draft_logits=stacked)


def sync_draft_prefix(
    draft: MockDraftModel,
    draft_ids: list[int],
    result: TargetResult,
) -> None:
    """Update draft prefix after target verification.

    Rebuild from ``prefix_len``, optional bundled bonus from the prior deferred
    token, accepted drafts, and the newly deferred bonus (for the next γ loop).
    """
    accepted = draft_ids[: result.n_accepted]
    extra: list[int] = []
    if result.cache_pos_after > result.prefix_len + result.n_accepted:
        if len(draft.prefix) > result.prefix_len:
            extra = [draft.prefix[result.prefix_len]]
    draft.prefix = draft.prefix[: result.prefix_len] + extra + accepted
    draft.prefix.append(result.bonus_token)


def _target_p_rows(
    prefix: Sequence[int],
    draft_ids: Sequence[int],
    *,
    commit_bonus: int | None,
    vocab_size: int,
) -> list[np.ndarray]:
    """Build p_1 … p_{γ+1} rows aligned with target verify_gamma + sampler."""
    gamma = len(draft_ids)
    rows: list[np.ndarray] = []
    if commit_bonus is not None:
        for i in range(gamma + 1):
            extended = list(prefix) + [commit_bonus] + list(draft_ids[:i])
            rows.append(mock_logits(extended, len(extended) - 1, "target", vocab_size))
    else:
        rows.append(mock_logits(prefix, 0, "target", vocab_size))
        for i in range(gamma):
            extended = list(prefix) + list(draft_ids[: i + 1])
            rows.append(mock_logits(extended, i + 1, "target", vocab_size))
    return rows


class MockTargetModel:
    """Target model M_p — parallel verify over prefix + draft tokens."""

    def __init__(self, *, vocab_size: int, prefix: list[int] | None = None) -> None:
        self.vocab_size = vocab_size
        self.prefix = list(prefix or [])
        self._pending_bonus: int | None = None

    def flush_pending_bonus(self) -> None:
        """Append final deferred bonus (mirrors target flush after last iter)."""
        if self._pending_bonus is None:
            return
        self.prefix.append(self._pending_bonus)
        self._pending_bonus = None

    def verify_and_accept(
        self,
        payload: DraftPayload,
        rng: random.Random,
    ) -> TargetResult:
        commit_bonus = self._pending_bonus
        self._pending_bonus = None

        gamma = len(payload.draft_token_ids)
        prefix_len = len(self.prefix)

        p_rows = _target_p_rows(
            self.prefix,
            payload.draft_token_ids,
            commit_bonus=commit_bonus,
            vocab_size=self.vocab_size,
        )
        p_logits = np.stack(p_rows, axis=0)
        q_logits = payload.draft_logits.astype(np.float32)

        n_accepted, bonus_token = speculative_acceptance(
            p_logits,
            q_logits,
            payload.draft_token_ids,
            rng,
        )

        accepted = payload.draft_token_ids[:n_accepted]
        rollback_base = prefix_len + (1 if commit_bonus is not None else 0)
        kept = list(self.prefix[:prefix_len])
        if commit_bonus is not None:
            kept.append(commit_bonus)
        kept.extend(accepted)
        self.prefix = kept
        self._pending_bonus = bonus_token
        cache_pos_after = rollback_base + n_accepted

        return TargetResult(
            n_accepted=n_accepted,
            bonus_token=bonus_token,
            prefix_len=prefix_len,
            cache_pos_after=cache_pos_after,
        )
