"""NumPy-only speculative sampler for MPI mock tests (no torch import).

Mirrors runtime/speculative/sampler.py so mpi_prototype avoids torch+MPI fork issues
on the login node. Production integration still uses the runtime torch sampler.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max()
    exp = np.exp(x)
    return (exp / exp.sum()).astype(np.float32)


def sample_from_probs(probs: np.ndarray, rng: random.Random) -> int:
    if probs.ndim != 1:
        raise ValueError("probs must be 1-D")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("cannot sample from zero-mass distribution")
    probs = probs / total
    r = rng.random()
    cdf = 0.0
    for idx, p in enumerate(probs.tolist()):
        cdf += p
        if r <= cdf:
            return idx
    return int(probs.size) - 1


def _adjusted_distribution(
    p_logits: np.ndarray,
    q_logits: np.ndarray,
    *,
    rejected: bool,
) -> np.ndarray:
    p = _softmax(p_logits)
    if not rejected:
        return p
    q = _softmax(q_logits)
    adjusted = np.clip(p - q, 0.0, None)
    if adjusted.sum() <= 0:
        return p
    return adjusted / adjusted.sum()


def speculative_acceptance(
    p_logits: np.ndarray,
    q_logits: np.ndarray,
    draft_token_ids: Sequence[int],
    rng: random.Random,
) -> tuple[int, int]:
    """Accept/reject + bonus resample. Logits shape [gamma+1, vocab]."""
    gamma = len(draft_token_ids)
    if p_logits.shape[0] != gamma + 1 or q_logits.shape[0] != gamma + 1:
        raise ValueError(
            f"expected {gamma + 1} logit rows, got p={p_logits.shape[0]} q={q_logits.shape[0]}"
        )

    p_probs = _softmax(p_logits)
    q_probs = _softmax(q_logits)

    n_accepted = gamma
    for i in range(gamma):
        token_id = int(draft_token_ids[i])
        p_i = float(p_probs[i, token_id])
        q_i = float(q_probs[i, token_id])
        if q_i <= 1e-30:
            n_accepted = i
            break
        if rng.random() > p_i / q_i:
            n_accepted = i
            break

    rejected = n_accepted < gamma
    p_prime = _adjusted_distribution(
        p_logits[n_accepted],
        q_logits[n_accepted],
        rejected=rejected,
    )
    bonus_token = sample_from_probs(p_prime, rng)
    return n_accepted, bonus_token
