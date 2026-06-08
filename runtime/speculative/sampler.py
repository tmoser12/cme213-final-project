"""Host-side stochastic speculative decoding sampler (Phase 8a).

Implements steps 3–4 of documentation/speculative_decoding.md on CPU.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax in fp32. ``logits`` shape ``[..., vocab]``."""
    return torch.softmax(logits.float(), dim=-1)


def sample_from_probs(probs: torch.Tensor, rng: random.Random) -> int:
    """Sample one token index from a 1-D probability vector."""
    if probs.dim() != 1:
        raise ValueError("probs must be 1-D")
    total = probs.sum().item()
    if total <= 0:
        raise ValueError("cannot sample from zero-mass distribution")
    probs = probs / total
    r = rng.random()
    cdf = 0.0
    for idx, p in enumerate(probs.tolist()):
        cdf += p
        if r <= cdf:
            return idx
    return int(probs.numel()) - 1


def adjusted_distribution(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    *,
    rejected: bool,
) -> torch.Tensor:
    """
    Build sampling distribution p' for the bonus token.

    ``p_logits`` / ``q_logits`` are 1-D logit vectors for p_{n+1} and q_{n+1}.
    """
    p = logits_to_probs(p_logits)
    if not rejected:
        return p
    q = logits_to_probs(q_logits)
    adjusted = torch.clamp(p - q, min=0.0)
    if adjusted.sum() <= 0:
        return p
    return adjusted / adjusted.sum()


def speculative_acceptance(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    draft_token_ids: Sequence[int],
    rng: random.Random,
) -> tuple[int, int]:
    """
    Run accept/reject + bonus resample.

    Args:
        p_logits: ``[γ+1, vocab]`` target logits (p_1 … p_{γ+1}).
        q_logits: ``[γ+1, vocab]`` draft logits (q_1 … q_{γ+1}).
        draft_token_ids: length γ draft tokens d_1 … d_γ (never includes a leading bonus).
        rng: Python RNG for reproducible tests.

    Returns:
        ``(n_accepted, bonus_token)`` where n is in [0, γ].
    """
    gamma = len(draft_token_ids)
    if p_logits.shape[0] != gamma + 1 or q_logits.shape[0] != gamma + 1:
        raise ValueError(
            f"expected {gamma + 1} logit rows, got p={p_logits.shape[0]} q={q_logits.shape[0]}"
        )

    p_probs = logits_to_probs(p_logits)
    q_probs = logits_to_probs(q_logits)

    n_accepted = gamma
    for i in range(gamma):
        token_id = int(draft_token_ids[i])
        p_i = p_probs[i, token_id].item()
        q_i = q_probs[i, token_id].item()
        if q_i <= 1e-30:
            n_accepted = i
            break
        r_i = rng.random()
        if r_i > p_i / q_i:
            n_accepted = i
            break

    rejected = n_accepted < gamma
    p_prime = adjusted_distribution(
        p_logits[n_accepted],
        q_logits[n_accepted],
        rejected=rejected,
    )
    bonus_token = sample_from_probs(p_prime, rng)
    return n_accepted, bonus_token
