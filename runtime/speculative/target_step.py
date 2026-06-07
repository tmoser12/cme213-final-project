"""Target-side speculative decoding step orchestration (Phase 8a, no MPI).

Iteration protocol (target rank)
--------------------------------
1. **First iter only:** ``prefill(L)``; draft also prefills and sends γ draft ids +
   ``[γ+1, vocab]`` draft logits. Target ``verify_gamma(γ drafts)`` — no bonus yet.
2. Host accept/reject on the γ drafts only; sample bonus ``t``; ``rollback_cache``;
   ``defer_bonus_token(t)`` — **no target forward on ``t`` yet**.
3. MPI ``n_accepted``, ``bonus_token``, ``prefix_len``, ``cache_pos_after`` to draft.
4. Draft syncs, runs γ autoregressive steps **continuing from ``t``** (γ excludes ``t``).
5. **Subsequent iters:** target ``take_pending_bonus()`` → ``verify_gamma(γ drafts,
   leading_bonus=t_prev)`` — one forward on ``[t_prev, d_1, …, d_γ]``. Accept/reject
   applies to ``d_1…d_γ`` only; ``t_prev`` is committed in KV but never re-sampled.
   Repeat 2–4.

After the final iteration, ``flush_pending_bonus()`` runs one S=1 forward for the
last deferred bonus (no following verify to bundle into).
"""

from __future__ import annotations

import random

import torch

from runtime.executor import Qwen2Executor
from runtime.speculative.sampler import speculative_acceptance
from runtime.speculative.types import SpeculativeStepResult


def target_speculative_step(
    executor: Qwen2Executor,
    draft_token_ids: torch.Tensor,
    draft_logits: torch.Tensor,
    rng: random.Random,
) -> SpeculativeStepResult:
    """
    One target speculative iteration (host sampler + GPU verify).

    See module docstring for the full multi-iter protocol. ``draft_logits`` must
    be ``[γ+1, vocab]`` aligned with the γ draft tokens (not including any leading
    bonus). **No MPI** in Phase 8a — draft payloads are passed in directly.
    """
    if draft_token_ids.dim() == 1:
        draft_token_ids = draft_token_ids.unsqueeze(0)
    if draft_token_ids.shape[0] != executor.buffers.batch:
        raise ValueError("draft_token_ids batch must match executor buffers")
    if draft_token_ids.shape[1] < 1:
        raise ValueError("draft_token_ids must contain at least one token")

    gamma = draft_token_ids.shape[1]
    prefix_len = executor.cache_pos
    if executor.p1_logits is None:
        raise RuntimeError("prefill() must run before target_speculative_step")

    leading_bonus = executor.take_pending_bonus()
    draft_list = draft_token_ids[0].tolist()
    rollback_base = prefix_len + (1 if leading_bonus is not None else 0)

    with torch.no_grad():
        p_verify = executor.verify_gamma(
            draft_token_ids,
            leading_bonus=leading_bonus,
        )
        if leading_bonus is not None:
            # Rows 0..γ-1 = p_1..p_γ for drafts; row γ = p_{γ+1} for bonus sample.
            # Leading bonus token is not passed to speculative_acceptance.
            p_all = p_verify[0]
        else:
            p_all = torch.cat(
                [executor.p1_logits.unsqueeze(0), p_verify[0]],
                dim=0,
            )
        p_cpu = p_all.cpu()
        q_cpu = draft_logits.float().cpu()
        if q_cpu.shape[0] != gamma + 1:
            raise ValueError(
                f"draft_logits must be [γ+1, vocab], got shape {tuple(q_cpu.shape)}"
            )

        n_accepted, bonus_token = speculative_acceptance(
            p_cpu,
            q_cpu,
            draft_list,
            rng,
        )

        executor.rollback_cache(rollback_base + n_accepted)
        executor.defer_bonus_token(bonus_token)

    cache_pos_after = rollback_base + n_accepted
    return SpeculativeStepResult(
        n_accepted=n_accepted,
        bonus_token=bonus_token,
        prefix_len=prefix_len,
        draft_token_ids=draft_list,
        new_prefix_len=cache_pos_after + 1,
        cache_pos_after=cache_pos_after,
        had_leading_bonus=leading_bonus is not None,
    )


def flush_pending_bonus(executor: Qwen2Executor) -> torch.Tensor | None:
    """Commit deferred bonus after the last speculative iteration (S=1 fallback)."""
    return executor.commit_pending_bonus()
