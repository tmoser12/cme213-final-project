"""Target verification for one round of greedy speculative decoding."""
from __future__ import annotations

import torch

from runtime.executor import Qwen2Executor


# WALKTHROUGH [VERIFY] — see demo/WALKTHROUGH.md.
# Align each draft with logits for the prefix BEFORE that draft. Stop at the
# first mismatch; logits after a rejected token no longer describe the accepted path.
def greedy_target_step(target: Qwen2Executor, draft_ids: torch.Tensor) -> tuple[int, int]:
    """Greedy (argmax) verify of γ drafts against the target, with the bundled-bonus
    protocol. Accepts draft dᵢ iff dᵢ == argmax(pᵢ); bonus = target argmax at the
    divergence (or after all γ). Returns (n_accepted, bonus_token)."""
    gamma = draft_ids.shape[1]
    leading_bonus = target.take_pending_bonus()
    prefix = target.cache_pos
    rollback_base = prefix + (1 if leading_bonus is not None else 0)

    p = target.verify_gamma(draft_ids, leading_bonus=leading_bonus)  # [1, S, vocab]
    if leading_bonus is not None:
        p_rows = p[0]                                                # p_1..p_{γ+1}
    else:
        p_rows = torch.cat([target.p1_logits.unsqueeze(0), p[0]], dim=0)

    drafts = draft_ids[0].tolist()
    n = gamma
    for i in range(gamma):
        if drafts[i] != int(p_rows[i].argmax().item()):
            n = i
            break
    bonus = int(p_rows[n].argmax().item())
    target.rollback_cache(rollback_base + n)
    target.defer_bonus_token(bonus)
    return n, bonus
