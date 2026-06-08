"""Single-process speculative decoding driver (Phase D6, no MPI).

Ties the draft (0.5B) and target (7B) executors together on one GPU:

    draft.generate_drafts(γ)  →  target verify + sample  →  draft.apply_target_feedback

Two modes:
  * ``greedy=True`` (default) — argmax standardization. Deterministic; the output is
    provably the target's own greedy sequence (the strong correctness gate).
  * ``greedy=False`` — the stochastic accept/reject sampler (``target_speculative_step``);
    a valid sample of the target distribution.

Both 7B (~14 GB) + 0.5B (~1 GB) fit on a 24 GB card, so no MPI is needed to validate
the protocol. Per-decode CUDA graphs on the draft (use_cuda_graph) are the speedup
lever; the target stays eager (memory-bound, ~1.00× graphed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.target_step import flush_pending_bonus, target_speculative_step


@dataclass
class SpecDecodeResult:
    tokens: list[int]                 # prompt + generated, truncated to prompt_len + n_new
    n_generated: int
    accepted_per_iter: list[int] = field(default_factory=list)
    n_iters: int = 0

    @property
    def accept_rate(self) -> float:
        """Mean drafts accepted per iteration / γ is reported by the caller; this is
        the mean accepted count per iteration."""
        return (sum(self.accepted_per_iter) / len(self.accepted_per_iter)
                if self.accepted_per_iter else 0.0)


def _greedy_target_step(target: Qwen2Executor, draft_ids: torch.Tensor) -> tuple[int, int]:
    """Greedy (argmax) verify of γ drafts against the target, with the bundled-bonus
    protocol. Accepts draft dᵢ iff dᵢ == argmax(pᵢ); bonus = target argmax at the
    divergence (or after all γ). Returns (n_accepted, bonus_token)."""
    gamma = draft_ids.shape[1]
    leading_bonus = target.take_pending_bonus()
    prefix = target.cache_pos
    rollback_base = prefix + (1 if leading_bonus is not None else 0)

    verify = target.verify_gamma_graph if target.use_cuda_graph else target.verify_gamma
    p = verify(draft_ids, leading_bonus=leading_bonus)  # [1, S, vocab]
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


def speculative_generate(
    target: Qwen2Executor,
    draft: DraftRunner,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
    gamma: int,
    *,
    greedy: bool = True,
    rng: random.Random | None = None,
) -> SpecDecodeResult:
    """Generate ``n_new_tokens`` via speculative decoding (draft proposes γ, target verifies)."""
    if rng is None:
        rng = random.Random(0)

    target.reset_kv_cache()
    target.prefill(prompt_ids)
    draft.prefill(prompt_ids)

    out = prompt_ids[0].tolist()
    accepted_per_iter: list[int] = []
    n_iters = 0
    while len(out) - prompt_ids.shape[1] < n_new_tokens:
        draft_ids, q_logits = draft.generate_drafts(gamma, greedy=greedy)
        if greedy:
            n_accepted, bonus = _greedy_target_step(target, draft_ids)
        else:
            result = target_speculative_step(target, draft_ids, q_logits, rng)
            n_accepted, bonus = result.n_accepted, result.bonus_token

        out.extend(draft_ids[0, :n_accepted].tolist())
        out.append(bonus)
        accepted_per_iter.append(n_accepted)
        n_iters += 1
        draft.apply_target_feedback(n_accepted, bonus)

    # Commit the last deferred bonus to the target KV (already in `out`).
    flush_pending_bonus(target)

    keep = prompt_ids.shape[1] + n_new_tokens
    return SpecDecodeResult(
        tokens=out[:keep],
        n_generated=keep - prompt_ids.shape[1],
        accepted_per_iter=accepted_per_iter,
        n_iters=n_iters,
    )
