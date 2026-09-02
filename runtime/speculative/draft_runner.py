"""Draft-side speculative decoding API (Phase D5).

A thin wrapper over a draft ``Qwen2Executor`` (kernel_set="draft") that produces
γ speculative tokens for the target to verify and applies the target's feedback
(n_accepted + bonus token). Mirrors the target-side ``target_step.py``.

Protocol (draft rank), per docs/speculative_decoding.md:
  1. ``prefill(prompt)`` once; q₁ = last prompt logit.
  2. ``generate_drafts(γ)`` → (draft_ids[γ], q_logits[γ+1, vocab]): sample x₁~q₁,
     then γ decode steps committing xᵢ and yielding q_{i+1}; collect q₁…q_{γ+1}.
  3. Target verifies, returns (n_accepted, bonus_token).
  4. ``apply_target_feedback(n_accepted, bonus_token)``: roll the draft KV back to
     prefix+n_accepted, commit the bonus, refresh q₁ for the next iteration.

Sampling stays outside any CUDA graph (a host/GPU decision between replays). With
``executor.use_cuda_graph`` the γ-loop replays the decode graph — the launch-bound
hot path where graphs pay off (~1.6× on the 0.5B draft).
"""

from __future__ import annotations

import torch

from runtime.executor import Qwen2Executor


class DraftRunner:
    def __init__(self, executor: Qwen2Executor, *, seed: int | None = None) -> None:
        self.executor = executor
        self._device = executor.device
        self._vocab = executor.cfg.vocab_size
        self._gen: torch.Generator | None = None
        if seed is not None:
            self._gen = torch.Generator(device=self._device)
            self._gen.manual_seed(seed)
        self._last_logits: torch.Tensor | None = None  # q₁ for the next generate_drafts
        self._prefix_len: int = 0

    @property
    def prefix_len(self) -> int:
        """Draft KV length at the start of the current iteration (accepted + bonus)."""
        return self._prefix_len

    @property
    def last_logits(self) -> torch.Tensor | None:
        """q₁ — the draft distribution for the next token (``[vocab]``)."""
        return self._last_logits

    def _decode_fn(self):
        return (self.executor.decode_step_graph
                if self.executor.use_cuda_graph else self.executor.decode_step)

    def _tok(self, t: int) -> torch.Tensor:
        return torch.tensor([t], dtype=torch.int64, device=self._device)

    def _sample(self, logits_row: torch.Tensor, greedy: bool) -> int:
        """Sample one token id from a 1-D logit row (xᵢ ~ qᵢ, or argmax if greedy)."""
        if greedy:
            return int(logits_row.argmax().item())
        probs = torch.softmax(logits_row.float(), dim=-1)
        return int(torch.multinomial(probs, 1, generator=self._gen).item())

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill the prompt; record q₁ and the prefix length."""
        logits = self.executor.prefill(input_ids)
        self._last_logits = logits[0, -1].clone()
        self._prefix_len = self.executor.cache_pos
        return logits

    def generate_drafts(
        self,
        gamma: int,
        *,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively propose γ draft tokens.

        Returns ``(draft_ids[1, γ] int64, q_logits[γ+1, vocab])`` where row i is qᵢ
        (q₁…q_{γ+1}). q₁ is the pre-existing distribution; q_{i+1} comes from the
        decode that committed xᵢ. The draft KV holds x₁…x_γ at [prefix, prefix+γ).
        """
        if self._last_logits is None:
            raise RuntimeError("generate_drafts requires prefill() first")
        if gamma < 1:
            raise ValueError("gamma must be >= 1")

        self._prefix_len = self.executor.cache_pos
        decode = self._decode_fn()

        q_rows = [self._last_logits]            # q₁
        draft_ids: list[int] = []
        cur = self._sample(self._last_logits, greedy)   # x₁
        for i in range(gamma):
            draft_ids.append(cur)
            logits = decode(self._tok(cur))     # commit xᵢ, get q_{i+1}
            row = logits[0, -1].clone()         # clone: graph buffer is reused next call
            q_rows.append(row)
            if i < gamma - 1:
                cur = self._sample(row, greedy)  # x_{i+1}

        draft_token_ids = torch.tensor([draft_ids], dtype=torch.int64, device=self._device)
        q_logits = torch.stack(q_rows, dim=0)   # [γ+1, vocab]
        return draft_token_ids, q_logits

    def apply_target_feedback(self, n_accepted: int, bonus_token: int) -> None:
        """
        Roll the draft KV back to the accepted prefix and commit the bonus token.

        After this, the draft cursor is at ``prefix + n_accepted + 1`` and q₁ is the
        distribution following the bonus — ready for the next ``generate_drafts``.
        """
        if not (0 <= n_accepted):
            raise ValueError("n_accepted must be non-negative")
        if bonus_token < 0 or bonus_token >= self._vocab:
            # Target may sample a token id in [draft_vocab, target_vocab) that the
            # draft cannot embed (Qwen2.5 vocab mismatch: 151936 vs 152064). These
            # ids are reserved/near-unused; surface rather than silently corrupt.
            raise ValueError(
                f"bonus_token {bonus_token} outside draft vocab [0, {self._vocab}); "
                "target sampled a token the draft cannot embed (vocab mismatch)"
            )
        self.executor.rollback_cache(self._prefix_len + n_accepted)
        logits = self._decode_fn()(self._tok(bonus_token))
        self._last_logits = logits[0, -1].clone()
        self._prefix_len = self._prefix_len + n_accepted + 1
