"""Greedy draft proposals and KV-cache rollback for the single-GPU demo."""
from __future__ import annotations

import torch

from runtime.executor import Qwen2Executor


class DraftRunner:
    def __init__(self, executor: Qwen2Executor) -> None:
        self.executor = executor
        self._last_logits: torch.Tensor | None = None
        self._prefix_len = 0

    def _decode(self, token: int) -> torch.Tensor:
        ids = torch.tensor([token], dtype=torch.int64, device=self.executor.device)
        return self.executor.decode_step_graph(ids)

    # WALKTHROUGH [DRAFT_PREFILL] — see demo/WALKTHROUGH.md.
    # Keep an independent final-row copy; graph replay later reuses its output buffer.
    def prefill(self, input_ids: torch.Tensor) -> None:
        logits = self.executor.prefill(input_ids)
        self._last_logits = logits[0, -1].clone()
        self._prefix_len = self.executor.cache_pos

    # WALKTHROUGH [DRAFT] — see demo/WALKTHROUGH.md.
    # Each proposal depends on the previous one: gamma draft forwards are serial.
    # Even the final proposal is decoded so its tentative KV state is present.
    def generate_drafts(self, gamma: int) -> torch.Tensor:
        """Propose γ greedy tokens and tentatively write them to the draft cache."""
        if self._last_logits is None:
            raise RuntimeError("generate_drafts requires prefill() first")
        if not 1 <= gamma <= 8:
            raise ValueError("gamma must be between 1 and 8")
        self._prefix_len = self.executor.cache_pos
        token = int(self._last_logits.argmax().item())
        ids = []
        for i in range(gamma):
            ids.append(token)
            logits = self._decode(token)
            if i + 1 < gamma:
                # Consume the result before graph replay overwrites its buffer.
                token = int(logits[0, -1].argmax().item())
        return torch.tensor([ids], dtype=torch.int64, device=self.executor.device)

    # WALKTHROUGH [FEEDBACK] — see demo/WALKTHROUGH.md.
    # The draft must consume the bonus now to predict the next proposal. Target
    # can defer it because it will receive all proposals together in the next verify.
    def apply_target_feedback(self, n_accepted: int, bonus_token: int) -> None:
        """Discard rejected drafts, commit the target bonus, refresh next logits."""
        proposed = self.executor.cache_pos - self._prefix_len
        if not 0 <= n_accepted <= proposed:
            raise ValueError("accepted count must be within the proposed draft length")
        if not 0 <= bonus_token < self.executor.cfg.vocab_size:
            raise ValueError(
                f"Target token {bonus_token} is outside the draft vocabulary "
                f"[0, {self.executor.cfg.vocab_size}); the draft cannot embed it."
            )
        self.executor.rollback_cache(self._prefix_len + n_accepted)
        self._last_logits = self._decode(bonus_token)[0, -1].clone()
        self._prefix_len = self.executor.cache_pos
