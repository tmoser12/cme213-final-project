"""Types for speculative decoding (Phase 8)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# small_q_attn_forward supports S in [1, 9]. Draft MPI payload is always γ tokens.
# Target VERIFY uses S=γ on the first post-prefill iter, S=γ+1 thereafter (bonus
# prepended). Phase 8b CUDA graphs capture fixed S=γ+1 with leading_bonus_valid
# masking so first and subsequent iters share one graph.
MAX_VERIFY_GAMMA = 8
MAX_VERIFY_SEQ_LEN = 9  # γ + 1 when leading bonus bundled


class ForwardMode(Enum):
    """Explicit forward stage — VERIFY is not DECODE with S>1."""

    PREFILL = "prefill"
    VERIFY = "verify"
    DECODE = "decode"


@dataclass
class SpeculativeStepResult:
    """Outcome of one target speculative-decoding iteration (no MPI)."""

    n_accepted: int
    bonus_token: int
    prefix_len: int
    draft_token_ids: list[int]
    new_prefix_len: int
    """Logical sequence length including deferred bonus (``prefix_len + n + 1``)."""

    cache_pos_after: int
    """KV cursor after rollback (accepted drafts + leading bonus if bundled)."""

    had_leading_bonus: bool
    """True when a deferred bonus from the prior iter was prepended (not re-verified)."""
