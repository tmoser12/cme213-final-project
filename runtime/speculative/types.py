"""Forward stages and the small-query attention limits."""

from __future__ import annotations

from enum import Enum

# Verification consumes up to 8 drafts plus the previous round's bonus.
MAX_VERIFY_GAMMA = 8
MAX_VERIFY_SEQ_LEN = 9  # γ + 1 when leading bonus bundled


class ForwardMode(Enum):
    """Explicit forward stage — VERIFY is not DECODE with S>1."""

    PREFILL = "prefill"
    VERIFY = "verify"
    DECODE = "decode"
