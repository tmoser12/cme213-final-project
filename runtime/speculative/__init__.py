"""Speculative decoding host logic (Phase 8). MPI coordinator added in Phase 8c."""

from runtime.speculative.sampler import speculative_acceptance
from runtime.speculative.types import (
    ForwardMode,
    MAX_VERIFY_GAMMA,
    MAX_VERIFY_SEQ_LEN,
    SpeculativeStepResult,
)

__all__ = [
    "ForwardMode",
    "MAX_VERIFY_GAMMA",
    "MAX_VERIFY_SEQ_LEN",
    "SpeculativeStepResult",
    "speculative_acceptance",
]
