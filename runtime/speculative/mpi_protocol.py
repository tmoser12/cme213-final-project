"""MPI wire format for 2-rank speculative decoding (Phase 8c).

Ported from the mpi_prototype prototype. NumPy-only (the ``comm`` is passed in, so this
module does NOT import the MPI bindings — keeps the speculative package importable
without an MPI stack). The coordinator is the only module that touches MPI.

  Draft → target: draft_token_ids[γ] (int32), draft_logits[γ+1, vocab] (fp16)
  Target → draft: [n_accepted, bonus_token, prefix_len, cache_pos_after] (int32[4])

fp16 logits are sent as a uint8 view — direct float16 Send/Recv segfaults on the
cluster's NVHPC Open MPI bindings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rank roles — target on GPU 0, draft on GPU 1.
TARGET_RANK = 0
DRAFT_RANK = 1

TAG_DRAFT_PAYLOAD = 10
TAG_TARGET_RESULT = 11


@dataclass(frozen=True)
class DraftPayload:
    draft_token_ids: list[int]
    draft_logits: np.ndarray  # float16, shape [γ+1, vocab]


@dataclass(frozen=True)
class TargetResult:
    n_accepted: int
    bonus_token: int
    prefix_len: int        # target KV cursor at iter start (excludes deferred bonus)
    cache_pos_after: int   # target KV cursor after rollback (excludes new deferred bonus)


def pack_draft_payload(payload: DraftPayload) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(payload.draft_token_ids, dtype=np.int32)
    logits = np.ascontiguousarray(payload.draft_logits, dtype=np.float16)
    if logits.ndim != 2:
        raise ValueError(f"draft_logits must be 2-D, got shape {logits.shape}")
    if logits.shape[0] != ids.shape[0] + 1:
        raise ValueError(
            f"expected {ids.shape[0] + 1} logit rows for gamma={ids.shape[0]}, "
            f"got {logits.shape[0]}"
        )
    return ids, logits


def draft_logits_byte_count(gamma: int, vocab: int) -> int:
    return (gamma + 1) * vocab * np.dtype(np.float16).itemsize


def send_draft_payload(
    comm,
    payload: DraftPayload,
    dest: int,
    *,
    tag_ids: int = TAG_DRAFT_PAYLOAD,
    tag_logits: int = TAG_DRAFT_PAYLOAD + 1,
) -> None:
    """Send draft ids + fp16 logits (logits as uint8 — fp16 Send/Recv segfaults on NVHPC OMPI)."""
    ids, logits = pack_draft_payload(payload)
    comm.Send(ids, dest=dest, tag=tag_ids)
    comm.Send(logits.view(np.uint8), dest=dest, tag=tag_logits)


def recv_draft_payload(
    comm,
    source: int,
    *,
    gamma: int,
    vocab: int,
    tag_ids: int = TAG_DRAFT_PAYLOAD,
    tag_logits: int = TAG_DRAFT_PAYLOAD + 1,
) -> DraftPayload:
    ids = np.empty(gamma, dtype=np.int32)
    logits_bytes = np.empty(draft_logits_byte_count(gamma, vocab), dtype=np.uint8)
    comm.Recv(ids, source=source, tag=tag_ids)
    comm.Recv(logits_bytes, source=source, tag=tag_logits)
    logits = logits_bytes.view(np.float16).reshape(gamma + 1, vocab)
    return DraftPayload(
        draft_token_ids=ids.astype(np.int64).tolist(),
        draft_logits=np.ascontiguousarray(logits, dtype=np.float16),
    )


def pack_target_result(result: TargetResult) -> np.ndarray:
    return np.array(
        [result.n_accepted, result.bonus_token, result.prefix_len, result.cache_pos_after],
        dtype=np.int32,
    )


def unpack_target_result(buf: np.ndarray) -> TargetResult:
    if buf.shape != (4,):
        raise ValueError(f"expected 4 int32 values, got shape {buf.shape}")
    n, bonus, plen, cache_after = (int(x) for x in buf.tolist())
    return TargetResult(n_accepted=n, bonus_token=bonus, prefix_len=plen, cache_pos_after=cache_after)


def send_target_result(comm, result: TargetResult, dest: int, *, tag: int = TAG_TARGET_RESULT) -> None:
    comm.Send(pack_target_result(result), dest=dest, tag=tag)


def recv_target_result(comm, source: int, *, tag: int = TAG_TARGET_RESULT) -> TargetResult:
    buf = np.empty(4, dtype=np.int32)
    comm.Recv(buf, source=source, tag=tag)
    return unpack_target_result(buf)
