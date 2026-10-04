"""Streaming adapter for the repository's single-GPU greedy verifier.

Only verified tokens leave this iterator. Its caches are disposable: a completed
or cancelled request resets both executors rather than flushing a final bonus.
"""


def speculative_chunks(target, draft, prompt, limit, gamma, eos_ids, *, verify=None):
    if verify is None:
        from runtime.speculative.target_step import greedy_target_step
        verify = greedy_target_step
    emitted = 0
    try:
        target.prefill(prompt)
        draft.prefill(prompt)
        while emitted < limit:
            # Leave room for every proposed token, including a deferred target
            # bonus from the previous round. The bonus already counts in emitted.
            proposed = min(gamma, limit - emitted)
            ids = draft.generate_drafts(proposed)
            accepted, bonus = verify(target, ids)
            chunk = (ids[0, :accepted].tolist() + [bonus])[:limit - emitted]
            for i, token in enumerate(chunk):
                if token in eos_ids:
                    chunk = chunk[:i + 1]
                    break
            emitted += len(chunk)
            finished = emitted == limit or chunk[-1] in eos_ids
            yield {"ids": chunk, "accepted": accepted, "proposed": proposed, "rounds": 1}
            if finished:
                break
            # Rolls back rejected drafts and commits the target's bonus. A bonus
            # outside the smaller draft vocabulary raises a clear error here.
            draft.apply_target_feedback(accepted, bonus)
    finally:
        target.reset_kv_cache()
        draft.executor.reset_kv_cache()
