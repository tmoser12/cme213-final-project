"""
src/inference/autoregressive.py
Manual greedy decode loop with KV-cache reuse.

This is written as an explicit step-by-step loop rather than model.generate()
for two reasons:
  1. Separate prefill time (TTFT) from per-step decode time precisely.
  2. The loop structure mirrors what the SpecDec draft and target loops will look like,
     making it straightforward to instrument and extend later.

Greedy only for now (argmax). Temperature/sampling added in Week 2 when
the accept/reject step requires it.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch
from src.models.loader import ModelBundle


@dataclass
class DecodeResult:
    input_tokens: int
    output_tokens: int
    output_text: str
    ttft_ms: float           # prefill latency (time to first token)
    decode_ms: float         # total decode time excluding prefill
    tokens_per_sec: float    # output_tokens / (decode_ms / 1000), excludes prefill
    per_step_ms: List[float] = field(default_factory=list)  # latency of each decode step


def _cuda_event_timer():
    """Returns a pair of CUDA events for timing a region."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def greedy_decode(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = 200,
) -> DecodeResult:
    """
    Run greedy autoregressive decoding with KV cache.

    The first forward pass (prefill) processes the entire prompt and
    produces the first output token. Subsequent forward passes each take
    a single token as input and extend the KV cache by one position.

    This is the baseline all SpecDec results will be compared against.
    """
    device = bundle.device

    # Tokenize
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]          # shape [1, prompt_len]
    n_input = input_ids.shape[1]

    # ---- Prefill (time to first token) ----
    prefill_start, prefill_end = _cuda_event_timer()
    torch.cuda.synchronize(device)
    prefill_start.record()

    with torch.no_grad():
        out = bundle.model(
            input_ids=input_ids,
            use_cache=True,
        )

    prefill_end.record()
    torch.cuda.synchronize(device)
    ttft_ms = prefill_start.elapsed_time(prefill_end)

    past_kv = out.past_key_values
    # Greedy: take argmax over vocab at the last position
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    generated = [next_token.item()]

    # ---- Decode loop ----
    per_step_ms = []
    decode_total_ms = 0.0

    for _ in range(max_new_tokens - 1):
        if next_token.item() == bundle.tokenizer.eos_token_id:
            break

        step_start, step_end = _cuda_event_timer()
        torch.cuda.synchronize(device)
        step_start.record()

        with torch.no_grad():
            out = bundle.model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )

        step_end.record()
        torch.cuda.synchronize(device)
        step_ms = step_start.elapsed_time(step_end)

        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        per_step_ms.append(step_ms)
        decode_total_ms += step_ms

    output_text = bundle.tokenizer.decode(generated, skip_special_tokens=True)
    n_output = len(generated)
    tps = (n_output / (decode_total_ms / 1000.0)) if decode_total_ms > 0 else 0.0

    return DecodeResult(
        input_tokens=n_input,
        output_tokens=n_output,
        output_text=output_text,
        ttft_ms=ttft_ms,
        decode_ms=decode_total_ms,
        tokens_per_sec=tps,
        per_step_ms=per_step_ms,
    )
