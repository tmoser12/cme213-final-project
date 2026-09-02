"""HuggingFace-eager baseline helpers for ``run_baseline.py``.

Loads a Qwen2.5 checkpoint through ``transformers`` and runs an explicit greedy
decode loop (prefill + per-token steps with KV-cache reuse) so time-to-first-token
and per-step decode latency are measured separately. This is the stock-PyTorch
control that the native ``runtime/`` engine and speculative decoding are compared
against; nothing here touches the custom kernels.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# --------------------------------------------------------------------------- loading


@dataclass
class ModelBundle:
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: torch.device


def load_model(model_id: str, device: str) -> ModelBundle:
    """Load model + tokenizer in FP16 onto ``device`` (e.g. ``"cuda:0"``)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    return ModelBundle(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
    )


# --------------------------------------------------------------------------- decode


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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def greedy_decode(bundle: ModelBundle, prompt: str, max_new_tokens: int = 200) -> DecodeResult:
    """Greedy autoregressive decode with KV cache; prefill and decode timed separately."""
    device = bundle.device

    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]          # [1, prompt_len]
    n_input = input_ids.shape[1]

    # ---- Prefill (time to first token) ----
    prefill_start, prefill_end = _cuda_event_timer()
    torch.cuda.synchronize(device)
    prefill_start.record()
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, use_cache=True)
    prefill_end.record()
    torch.cuda.synchronize(device)
    ttft_ms = prefill_start.elapsed_time(prefill_end)

    past_kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    generated = [next_token.item()]

    # ---- Decode loop ----
    per_step_ms: List[float] = []
    decode_total_ms = 0.0
    for _ in range(max_new_tokens - 1):
        if next_token.item() == bundle.tokenizer.eos_token_id:
            break

        step_start, step_end = _cuda_event_timer()
        torch.cuda.synchronize(device)
        step_start.record()
        with torch.no_grad():
            out = bundle.model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
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


# --------------------------------------------------------------------------- stats


@dataclass
class BenchmarkStats:
    model_id: str
    prompt_preview: str
    n_trials: int
    input_tokens: int
    output_tokens_mean: float
    ttft_ms_mean: float
    ttft_ms_std: float
    tokens_per_sec_mean: float
    tokens_per_sec_std: float
    peak_vram_gb: float


def run_trials(
    decode_fn: Callable[[], DecodeResult],
    n_warmup: int = 2,
    n_trials: int = 5,
) -> List[DecodeResult]:
    """Run ``decode_fn`` for ``n_warmup`` discarded runs, then ``n_trials`` measured runs."""
    for _ in range(n_warmup):
        decode_fn()
    return [decode_fn() for _ in range(n_trials)]


def summarise(model_id: str, prompt: str, results: List[DecodeResult]) -> BenchmarkStats:
    tps_values = [r.tokens_per_sec for r in results]
    ttft_values = [r.ttft_ms for r in results]
    return BenchmarkStats(
        model_id=model_id,
        prompt_preview=prompt[:60].replace("\n", " "),
        n_trials=len(results),
        input_tokens=results[0].input_tokens,
        output_tokens_mean=statistics.mean(r.output_tokens for r in results),
        ttft_ms_mean=statistics.mean(ttft_values),
        ttft_ms_std=statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0.0,
        tokens_per_sec_mean=statistics.mean(tps_values),
        tokens_per_sec_std=statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def print_stats(stats: BenchmarkStats) -> None:
    print(f"  Model          : {stats.model_id}")
    print(f"  Prompt         : \"{stats.prompt_preview}...\"")
    print(f"  Input tokens   : {stats.input_tokens}")
    print(f"  Output tokens  : {stats.output_tokens_mean:.0f} (mean)")
    print(f"  TTFT           : {stats.ttft_ms_mean:.1f} ms  ± {stats.ttft_ms_std:.1f}")
    print(f"  Decode speed   : {stats.tokens_per_sec_mean:.1f} tok/s  ± {stats.tokens_per_sec_std:.1f}")
    print(f"  Peak VRAM      : {stats.peak_vram_gb:.2f} GB")
