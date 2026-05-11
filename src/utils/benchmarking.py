"""
src/utils/benchmarking.py
Utilities for running timed inference trials and reporting results.
"""

import statistics
import torch
from dataclasses import dataclass
from typing import Callable, List
from src.inference.autoregressive import DecodeResult


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
    """
    Run decode_fn for n_warmup + n_trials iterations.
    Warmup runs are discarded. Returns only the measured trials.

    Warmup matters because the first one or two runs on a GPU node
    include CUDA kernel compilation and caching overhead that would
    inflate the first measurement.
    """
    for _ in range(n_warmup):
        decode_fn()
    results = []
    for _ in range(n_trials):
        results.append(decode_fn())
    return results


def summarise(
    model_id: str,
    prompt: str,
    results: List[DecodeResult],
) -> BenchmarkStats:
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
