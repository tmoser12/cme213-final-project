#!/usr/bin/env python3
"""
benchmarks/run_baseline.py
Measure autoregressive greedy decode tokens/sec for
Qwen2.5-7B-Instruct and Qwen2.5-0.5B-Instruct on a single GPU.

Run via SLURM:
  sbatch slurm/baseline.slurm

Or interactively:
  srun --partition=gpu-turing --gres=gpu:1 --pty \\
       python benchmarks/run_baseline.py --model 7b

The baseline numbers from this script are the control values for the
entire project. Record them in results/baseline_<date>.txt.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path so src/ imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_model
from src.inference.autoregressive import greedy_decode
from src.utils.benchmarking import run_trials, summarise, print_stats

MODELS = {
    "7b":  "./models/Qwen2.5-7B-Instruct",
    "0.5b": "./models/Qwen2.5-0.5B-Instruct",
}


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def benchmark_model(model_id: str, prompts: list[dict], args) -> None:
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_id}")
    print(f"Device: cuda:0  |  max_new_tokens={args.max_new_tokens}  |  "
          f"warmup={args.warmup}  |  trials={args.trials}")
    print('='*60)

    torch.cuda.reset_peak_memory_stats()
    bundle = load_model(model_id, device="cuda:0")

    for p in prompts:
        prompt_text = p["prompt"]
        category = p.get("category", "unknown")

        def trial():
            return greedy_decode(bundle, prompt_text, max_new_tokens=args.max_new_tokens)

        results = run_trials(trial, n_warmup=args.warmup, n_trials=args.trials)
        stats = summarise(model_id, prompt_text, results)

        print(f"\n  [{category}]")
        print_stats(stats)

    # Cleanup before loading next model
    del bundle.model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Autoregressive baseline benchmark")
    parser.add_argument(
        "--model", choices=["7b", "0.5b", "both"], default="both",
        help="Which model(s) to benchmark"
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup runs (discarded)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of measured trials per prompt")
    parser.add_argument(
        "--prompts", type=str,
        default="benchmarks/prompts/mt_bench_subset.jsonl",
        help="Path to .jsonl prompt file"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run this on a GPU node.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    models_to_run = (
        list(MODELS.items()) if args.model == "both"
        else [(args.model, MODELS[args.model])]
    )

    for key, model_id in models_to_run:
        benchmark_model(model_id, prompts, args)

    print("\nDone. Record these numbers in results/baseline_$(date +%Y%m%d).txt")


if __name__ == "__main__":
    main()
