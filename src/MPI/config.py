from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MPIBaselineConfig:
    draft_model_path: str
    target_model_path: str
    prompt_text: str | None
    prompt_file: str | None
    max_new_tokens: int
    temperature: float
    do_sample: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal MPI two-GPU baseline for Qwen draft/target models."
    )
    parser.add_argument(
        "--draft-model-path",
        default=os.environ.get("QWEN_05B_PATH"),
        help="Path to the draft model (default: $QWEN_05B_PATH).",
    )
    parser.add_argument(
        "--target-model-path",
        default=os.environ.get("QWEN_7B_PATH"),
        help="Path to the target model (default: $QWEN_7B_PATH).",
    )
    parser.add_argument(
        "--prompt",
        default="Write one sentence about speculative decoding.",
        help="Prompt text to run on both ranks.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional prompt file path. If set, overrides --prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 gives greedy decoding.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling in generation (default: greedy).",
    )
    return parser


def parse_config() -> MPIBaselineConfig:
    args = build_arg_parser().parse_args()
    if not args.draft_model_path:
        raise ValueError(
            "Draft model path is unset. Pass --draft-model-path or export QWEN_05B_PATH."
        )
    if not args.target_model_path:
        raise ValueError(
            "Target model path is unset. Pass --target-model-path or export QWEN_7B_PATH."
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative.")

    return MPIBaselineConfig(
        draft_model_path=args.draft_model_path,
        target_model_path=args.target_model_path,
        prompt_text=args.prompt,
        prompt_file=args.prompt_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )

