#!/usr/bin/env python3
"""
Batch-size-1 speculative decoding benchmark sweep (target + draft CUDA graphs).

This script benchmarks the full single-process speculative decoding path in
`runtime/speculative/spec_decode.py` while forcing CUDA-graph mode on both:
  - target executor (verify/decode graph paths)
  - draft executor (decode graph path)

It sweeps across prompt lengths and generated sequence lengths, and reports:
  - tokens/sec and ms/token
  - acceptance rate stats (accepted drafts / iter and / gamma)
  - iteration efficiency metrics

Run on a GPU node, e.g.:
  bash slurm/run_python.sh runtime/benchmarks/spec_decode_full_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.buffers import allocate_buffers
from runtime.core.config import CONFIG_05B, CONFIG_7B, RuntimeConfig
from runtime.core.weights import load_weights_on_gpu
from runtime.executor import Qwen2Executor
from runtime.speculative.draft_runner import DraftRunner
from runtime.speculative.spec_decode import speculative_generate


@dataclass
class PromptCase:
    prompt_id: str
    category: str
    source_len: int
    prompt_len: int
    token_ids: list[int]


@dataclass
class TrialMetrics:
    prompt_id: str
    category: str
    prompt_len: int
    n_new_tokens: int
    trial_idx: int
    elapsed_s: float
    tokens_per_s: float
    ms_per_token: float
    n_generated: int
    n_iters: int
    accepted_total: int
    accept_per_iter: float
    accept_ratio: float
    tokens_per_iter: float
    iters_per_token: float


def parse_csv_ints(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one integer value")
    if any(v <= 0 for v in vals):
        raise ValueError("all values must be positive")
    return vals


def load_prompt_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"prompt file has no records: {path}")
    return records


def _repeat_to_length(tokens: list[int], target_len: int) -> list[int]:
    if not tokens:
        raise ValueError("cannot materialize prompt from empty token list")
    out: list[int] = []
    while len(out) < target_len:
        out.extend(tokens)
    return out[:target_len]


def build_prompt_cases(
    *,
    prompt_records: list[dict],
    tokenizer: AutoTokenizer,
    prompt_lengths: list[int],
    prompts_per_length: int,
) -> list[PromptCase]:
    tokenized: list[tuple[str, str, list[int]]] = []
    for rec in prompt_records:
        text = rec.get("prompt", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        tokenized.append((str(rec.get("id", "unknown")), str(rec.get("category", "unknown")), ids))
    if not tokenized:
        raise ValueError("none of the prompts tokenized to non-empty token IDs")

    cases: list[PromptCase] = []
    for len_idx, target_len in enumerate(prompt_lengths):
        for i in range(prompts_per_length):
            src_idx = (len_idx + i) % len(tokenized)
            prompt_id, category, src_ids = tokenized[src_idx]
            ids = _repeat_to_length(src_ids, target_len)
            cases.append(
                PromptCase(
                    prompt_id=f"{prompt_id}_L{target_len}_v{i}",
                    category=category,
                    source_len=len(src_ids),
                    prompt_len=target_len,
                    token_ids=ids,
                )
            )
    return cases


def summarize_run(rows: list[TrialMetrics]) -> dict[str, float]:
    tps = [r.tokens_per_s for r in rows]
    mspt = [r.ms_per_token for r in rows]
    acc = [r.accept_ratio for r in rows]
    acc_iter = [r.accept_per_iter for r in rows]
    tpi = [r.tokens_per_iter for r in rows]
    ipt = [r.iters_per_token for r in rows]
    return {
        "tokens_per_s_mean": statistics.mean(tps),
        "tokens_per_s_stdev": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
        "ms_per_token_mean": statistics.mean(mspt),
        "accept_ratio_mean": statistics.mean(acc),
        "accept_per_iter_mean": statistics.mean(acc_iter),
        "tokens_per_iter_mean": statistics.mean(tpi),
        "iters_per_token_mean": statistics.mean(ipt),
    }


def progress(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_trial(
    *,
    target: Qwen2Executor,
    draft: DraftRunner,
    prompt_ids: torch.Tensor,
    n_new_tokens: int,
    gamma: int,
    greedy: bool,
    rng_seed: int,
) -> TrialMetrics:
    # Robustness guard: executor.prefill validates against current cache_pos before
    # resetting internals, so ensure each trial starts from a clean cursor.
    target.reset_kv_cache()
    draft.executor.reset_kv_cache()

    rng = random.Random(rng_seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = speculative_generate(
        target,
        draft,
        prompt_ids,
        n_new_tokens,
        gamma,
        greedy=greedy,
        rng=rng,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    accepted_total = sum(res.accepted_per_iter)
    tokens_per_s = res.n_generated / dt
    return TrialMetrics(
        prompt_id="",
        category="",
        prompt_len=prompt_ids.shape[1],
        n_new_tokens=n_new_tokens,
        trial_idx=-1,
        elapsed_s=dt,
        tokens_per_s=tokens_per_s,
        ms_per_token=1000.0 * dt / res.n_generated,
        n_generated=res.n_generated,
        n_iters=res.n_iters,
        accepted_total=accepted_total,
        accept_per_iter=res.accept_rate,
        accept_ratio=(res.accept_rate / gamma) if gamma > 0 else 0.0,
        tokens_per_iter=(res.n_generated / res.n_iters) if res.n_iters else 0.0,
        iters_per_token=(res.n_iters / res.n_generated) if res.n_generated else 0.0,
    )


def run_two_gpu_mpi_suite(
    *,
    cases: list[dict],
    gamma: int,
    max_seq_len: int,
    greedy: bool,
) -> list[dict]:
    if not cases:
        return []
    suite_path = PROJECT_ROOT / "runtime" / "benchmarks" / "_tmp_mpi_suite.json"
    suite_path.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")

    cmd = [
        "mpirun",
        "--oversubscribe",
        "-np",
        "2",
        sys.executable,
        "-m",
        "runtime.speculative.mpi_coordinator",
        "--gamma",
        str(gamma),
        "--max-seq",
        str(max_seq_len),
        "--benchmark-suite-json",
        str(suite_path),
        "--benchmark-json",
    ]
    if greedy:
        cmd.append("--greedy")

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    progress(f"Launching persistent MPI benchmark suite with {len(cases)} case(s)")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    rows: list[dict] = []
    merged_lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        merged_lines.append(line)
        if line.startswith("BENCHMARK_JSON:"):
            rows.append(json.loads(line.split(":", 1)[1].strip()))
            progress(f"2-GPU progress: completed {len(rows)}/{len(cases)} suite case(s)")
        elif line.startswith("OK(case"):
            progress(f"MPI: {line}")

    rc = proc.wait()
    if rc != 0:
        merged = "\n".join(merged_lines)
        raise RuntimeError(
            "2-GPU MPI run failed with non-zero exit code.\n"
            f"Command: {' '.join(cmd)}\n"
            f"OUTPUT:\n{merged}\n"
        )

    if rows:
        return rows

    merged = "\n".join(merged_lines)
    raise RuntimeError(
        "2-GPU MPI run did not emit BENCHMARK_JSON output.\n"
        f"OUTPUT:\n{merged}"
    )


def summarize_two_gpu(rows: list[dict]) -> dict[str, float]:
    tps = [float(r["tokens_per_s"]) for r in rows]
    mspt = [float(r["ms_per_token"]) for r in rows]
    acc = [float(r["accept_ratio"]) for r in rows]
    acc_iter = [float(r["accept_per_iter"]) for r in rows]
    tpi = [float(r["tokens_per_iter"]) for r in rows]
    ipt = [float(r["iters_per_token"]) for r in rows]
    return {
        "tokens_per_s_mean": statistics.mean(tps),
        "tokens_per_s_stdev": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
        "ms_per_token_mean": statistics.mean(mspt),
        "accept_ratio_mean": statistics.mean(acc),
        "accept_per_iter_mean": statistics.mean(acc_iter),
        "tokens_per_iter_mean": statistics.mean(tpi),
        "iters_per_token_mean": statistics.mean(ipt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Speculative decode benchmark sweep (B=1, CUDA graphs on)")
    parser.add_argument("--prompts", type=str, default="runtime/benchmarks/prompts/mt_bench_subset.jsonl")
    parser.add_argument("--prompt-lengths", type=str, default="16,64,256")
    parser.add_argument("--prompts-per-length", type=int, default=2)
    parser.add_argument("--new-tokens", type=str, default="32,96,192")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic acceptance/sampling instead of greedy")
    parser.add_argument("--output-json", type=str, default="", help="Optional output path for machine-readable results")
    parser.add_argument("--setup", choices=["1gpu", "2gpu", "both"], default="both")
    parser.add_argument("--mpi-steps", type=int, default=32, help="Speculative iterations for each 2-GPU MPI trial")
    parser.add_argument("--mpi-trials", type=int, default=2, help="Number of MPI trials per selected prompt")
    parser.add_argument("--mpi-prompt-cases", type=int, default=3, help="How many prompt cases to benchmark in 2-GPU mode")
    args = parser.parse_args()

    if args.gamma < 1:
        raise ValueError("--gamma must be >= 1")
    if args.prompts_per_length < 1:
        raise ValueError("--prompts-per-length must be >= 1")
    if args.warmup < 0 or args.trials < 1:
        raise ValueError("--warmup must be >= 0 and --trials must be >= 1")
    if args.mpi_steps < 1 or args.mpi_trials < 1 or args.mpi_prompt_cases < 1:
        raise ValueError("--mpi-steps, --mpi-trials, and --mpi-prompt-cases must be >= 1")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on a GPU node.")
        return 2
    if args.setup in ("2gpu", "both") and torch.cuda.device_count() < 2:
        print(
            f"ERROR: --setup {args.setup} requires 2 visible GPUs; "
            f"found {torch.cuda.device_count()}."
        )
        return 2

    prompt_lengths = parse_csv_ints(args.prompt_lengths)
    new_tokens_grid = parse_csv_ints(args.new_tokens)
    greedy = not args.stochastic

    cfg_t = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    cfg_d = RuntimeConfig.from_yaml(CONFIG_05B, project_root=PROJECT_ROOT)
    prompt_records = load_prompt_records(PROJECT_ROOT / args.prompts)

    tokenizer = AutoTokenizer.from_pretrained(cfg_t.model_path, trust_remote_code=True)
    prompt_cases = build_prompt_cases(
        prompt_records=prompt_records,
        tokenizer=tokenizer,
        prompt_lengths=prompt_lengths,
        prompts_per_length=args.prompts_per_length,
    )

    max_prompt_len = max(c.prompt_len for c in prompt_cases)
    max_new = max(new_tokens_grid)
    max_seq_len = max_prompt_len + max_new + args.gamma + 4
    if max_seq_len > cfg_t.max_seq_len or max_seq_len > cfg_d.max_seq_len:
        raise ValueError(
            f"required max_seq_len={max_seq_len} exceeds model limits "
            f"(target={cfg_t.max_seq_len}, draft={cfg_d.max_seq_len})"
        )

    print("=== Speculative Decoding Benchmark (batch=1) ===")
    print(f"GPUs visible: {torch.cuda.device_count()}  |  primary: {torch.cuda.get_device_name(0)}")
    print(f"Setup={args.setup}  |  gamma={args.gamma}  |  mode={'greedy' if greedy else 'stochastic'}")
    progress("Tokenizing prompts and constructing benchmark cases")

    payload: dict = {
        "args": vars(args),
        "gpus_visible": torch.cuda.device_count(),
        "gpu0": torch.cuda.get_device_name(0),
    }

    if args.setup in ("1gpu", "both"):
        print("\n=== 1-GPU full-engine sweep (target + draft on one GPU, both graphed) ===")
        print(f"Prompt cases: {len(prompt_cases)}  |  Prompt lengths: {prompt_lengths}")
        print(f"New-token sweep: {new_tokens_grid}  |  Warmup={args.warmup}, trials={args.trials}")
        progress("Loading target + draft weights for 1-GPU path")

        wt, _ = load_weights_on_gpu(cfg_t, batch=1, device="cuda")
        bt = allocate_buffers(cfg_t, batch=1, max_seq_len=max_seq_len, device="cuda")
        target = Qwen2Executor(cfg_t, wt, bt, use_cuda_graph=True)

        wd, _ = load_weights_on_gpu(cfg_d, batch=1, device="cuda")
        bd = allocate_buffers(cfg_d, batch=1, max_seq_len=max_seq_len, device="cuda")
        draft = DraftRunner(Qwen2Executor(cfg_d, wd, bd, use_cuda_graph=True), seed=args.seed)

        all_trials: list[TrialMetrics] = []
        total_settings = len(prompt_cases) * len(new_tokens_grid)
        completed_settings = 0
        settings_t0 = time.perf_counter()

        for case_idx, case in enumerate(prompt_cases):
            prompt_ids = torch.tensor([case.token_ids], dtype=torch.int64, device="cuda")
            for n_new in new_tokens_grid:
                setting_start = time.perf_counter()
                progress(
                    f"1-GPU setting {completed_settings + 1}/{total_settings}: "
                    f"{case.prompt_id}, prompt={case.prompt_len}, new={n_new}"
                )
                warm_n = max(n_new, args.gamma + 2)
                for w in range(max(args.warmup, 1)):
                    _ = run_trial(
                        target=target,
                        draft=draft,
                        prompt_ids=prompt_ids,
                        n_new_tokens=warm_n,
                        gamma=args.gamma,
                        greedy=greedy,
                        rng_seed=args.seed + case_idx * 1000 + n_new * 10 + w,
                    )

                target_graph_ok = (
                    target._verify_state.get(args.gamma, {}).get("graph") is not None
                    and target._verify_state.get(args.gamma + 1, {}).get("graph") is not None
                )
                draft_graph_ok = draft.executor._decode_graph is not None
                if not target_graph_ok or not draft_graph_ok:
                    raise RuntimeError(
                        "CUDA graph capture not observed for both models: "
                        f"target_graph_ok={target_graph_ok}, draft_graph_ok={draft_graph_ok}"
                    )

                rows: list[TrialMetrics] = []
                for t in range(args.trials):
                    row = run_trial(
                        target=target,
                        draft=draft,
                        prompt_ids=prompt_ids,
                        n_new_tokens=n_new,
                        gamma=args.gamma,
                        greedy=greedy,
                        rng_seed=args.seed + case_idx * 10000 + n_new * 100 + t,
                    )
                    row.prompt_id = case.prompt_id
                    row.category = case.category
                    row.trial_idx = t
                    rows.append(row)
                    all_trials.append(row)

                summary = summarize_run(rows)
                print(
                    f"[{case.category:10s}] {case.prompt_id:20s} "
                    f"prompt={case.prompt_len:4d} new={n_new:4d}  "
                    f"tok/s={summary['tokens_per_s_mean']:7.2f} "
                    f"(±{summary['tokens_per_s_stdev']:5.2f})  "
                    f"acc={summary['accept_per_iter_mean']:.2f}/{args.gamma} "
                    f"({100.0 * summary['accept_ratio_mean']:.1f}%)"
                )
                completed_settings += 1
                elapsed = time.perf_counter() - settings_t0
                avg = elapsed / completed_settings
                remaining = avg * (total_settings - completed_settings)
                progress(
                    f"1-GPU progress: {completed_settings}/{total_settings} settings done "
                    f"(last {time.perf_counter() - setting_start:.1f}s, ETA {remaining:.1f}s)"
                )

        overall_1gpu = summarize_run(all_trials)
        print("\n=== 1-GPU Aggregate ===")
        print(
            f"tokens/sec: {overall_1gpu['tokens_per_s_mean']:.2f} ± {overall_1gpu['tokens_per_s_stdev']:.2f}  |  "
            f"ms/token: {overall_1gpu['ms_per_token_mean']:.3f}"
        )
        print(
            f"acceptance: {overall_1gpu['accept_per_iter_mean']:.3f}/{args.gamma}  "
            f"({100.0 * overall_1gpu['accept_ratio_mean']:.2f}%)"
        )
        print(
            f"efficiency: tokens/iter={overall_1gpu['tokens_per_iter_mean']:.3f}  "
            f"iters/token={overall_1gpu['iters_per_token_mean']:.3f}"
        )
        payload["single_gpu"] = {
            "overall": overall_1gpu,
            "trials": [asdict(r) for r in all_trials],
        }

    if args.setup in ("2gpu", "both"):
        print("\n=== 2-GPU MPI sweep (target cuda:0, draft cuda:1; both graphed) ===")
        selected_cases = prompt_cases[: min(args.mpi_prompt_cases, len(prompt_cases))]
        suite_cases: list[dict] = []
        for case_idx, case in enumerate(selected_cases):
            for t in range(args.mpi_trials):
                suite_cases.append(
                    {
                        "prompt_id": case.prompt_id,
                        "category": case.category,
                        "prompt": case.token_ids,
                        "steps": args.mpi_steps,
                        "trial_idx": t,
                        "seed": args.seed + case_idx * 1000 + t,
                    }
                )

        progress(
            f"Prepared 2-GPU suite: {len(selected_cases)} prompt case(s), "
            f"{len(suite_cases)} total MPI trial case(s)"
        )
        mpi_rows = run_two_gpu_mpi_suite(
            cases=suite_cases,
            gamma=args.gamma,
            max_seq_len=max_seq_len,
            greedy=greedy,
        )
        for row in mpi_rows:
            print(
                f"[{row['category']:10s}] {row['prompt_id']:20s} "
                f"prompt={int(row['prompt_len']):4d} steps={int(row['steps']):4d} "
                f"tok/s={float(row['tokens_per_s']):7.2f}  "
                f"acc={float(row['accept_per_iter']):.2f}/{args.gamma} "
                f"({100.0 * float(row['accept_ratio']):.1f}%)"
            )

        overall_2gpu = summarize_two_gpu(mpi_rows)
        print("\n=== 2-GPU Aggregate ===")
        print(
            f"tokens/sec: {overall_2gpu['tokens_per_s_mean']:.2f} ± {overall_2gpu['tokens_per_s_stdev']:.2f}  |  "
            f"ms/token: {overall_2gpu['ms_per_token_mean']:.3f}"
        )
        print(
            f"acceptance: {overall_2gpu['accept_per_iter_mean']:.3f}/{args.gamma}  "
            f"({100.0 * overall_2gpu['accept_ratio_mean']:.2f}%)"
        )
        print(
            f"efficiency: tokens/iter={overall_2gpu['tokens_per_iter_mean']:.3f}  "
            f"iters/token={overall_2gpu['iters_per_token_mean']:.3f}"
        )
        payload["two_gpu_mpi"] = {
            "overall": overall_2gpu,
            "trials": mpi_rows,
        }

    if args.output_json:
        out_path = PROJECT_ROOT / args.output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
