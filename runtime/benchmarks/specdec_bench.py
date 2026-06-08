#!/usr/bin/env python3
"""
runtime/benchmarks/specdec_bench.py — speculative-decoding benchmark suite.

For every natural-language prompt in `benchmarks/prompts/*.jsonl`, measures decode
throughput (tok/s) in three configurations:

  1. baseline       : target-only (7B) stochastic decode on one GPU       (the control)
  2. single-GPU spec: 7B target + 0.5B draft on one GPU, draft graph ON    (for each γ)
  3. dual-GPU  spec : 7B target on cuda:0 + 0.5B draft on cuda:1 over MPI   (for each γ)

Prompts are tokenized with Qwen's HF tokenizer (chat template). Decoding is stochastic
(seeded); the draft model always runs in CUDA-graph mode. γ is swept over {2,4,6}.

This process does NO CUDA work itself (tokenization only). It writes a JSON job file,
runs the single-GPU phase as one subprocess and the dual-GPU phase as a 2-rank `mpirun`
subprocess (so each phase owns the GPUs cleanly), then merges and tabulates the results.

Run (2 GPUs):  bash slurm/run_specdec_bench.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from runtime.core.config import CONFIG_7B, RuntimeConfig
from runtime.speculative.types import MAX_VERIFY_GAMMA

DEFAULT_PROMPTS = PROJECT_ROOT / "runtime/benchmarks/prompts/mt_bench_subset.jsonl"
WORKER_MODULE = "runtime.benchmarks._specdec_worker"


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def tokenize(prompts: list[dict], model_path: str) -> list[dict]:
    """Token-ids per prompt via Qwen's chat template (Instruct format)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    out = []
    for p in prompts:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            add_generation_prompt=True,
            return_tensors=None,
        )
        out.append({"id": p["id"], "category": p.get("category", "?"), "ids": list(map(int, ids))})
    return out


def run_worker(mode: str, jobfile: str, resultfile: str) -> bool:
    """Launch the GPU worker (single or mpi). Returns True on success."""
    if mode == "single":
        cmd = [sys.executable, "-m", WORKER_MODULE, "--mode", "single", jobfile, resultfile]
    else:
        cmd = ["mpirun", "--oversubscribe", "-np", "2",
               sys.executable, "-m", WORKER_MODULE, "--mode", "mpi", jobfile, resultfile]

    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    print(f"\n>>> {mode}: {' '.join(cmd)}\n", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if proc.returncode != 0:
        print(f"!!! {mode} worker exited with code {proc.returncode}", file=sys.stderr, flush=True)
        return False
    return True


def _row(rec: dict) -> str:
    g = "   " if rec["gamma"] is None else f"γ={rec['gamma']}"
    label = {"baseline": "baseline (target-only)",
             "single": f"single-GPU spec  {g}",
             "mpi": f"dual-GPU  spec   {g}"}[rec["config"]]
    extra = "" if rec["accept_rate"] is None else \
        f"  (accept {rec['accept_rate']:.2f}/{rec['gamma']}, {rec['n_iters']} iters)"
    return f"  {label:<26} {rec['tok_s']:7.1f} tok/s  {rec['_speedup']:5.2f}x{extra}"


def report(prompts: list[dict], records: list[dict], spec: dict) -> str:
    by_id: dict[str, list[dict]] = {}
    for r in records:
        by_id.setdefault(r["id"], []).append(r)

    lines = [
        "=" * 72,
        "Speculative decoding benchmark — tokens/sec",
        "=" * 72,
        f"  decode      : stochastic (seed {spec['seed']}), {spec['n_new']} new tokens/prompt",
        f"  gamma sweep : {spec['gammas']}    draft CUDA graph: ON",
        f"  measurement : median of {spec['trials']} trials ({spec['warmup']} warmup)",
        "",
    ]
    order = {"baseline": 0, "single": 1, "mpi": 2}
    for p in prompts:
        recs = sorted(by_id.get(p["id"], []), key=lambda r: (order[r["config"]], r["gamma"] or 0))
        base = next((r["tok_s"] for r in recs if r["config"] == "baseline"), None)
        for r in recs:
            r["_speedup"] = r["tok_s"] / base if base else float("nan")
        ntok = len(next(pp for pp in spec["prompts"] if pp["id"] == p["id"])["ids"])
        lines.append(f"{p['id']} [{p.get('category', '?')}]  (prompt {ntok} tok)")
        lines += [_row(r) for r in recs]
        lines.append("")

    # cross-prompt mean per config
    lines += ["-" * 72, "Mean across prompts (tok/s):", ""]
    keys = [("baseline", None)] + [("single", g) for g in spec["gammas"]] + \
           [("mpi", g) for g in spec["gammas"]]
    for cfg, g in keys:
        vals = [r["tok_s"] for r in records if r["config"] == cfg and r["gamma"] == g]
        if not vals:
            continue
        label = "baseline (target-only)" if cfg == "baseline" else \
            ("single-GPU spec" if cfg == "single" else "dual-GPU  spec") + f"  γ={g}"
        lines.append(f"  {label:<26} {sum(vals) / len(vals):7.1f} tok/s")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Speculative decoding benchmark suite")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--n-new", type=int, default=128, dest="n_new")
    ap.add_argument("--gammas", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq", type=int, default=2048, dest="max_seq")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "runtime/benchmarks/specdec_report.txt")
    ap.add_argument("--single-only", action="store_true")
    ap.add_argument("--mpi-only", action="store_true")
    args = ap.parse_args()

    if any(g < 1 or g > MAX_VERIFY_GAMMA for g in args.gammas):
        print(f"error: every gamma must be in [1, {MAX_VERIFY_GAMMA}] (MAX_VERIFY_GAMMA)",
              file=sys.stderr)
        return 1

    cfg_t = RuntimeConfig.from_yaml(CONFIG_7B, project_root=PROJECT_ROOT)
    raw = load_prompts(args.prompts)
    tok_prompts = tokenize(raw, cfg_t.model_path)
    print(f"Tokenized {len(tok_prompts)} prompts from {args.prompts} "
          f"(lengths: {[len(p['ids']) for p in tok_prompts]})")

    spec = {"n_new": args.n_new, "seed": args.seed, "max_seq": args.max_seq,
            "warmup": args.warmup, "trials": args.trials, "gammas": args.gammas,
            "prompts": tok_prompts}

    tmp = Path(tempfile.mkdtemp(prefix="specdec_"))
    jobfile = tmp / "job.json"
    jobfile.write_text(json.dumps(spec))

    records: list[dict] = []
    if not args.mpi_only:
        res = tmp / "single.json"
        if run_worker("single", str(jobfile), str(res)) and res.exists():
            records += json.loads(res.read_text())["results"]
    if not args.single_only:
        res = tmp / "mpi.json"
        if run_worker("mpi", str(jobfile), str(res)) and res.exists():
            records += json.loads(res.read_text())["results"]
        else:
            print("!!! dual-GPU (MPI) phase failed or produced no results — "
                  "reporting single-GPU only.", file=sys.stderr)

    if not records:
        print("error: no results collected.", file=sys.stderr)
        return 1

    text = report(raw, records, spec)
    print("\n" + text)
    args.out.write_text(text + "\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
