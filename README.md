# Speculative Decoding — CME 213 Final Project
Tobias Moser · Eli Wandless

## Day 1 commands (login node)

```bash
# 1. Clone / init repo, then from project root:
bash setup.sh                         # pip install into conda env cme213

# 2. Download model weights (~15 GB total, ~10-20 min depending on network)
bash scripts/download_models.sh

# 3. Verify both models load and produce output on a GPU node
srun --partition=gpu-turing --gres=gpu:1 --pty \
     bash -c "source activate cme213 && python scripts/verify_env.py"
```

Expected verify_env.py output:
```
=== CUDA ===
  PyTorch version : 2.x.x
  CUDA version    : 12.x
  Device          : Quadro RTX 6000
  Compute cap     : SM 7.5
  VRAM            : 24.0 GB

=== Loading Qwen/Qwen2.5-7B-Instruct ===
  Load time   : ~30s
  Weight mem  : ~14.xx GB
  Vocab size  : 152064
  Sample output: ' Paris'

=== Loading Qwen/Qwen2.5-0.5B-Instruct ===
  Load time   : ~5s
  Weight mem  : ~0.9x GB
  Vocab size  : 152064
  Sample output: ' Paris'

All checks passed.
```

Both models must show `Vocab size: 152064` — this confirms the shared tokenizer
required for vanilla speculative decoding.

## Day 2 commands

```bash
# Submit baseline benchmark job
mkdir -p logs results
sbatch slurm/baseline.slurm

# Monitor job
squeue -u $USER
tail -f logs/baseline_<JOBID>.out

# Or run interactively (faster feedback during development)
bash slurm/interactive.sh
# Then inside the GPU shell:
source activate cme213
python benchmarks/run_baseline.py --model 7b --trials 3   # quick check
python benchmarks/run_baseline.py --model both            # full run

# Save results
python benchmarks/run_baseline.py --model both > results/baseline_$(date +%Y%m%d).txt
```

## Repo structure

```
spec-decoding/
├── README.md
├── requirements.txt
├── setup.sh                          # one-time env setup (login node)
├── .gitignore
│
├── scripts/
│   ├── download_models.sh            # Day 1: download Qwen2.5 weights
│   └── verify_env.py                 # Day 1: sanity check CUDA + both models
│
├── src/
│   ├── models/
│   │   └── loader.py                 # load_model(model_id, device) -> ModelBundle
│   ├── inference/
│   │   └── autoregressive.py         # greedy_decode() with KV cache + per-step timing
│   └── utils/
│       └── benchmarking.py           # run_trials(), summarise(), print_stats()
│
├── benchmarks/
│   ├── run_baseline.py               # Day 2: measure tokens/sec for both models
│   └── prompts/
│       └── mt_bench_subset.jsonl     # 5 prompts across writing/reasoning/code/math/knowledge
│
├── slurm/
│   ├── baseline.slurm                # sbatch script for Day 2 benchmark
│   └── interactive.sh                # convenience wrapper for srun --pty bash
│
└── results/                          # gitignored; save baseline_YYYYMMDD.txt here
    logs/                             # gitignored; SLURM stdout/stderr
```

## What to add in subsequent days

Week 1 remainder:
- `src/inference/speculative.py` — single-process SpecDec loop (both models on GPU 0)
- `benchmarks/run_speculative.py` — gamma sweep + alpha measurement

Week 2:
- `src/mpi/worker_draft.py` — rank 0 process (draft model loop + MPI send/recv)
- `src/mpi/worker_target.py` — rank 1 process (target verify loop + MPI send/recv)
- `benchmarks/run_mpi_baseline.py` — two-process blocking MPI benchmark
- `slurm/mpi_baseline.slurm` — ntasks=2, gres=gpu:2

Week 3:
- `kernels/verify_and_sample.cu` — fused accept/reject CUDA kernel
- `kernels/verify_and_sample_naive.cu` — multi-launch baseline for comparison
- `kernels/setup.py` — PyTorch C++ extension build script

Week 4:
- `benchmarks/run_ablation.py` — full ablation table (4 configurations)
- `analysis/roofline.py` — roofline plot generation
