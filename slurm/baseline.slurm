#!/bin/bash
#SBATCH --job-name=spec-baseline
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# slurm/baseline.slurm
# Runs the baseline benchmark for both models.
# Submit from the project root:
#   mkdir -p logs
#   sbatch slurm/baseline.slurm

set -e

module load course/cme213/nvhpc/24.1
source .venv/bin/activate

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started   : $(date)"
echo ""

python benchmarks/run_baseline.py \
    --model both \
    --max-new-tokens 200 \
    --warmup 2 \
    --trials 5 \
    --prompts benchmarks/prompts/mt_bench_subset.jsonl

echo ""
echo "Finished  : $(date)"
