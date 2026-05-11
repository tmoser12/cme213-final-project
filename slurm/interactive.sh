#!/bin/bash
# slurm/interactive.sh
# Convenience wrapper for getting an interactive GPU shell.
# Usage: bash slurm/interactive.sh [num_gpus]
#
# Examples:
#   bash slurm/interactive.sh       # 1 GPU
#   bash slurm/interactive.sh 2     # 2 GPUs (for Week 2 MPI work)

NUM_GPUS=${1:-1}

srun \
    --partition=gpu-turing \
    --gres=gpu:${NUM_GPUS} \
    --ntasks=${NUM_GPUS} \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --pty bash
