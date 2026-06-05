---
name: gpu-cluster
description: >-
  Stanford ICME HPCC GPU cluster environment and SLURM job execution. Use before
  running Python, CUDA kernels, benchmarks, or any shell commands on this
  project. Covers setup.sh requirements, gpu-turing partition limits, srun/sbatch
  semantics, and Quadro RTX 6000 hardware constraints.
---

# GPU Cluster & Execution Environment

This project runs on the Stanford ICME **HPCC** cluster, managed by **SLURM**. GPU work must go through SLURM — never run GPU workloads directly on the login node.

## Mandatory: Source `setup.sh` First

**Before executing any project code** (Python, benchmarks, kernel builds, shell scripts), the agent must ensure the environment is loaded:

```bash
cd /home/cme213/tobiascm/cme213-final-project
source setup.sh
```

`setup.sh` is not optional. It:

1. Activates the `cme213` conda env (Python 3.11, PyTorch 2.4.0, transformers 4.45.0)
2. Loads `gnu12/12.3.0` — required for PyTorch C++ extension builds (system GCC 8.5 is too old)
3. Sets model path variables: `QWEN_7B_PATH`, `QWEN_05B_PATH`, `HF_HOME`, `PROJECT_ROOT`

When writing or running commands, always prefix GPU work with `source setup.sh` (or `cd` to project root and source it in the same shell session). Kernel benchmark scripts like `src/kernels/*/run_benchmark.sh` already do this internally.

For one-off agent commands:

```bash
cd /home/cme213/tobiascm/cme213-final-project && source setup.sh && srun --partition=gpu-turing --gres=gpu:1 python -m src.kernels.rmsnorm.benchmark
```

---

## Cluster Overview

| Property | Value |
|----------|-------|
| Scheduler | SLURM 23.11.4 |
| GPU partition | `gpu-turing` |
| Nodes | 5 (`hpcc-gpu-5-1` … `hpcc-gpu-5-5`) |
| GPUs per node | 4 |
| **Total GPUs** | **20** |
| CPUs per node | 16 (8 cores × 2 threads) |
| RAM per node | 128 GB |
| Default job time | 30 minutes |
| **Max job time** | **30 minutes** (`MaxTime=00:30:00`) |

Jobs requesting more than 30 minutes will be rejected (`PartitionTimeLimit`). Plan benchmarks and training runs accordingly; use shorter trials or checkpoint/resume patterns if needed.

Access is restricted to groups `hpcc` and `cme213`.

---

## Running Python on GPU (general)

Use these patterns for **any** Python script or module — not just kernels or benchmarks.

### Recommended: `slurm/run_python.sh` wrapper

From the project root, this sources `setup.sh` and submits via `srun` automatically:

```bash
# Run a script
bash slurm/run_python.sh scripts/verify_env.py

# Run a module
bash slurm/run_python.sh -m src.kernels.rmsnorm.benchmark

# Pass arguments to your script
bash slurm/run_python.sh benchmarks/run_baseline.py --model both --trials 3

# Request multiple GPUs or custom resources
bash slurm/run_python.sh --gpus 2 --time 00:15:00 scripts/my_script.py --flag value
```

Wrapper options (before Python args): `--gpus N`, `--time HH:MM:SS`, `--mem SIZE`, `--cpus N`.

### One-off `srun` (manual)

Equivalent to the wrapper, for agent-executed commands:

```bash
cd /home/cme213/tobiascm/cme213-final-project
source setup.sh

# Script
srun --partition=gpu-turing --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
    python scripts/verify_env.py

# Module
srun --partition=gpu-turing --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
    python -m benchmarks.run_baseline --model both

# Script with CLI args
srun --partition=gpu-turing --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
    python my_script.py --epochs 5 --batch-size 8
```

### Interactive GPU shell, then Python

For iterative development — allocate a shell on a GPU node, then run Python commands inside it:

```bash
# Step 1: get a GPU shell (from project root)
bash slurm/interactive.sh          # 1 GPU
bash slurm/interactive.sh 2        # 2 GPUs

# Step 2: inside the allocated shell
cd /home/cme213/tobiascm/cme213-final-project
source setup.sh

python scripts/verify_env.py
python -m src.kernels.rmsnorm.benchmark
python my_script.py --arg value

# Quick CUDA check
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

The interactive shell lasts up to the partition max (30 minutes). Re-request if it expires.

### Verify the environment

Always run this after initial setup or when debugging GPU access:

```bash
bash slurm/run_python.sh scripts/verify_env.py
```

This confirms CUDA, PyTorch, and model loading on a real GPU node.

### Batch job template for arbitrary Python

Create a `.slurm` script for unattended runs:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/my-job_%j.out
#SBATCH --error=logs/my-job_%j.err

set -e
cd /home/cme213/tobiascm/cme213-final-project
source setup.sh

python scripts/my_script.py --arg value
# or: python -m package.module
```

Submit with `mkdir -p logs && sbatch slurm/my-job.slurm`.

---

## SLURM: How to Run Code

### One-off / interactive GPU commands — `srun`

Use `srun` for quick tests, benchmarks, and agent-executed commands:

```bash
source setup.sh
srun --partition=gpu-turing --gres=gpu:1 python script.py
```

Common flags used in this project:

| Flag | Typical value | Purpose |
|------|---------------|---------|
| `--partition=gpu-turing` | required | Target the Turing GPU nodes |
| `--gres=gpu:N` | 1–4 | Number of GPUs (max 4 per node) |
| `--ntasks=N` | match GPUs | MPI/multi-GPU tasks |
| `--cpus-per-task=4` | 4 | CPU cores per task |
| `--mem=32G` | 32G | Memory limit |
| `--time=HH:MM:SS` | ≤ 00:30:00 | Wall time (cannot exceed partition max) |
| `--pty bash` | interactive | Allocates a shell on a GPU node |

**Interactive GPU shell** (convenience wrapper):

```bash
bash slurm/interactive.sh       # 1 GPU, 2-hour request (capped by partition max)
bash slurm/interactive.sh 2     # 2 GPUs
```

### Batch jobs — `sbatch`

For longer scripted runs, submit a batch job:

```bash
mkdir -p logs
sbatch slurm/baseline.slurm
```

Batch scripts use `#SBATCH` directives at the top. Example from `slurm/baseline.slurm`:

```bash
#SBATCH --job-name=spec-baseline
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00          # will be capped to 30 min by partition
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
```

Inside the job script, activate the environment:

```bash
source setup.sh   # preferred — also loads gnu12
# or: source activate cme213
```

### Kernel benchmark pattern

Each kernel ships a `run_benchmark.sh` that sources `setup.sh`, clears the PyTorch JIT cache, and submits via `srun`:

```bash
bash src/kernels/rmsnorm/run_benchmark.sh
```

Equivalent manual invocation:

```bash
source setup.sh
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_rmsnorm_ops
srun --partition=gpu-turing --gres=gpu:1 python -m src.kernels.rmsnorm.benchmark
```

### Useful SLURM commands

```bash
sinfo -p gpu-turing                    # partition status
squeue -p gpu-turing                 # queued/running jobs
scontrol show job <jobid>              # job details
scancel <jobid>                        # cancel a job
```

---

## GPU Hardware: Quadro RTX 6000 (Turing)

All `gpu-turing` nodes have **NVIDIA Quadro RTX 6000** GPUs.

| Spec | Value |
|------|-------|
| Architecture | Turing (TU102) |
| Compute capability | **SM 7.5** |
| VRAM | 24 GB GDDR6 |
| Tensor Cores | 576 (2nd gen) — FP16, INT8, INT4 |
| FP32 | ~16.3 TFLOPS |
| Max threads/block | 1024 |
| Warp size | 32 |

**Agent implications:**

- Compile CUDA kernels with `-arch=sm_75`
- Prefer **FP16** mixed precision; Turing has **no BF16 or TF32** hardware acceleration
- 24 GB VRAM — Qwen2.5-7B fits in FP16 (~14 GB weights) with limited batch/sequence headroom
- PyTorch: 2.4.0+cu121; cluster nvcc via `cuda/12.2` module

For full hardware details, see [gpu-spec.md](gpu-spec.md) or `info/gpu_spec.md`.

---

## Environment Modules

Beyond `setup.sh`, the cluster uses Lmod modules. Relevant modules:

| Module | Purpose |
|--------|---------|
| `gnu12/12.3.0` | Loaded by `setup.sh`; GCC ≥ 9 for PyTorch extensions |
| `cuda/12.2` | CUDA toolkit (nvcc, headers) — usually pre-loaded |
| `course/cme213/nvhpc/24.1` | Course MPI/nvc++; safe to keep loaded — `jit.py` sanitizes NVHPC env pollution |

Verify with `module list` if builds fail. See `info/common_kernel_errors.md` for known build issues.

---

## Agent Checklist

Before running code for the user:

```
- [ ] Shell is on the project (cd to PROJECT_ROOT)
- [ ] Ran `source setup.sh` in the current shell session (or used `slurm/run_python.sh` which does this)
- [ ] GPU work goes through `srun` or `sbatch`, not bare login-node execution
- [ ] Prefer `bash slurm/run_python.sh <args>` for general Python execution
- [ ] Using `--partition=gpu-turing` and `--gres=gpu:N`
- [ ] Job time ≤ 30 minutes
- [ ] CUDA code targets sm_75 / FP16
```

After editing `.cu` or `bindings.cpp`, clear the JIT cache before re-benchmarking:

```bash
rm -rf ~/.cache/torch_extensions/py311_cu121/custom_<kernel>_ops
```
