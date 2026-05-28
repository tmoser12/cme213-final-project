# Minimal MPI Two-GPU Qwen Baseline

This directory contains a milestone baseline that launches two MPI ranks on two GPUs:

- rank 0 (`cuda:0`): draft model `Qwen2.5-0.5B-Instruct`
- rank 1 (`cuda:1`): target model `Qwen2.5-7B-Instruct`

Both ranks run inference in parallel and return timing metrics to rank 0. This baseline does **not** implement speculative decoding acceptance/rejection yet.

## Prerequisites

Before running, ensure the environment is active:

```bash
source /home/cme213/tobiascm/cme213-final-project/setup.sh
module load course/cme213/nvhpc/24.1
```

`setup.sh` should set:

- `QWEN_7B_PATH`
- `QWEN_05B_PATH`

Dependencies include `mpi4py`, `torch`, and `transformers`.

## Quick run

### Recommended (from login shell)

```bash
bash /home/cme213/tobiascm/cme213-final-project/src/MPI/run_mpi.sh
```

### Use mpirun inside an existing interactive allocation

```bash
LAUNCHER=mpirun bash /home/cme213/tobiascm/cme213-final-project/src/MPI/run_mpi.sh
```

### Custom prompt and token count

```bash
MAX_NEW_TOKENS=48 PROMPT="Explain speculative decoding in one sentence." \
bash /home/cme213/tobiascm/cme213-final-project/src/MPI/run_mpi.sh
```

## Direct entrypoint

```bash
srun --partition=gpu-turing --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 \
  python -m src.MPI.run_mpi_baseline --max-new-tokens 32 --prompt "Hello from MPI"
```

## Expected output signature

You should see:

- rank initialization logs showing role/device mapping
  - rank 0 -> `draft` on `cuda:0`
  - rank 1 -> `target` on `cuda:1`
- prompt hash printed by rank 0
- a final summary block with one JSON metrics line per rank, including:
  - `load_s`
  - `gen_s`
  - `new_tokens`
  - `tok_per_s`

## Limitations (intentional for this milestone)

- No speculative decode token proposal/verification loop yet
- No custom CUDA fused kernels in this path
- No tensor or pipeline model sharding

