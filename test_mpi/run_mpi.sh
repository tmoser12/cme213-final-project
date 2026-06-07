#!/bin/bash
# Run mock MPI speculative-decoding prototype.
#
# Follows cme213-hw7 pattern: mpirun under SLURM (see cme213-hw7/hw7_q2.sh, Makefile).
# hw7 compiles with mpic++ then runs `mpirun build/matvec` inside an sbatch/srun allocation.
#
# Local (login node — NVHPC module must be loaded, as in interactive sessions):
#   bash test_mpi/run_mpi.sh
#
# SLURM CPU:
#   bash test_mpi/run_mpi.sh --slurm-cpu
#
# SLURM GPU (2 GPUs, production layout rehearsal):
#   bash test_mpi/run_mpi.sh --slurm-gpu

set -euo pipefail

cd "$(dirname "$0")/.."
CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
PYTHON_BIN="${CONDA_PATH}/envs/cme213/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="${HOME}/.conda/envs/cme213/bin/python"
fi
export PROJECT_ROOT="$(pwd)"

MODE="local"
ENTRY="test_mpi.main"
MPI_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm-cpu) MODE="slurm-cpu"; shift ;;
        --slurm-gpu) MODE="slurm-gpu"; shift ;;
        --benchmark) ENTRY="test_mpi.benchmark"; shift ;;
        --) shift; MPI_ARGS+=("$@"); break ;;
        *) MPI_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#MPI_ARGS[@]} -eq 0 ]]; then
    if [[ "$ENTRY" == "test_mpi.benchmark" ]]; then
        MPI_ARGS=(--trials 30 --warmup 5)
    else
        MPI_ARGS=(--steps 5 --gamma 4 --seed 42)
    fi
fi

# GPU SLURM runs always verify distinct cuda:0 / cuda:1 binding.
if [[ "$MODE" == "slurm-gpu" ]]; then
    if [[ "$ENTRY" == "test_mpi.main" ]]; then
        MPI_ARGS=(--require-gpu "${MPI_ARGS[@]}")
    fi
fi

MPI_CMD=(env PYTHONNOUSERSITE=1 mpirun -np 2 python -m "${ENTRY}" "${MPI_ARGS[@]}")
SLURM_MPI_CMD=(env PYTHONNOUSERSITE=1 mpirun --oversubscribe -np 2 "${PYTHON_BIN}" -m "${ENTRY}" "${MPI_ARGS[@]}")

module_init() {
    module load course/cme213/nvhpc/24.1 gnu12/12.3.0 2>/dev/null || true
}

echo "Mode: ${MODE}"
echo "Entry: ${ENTRY}"
echo "Args: ${MPI_ARGS[*]}"

case "$MODE" in
    local)
        module_init
        "${MPI_CMD[@]}"
        ;;
    slurm-cpu)
        srun \
            --partition=cpu \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --mem=8G \
            --time=00:05:00 \
            bash -lc "module load course/cme213/nvhpc/24.1 gnu12/12.3.0; ${SLURM_MPI_CMD[*]}"
        ;;
    slurm-gpu)
        srun \
            --partition=gpu-turing \
            --gres=gpu:2 \
            --ntasks=1 \
            --cpus-per-task=4 \
            --mem=16G \
            --time=00:05:00 \
            bash -lc "module load course/cme213/nvhpc/24.1 gnu12/12.3.0; ${SLURM_MPI_CMD[*]}"
        ;;
esac
