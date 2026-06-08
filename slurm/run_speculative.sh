#!/bin/bash
# Launch 2-rank MPI speculative decoding (rank 0 = 7B target on cuda:0,
# rank 1 = 0.5B draft on cuda:1). cme213-hw7 pattern: ONE SLURM task,
# mpirun spawns the 2 ranks inside it.
#
#   bash slurm/run_speculative.sh                 # SLURM, 2 GPUs (default)
#   bash slurm/run_speculative.sh --steps 32 --gamma 4
#   bash slurm/run_speculative.sh --local         # login node (needs NVHPC module + 2 visible GPUs)
#
# Extra args after the mode flag pass through to the coordinator.

set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_PATH="/opt/ohpc/pub/compiler/anaconda3/2023.09-0"
PYTHON_BIN="${CONDA_PATH}/envs/cme213/bin/python"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="${HOME}/.conda/envs/cme213/bin/python"
export PROJECT_ROOT="$(pwd)"

MODE="slurm"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local) MODE="local"; shift ;;
        --) shift; ARGS+=("$@"); break ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(--steps 16 --gamma 4)

ENTRY="runtime.speculative.mpi_coordinator"
MPI_RUN=(env PYTHONNOUSERSITE=1 mpirun --oversubscribe -np 2 "${PYTHON_BIN}" -m "${ENTRY}" "${ARGS[@]}")

echo "Entry: ${ENTRY}   Args: ${ARGS[*]}   Mode: ${MODE}"

case "$MODE" in
    local)
        module load course/cme213/nvhpc/24.1 gnu12/12.3.0 2>/dev/null || true
        "${MPI_RUN[@]}"
        ;;
    slurm)
        srun \
            --partition=gpu-turing \
            --gres=gpu:2 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=00:20:00 \
            bash -lc "module load course/cme213/nvhpc/24.1 gnu12/12.3.0; ${MPI_RUN[*]}"
        ;;
esac
