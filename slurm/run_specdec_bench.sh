#!/bin/bash
# Speculative-decoding benchmark suite (runtime/benchmarks/specdec_bench.py).
# Needs 2 GPUs: the orchestrator runs the single-GPU phase in one subprocess, then
# spawns a 2-rank `mpirun` subprocess for the dual-GPU phase (rank 0 = 7B target on
# cuda:0, rank 1 = 0.5B draft on cuda:1). NVHPC module is loaded so mpirun/mpi4py work.
#
#   bash slurm/run_specdec_bench.sh                          # SLURM, 2 GPUs (default)
#   bash slurm/run_specdec_bench.sh --slurm-gpu              # explicit SLURM (same)
#   bash slurm/run_specdec_bench.sh --n-new 256 --gammas 2 4 6
#   bash slurm/run_specdec_bench.sh --single-only            # skip the MPI phase (1 GPU OK)
#   bash slurm/run_specdec_bench.sh --local                  # login/interactive node, 2 visible GPUs
#
# Extra args pass through to specdec_bench.py.

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
        --slurm-gpu|--slurm) MODE="slurm"; shift ;;
        --) shift; ARGS+=("$@"); break ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

ENTRY="runtime.benchmarks.specdec_bench"
RUN=(env PYTHONNOUSERSITE=1 "${PYTHON_BIN}" -m "${ENTRY}" "${ARGS[@]}")

echo "Entry: ${ENTRY}   Args: ${ARGS[*]:-<defaults>}   Mode: ${MODE}"

case "$MODE" in
    local)
        module load course/cme213/nvhpc/24.1 gnu12/12.3.0 2>/dev/null || true
        "${RUN[@]}"
        ;;
    slurm)
        srun \
            --partition=gpu-turing \
            --gres=gpu:2 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=00:30:00 \
            bash -lc "module load course/cme213/nvhpc/24.1 gnu12/12.3.0; ${RUN[*]}"
        ;;
esac
