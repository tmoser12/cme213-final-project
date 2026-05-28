#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SETUP_FILE="${PROJECT_ROOT}/setup.sh"

if [[ ! -f "${SETUP_FILE}" ]]; then
  echo "ERROR: Missing setup file at ${SETUP_FILE}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is unavailable in this shell. Initialize conda first." >&2
  exit 1
fi

source "${SETUP_FILE}"

if command -v module >/dev/null 2>&1; then
  module load course/cme213/nvhpc/24.1
  module_list="$(module list 2>&1 || true)"
  if [[ "${module_list}" != *"gnu12/12.3.0"* ]]; then
    echo "ERROR: gnu12/12.3.0 is not loaded. setup.sh should load it." >&2
    exit 1
  fi
  if [[ "${module_list}" != *"course/cme213/nvhpc/24.1"* ]]; then
    echo "ERROR: course/cme213/nvhpc/24.1 is not loaded." >&2
    exit 1
  fi
else
  echo "WARNING: module command unavailable. Skipping module checks." >&2
fi

if [[ -z "${QWEN_7B_PATH:-}" || -z "${QWEN_05B_PATH:-}" ]]; then
  echo "ERROR: QWEN_7B_PATH and QWEN_05B_PATH must be set (via setup.sh)." >&2
  exit 1
fi

python - <<'PY'
import importlib
for name in ("mpi4py", "torch", "transformers"):
    importlib.import_module(name)
print("Python dependency check passed.")
PY

LAUNCHER="${LAUNCHER:-srun}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMPT="${PROMPT:-Write one sentence about speculative decoding.}"
EXTRA_ARGS=("$@")

if [[ "${LAUNCHER}" == "srun" ]]; then
  exec srun \
    --partition=gpu-turing \
    --gres=gpu:2 \
    --ntasks=2 \
    --ntasks-per-node=2 \
    python -m src.MPI.run_mpi_baseline \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt "${PROMPT}" \
    "${EXTRA_ARGS[@]}"
elif [[ "${LAUNCHER}" == "mpirun" ]]; then
  exec mpirun -np 2 python -m src.MPI.run_mpi_baseline \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt "${PROMPT}" \
    "${EXTRA_ARGS[@]}"
else
  echo "ERROR: Unsupported LAUNCHER=${LAUNCHER}. Use srun or mpirun." >&2
  exit 1
fi

