#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:5090:1
#SBATCH --time=02:00:00
#SBATCH --output=issue206-df-a-%j.out
#SBATCH --error=issue206-df-a-%j.err
set -euo pipefail

# Resolve the checkout from this script so the runner works from any clone.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ISSUE206_OUTPUT_DIR:-${ROOT}/.artifacts/issue206-df-a}"
CUDA_ROOT="${CUDA_HOME:-/group/software/cuda-12.9.1}"
PYTHON_BIN="${ISSUE206_PYTHON:-python3}"

mkdir -p "${OUTPUT_DIR}"
export CUDA_HOME="${CUDA_ROOT}"
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VIBEQC_LIBRARY="${ROOT}/build/cuda-dev-fast/libvibeqc.so"
export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
"${PYTHON_BIN}" benchmarks/issue206_df_matrix.py \
  --run \
  --repeats "${ISSUE206_REPEATS:-5}" \
  --python "${PYTHON_BIN}" \
  --library "${VIBEQC_LIBRARY}" \
  --output-dir "${OUTPUT_DIR}"
