#!/usr/bin/env bash
set -euo pipefail
cd /home/jzzeng/codes/vibeqc-issue-141
: "${SLURM_JOB_ID:?run inside Slurm}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler visibility}"
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=python:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="$PWD/build-cuda/libvibeqc.so"
export LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:${LD_LIBRARY_PATH:-}"
export VIBEQC_ONE_ELECTRON_DERIVATIVE_CUDA_TEST=1
python141=/home/jzzeng/codes/vibeqc-issue-204/.venv/bin/python
archive141=benchmarks/results/one-electron-derivatives-rtx5090
ctest --test-dir build-cuda --output-on-failure
"$python141" -m pytest tests/python/test_one_electron_derivatives_cuda.py -x -q --junitxml="$archive141/gpu-derivatives.xml"
VIBEQC_CHECKPOINT_DEVICE=cuda VIBEQC_ONE_ELECTRON_DERIVATIVES=generated "$python141" -m pytest tests/python/test_checkpoint.py -x -q --junitxml="$archive141/gpu-checkpoint.xml"
VIBEQC_RESOURCE_CUDA_TEST=1 VIBEQC_ONE_ELECTRON_DERIVATIVES=generated "$python141" -m pytest tests/python/test_hf_resources_cuda.py -x -q --junitxml="$archive141/gpu-resources.xml"
