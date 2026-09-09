#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$repository_root"
: "${SLURM_JOB_ID:?run through a finite Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?preserve scheduler-assigned visibility}"
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH=python:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VIBEQC_LIBRARY="${VIBEQC_LIBRARY:-$PWD/build-cuda/libvibeqc.so}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
PYTHON=${PYTHON:-python}
archive143=benchmarks/results/df-derivatives-rtx5090

NSYS=${NSYS:-nsys}
for budget143 in 0 1048576; do
 for mode143 in reference generated; do
  output143="$archive143/profile-$mode143-$budget143"
  "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output "$output143" "$PYTHON" benchmarks/profile_one_electron_force.py --case sp8 --batch 3 --mode generated_thread --fitted --df-budget "$budget143" --df-response "$mode143" --repeats 5 --output "$output143.json"
  "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum --format csv --force-export=true "$output143.nsys-rep" > "$output143.csv"
  gzip -f "$output143.sqlite"
 done
done
