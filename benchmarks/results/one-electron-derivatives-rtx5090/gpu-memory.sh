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
python141=/home/jzzeng/codes/vibeqc-issue-204/.venv/bin/python
archive141=benchmarks/results/one-electron-derivatives-rtx5090
"$python141" benchmarks/one_electron_gradient_contract.py --case sp8 --output "$archive141/contract-sp8.json"
"$python141" benchmarks/one_electron_gradient_contract.py --case sp8 --contraction-length 8 --output "$archive141/contract-sp8-long8.json"
"$python141" benchmarks/one_electron_gradient_contract.py --case sdf18-direct --output "$archive141/contract-sdf18.json"
for mode141 in scalar generated_thread; do
 /group/software/cuda-12.9.1/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output "/tmp/issue141-profile-df-$mode141" "$python141" benchmarks/profile_one_electron_force.py --case sp8 --batch 3 --mode "$mode141" --fitted --repeats 5 --output "$archive141/profile-df-$mode141.json"
 /group/software/cuda-12.9.1/bin/nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum --format csv --force-export=true "/tmp/issue141-profile-df-$mode141.nsys-rep" > "$archive141/profile-df-$mode141.csv"
done
