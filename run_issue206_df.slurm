#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:5090:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/lhshen/vibeqc-issue206/.artifacts/issue206-df-a/slurm-%j.out
#SBATCH --error=/home/lhshen/vibeqc-issue206/.artifacts/issue206-df-a/slurm-%j.err
set -euo pipefail
cd /home/lhshen/vibeqc-issue206
mkdir -p .artifacts/issue206-df-a
export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=/home/lhshen/.local/bin:/group/software/deepmd-kit-3.1.1/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VIBEQC_LIBRARY=$PWD/build/cuda-dev-fast/libvibeqc.so
export PYTHONPATH=$PWD/python
/home/lhshen/miniconda3/envs/dftgpu/bin/python benchmarks/issue206_df_matrix.py --run --repeats 5 --output-dir .artifacts/issue206-df-a
