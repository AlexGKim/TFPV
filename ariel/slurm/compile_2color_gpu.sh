#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J compile_2color_gpu
#SBATCH --mail-type=FAIL

# One-time: compile the 2color Stan model with GPU (OpenCL) support.
# Run from the ariel/ directory: sbatch slurm/compile_2color_gpu.sh

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia

# Workaround for CUDA OpenCL link error on Perlmutter
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

cd ../../cmdstan
make -j4 STAN_OPENCL=true ../TFPV/ariel/2color_g
cd ../TFPV/ariel

echo "Compiled: $(ls -lh 2color_g)"
