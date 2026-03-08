#!/bin/bash
#SBATCH -A desi
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:01:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

#export SLURM_CPU_BIND="cores"
srun ./examples/bernoulli/bernoulli sample\
  data file=examples/bernoulli/bernoulli.data.json
