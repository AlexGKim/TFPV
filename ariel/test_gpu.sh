#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

#export SLURM_CPU_BIND="cores"

srun ./tophat_g sample num_warmup=50 num_samples=50 num_chains=1 data file=output/c000_ph000_r000/input.json init=output/c000_ph000_r000/init.json \
    output file=output/c000_ph000_r000/tophat.csv
