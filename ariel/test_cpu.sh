#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks-per-node=1         # Request 1 task per node (total tasks = 1)
#SBATCH --cpus-per-task=1           # Request 1 CPU core per task (total cores = 1)

#export SLURM_CPU_BIND="cores"

echo "Job started at: $(date)"
srun ./tophat sample num_warmup=50 num_samples=50 num_chains=1 data file=output/c000_ph000_r000/input.json init=output/c000_ph000_r000/init.json \
    output file=output/c000_ph000_r000/tophat.csv
echo "Job finished at: $(date)"
