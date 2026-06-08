#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 18:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J step6_chain
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step6_chain%a_%j.out

# Step 6: Single MCMC chain. Submit 4 of these independently via step6_submit.sh.
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json,CHAIN_ID=1 slurm/step6_chain.sh
# Requires: output/$RUN/input.json, init_MAP.json, metric.json (steps 5d+5e done)

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
CHAIN_ID=${CHAIN_ID:?'CHAIN_ID must be set (1-4)'}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

echo "Step 6: MCMC chain $CHAIN_ID for run=$RUN"
./2color_g sample num_warmup=250 num_samples=1000 \
    adapt save_metric=1 \
    algorithm=hmc metric=dense_e \
    metric_file=output/$RUN/metric.json \
    id=$CHAIN_ID \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color_${CHAIN_ID}.csv

touch output/$RUN/.step6_chain${CHAIN_ID}_done
echo "DONE: step6 chain $CHAIN_ID → output/$RUN/2color_${CHAIN_ID}.csv"
