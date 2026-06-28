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
#
# DEBUG mode: set DEBUG=1 (e.g. --export=...,DEBUG=1) to run a tiny 10+10 chain
# that still writes the standard 2color_${CHAIN_ID}.csv so step7/step8 can consume
# it. Intended for fast end-to-end plumbing tests; submit with `-q debug -t 0:05:00`.
# num_warmup/num_samples can also be overridden directly via NUM_WARMUP/NUM_SAMPLES.

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
CHAIN_ID=${CHAIN_ID:?'CHAIN_ID must be set (1-4)'}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

# Sampling depth: full by default, tiny in DEBUG mode (plumbing test only).
if [ "${DEBUG:-0}" = "1" ]; then
    NUM_WARMUP=${NUM_WARMUP:-10}
    NUM_SAMPLES=${NUM_SAMPLES:-10}
    echo "Step 6: DEBUG mode — $NUM_WARMUP warmup / $NUM_SAMPLES samples (results not science-grade)"
else
    NUM_WARMUP=${NUM_WARMUP:-250}
    NUM_SAMPLES=${NUM_SAMPLES:-1000}
fi

echo "Step 6: MCMC chain $CHAIN_ID for run=$RUN"
./2color_g sample num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES \
    adapt save_metric=1 \
    algorithm=hmc metric=dense_e \
    metric_file=output/$RUN/metric.json \
    id=$CHAIN_ID \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color_${CHAIN_ID}.csv

touch output/$RUN/.step6_chain${CHAIN_ID}_done
echo "DONE: step6 chain $CHAIN_ID → output/$RUN/2color_${CHAIN_ID}.csv"
