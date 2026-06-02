#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J step5e_metric
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step5e_metric_%j.out

# Step 5e: Short MCMC run to build the pre-computed mass matrix (metric.json).
# ONE-TIME SETUP: metric.json can be reused across different FITS inputs.
# To reuse an existing metric, copy it: cp output/DR1_v6_2color/metric.json output/$RUN/
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5e_metric.sh
# Requires: output/$RUN/input.json, init_MAP.json (step5d must be done)

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

# Use an existing metric as the initial covariance if available
METRIC_ARG=""
if [ -f "output/$RUN/metric.json" ]; then
    METRIC_ARG="metric_file=output/$RUN/metric.json"
    echo "Step 5e: using existing metric.json as initial covariance"
elif [ -f "output/DR1_v6_2color/metric.json" ]; then
    METRIC_ARG="metric_file=output/DR1_v6_2color/metric.json"
    echo "Step 5e: using DR1_v6_2color/metric.json as initial covariance"
else
    echo "Step 5e: no existing metric found, starting from identity"
fi

echo "Step 5e: short metric-building run for run=$RUN (~7h)"
./2color_g sample num_warmup=100 num_samples=100 num_chains=1 \
    algorithm=hmc metric=dense_e \
    $METRIC_ARG \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color_metric_build.csv

python make_metric.py --run $RUN \
    --csv output/$RUN/2color_metric_build.csv \
    --out output/$RUN/metric.json

touch output/$RUN/.step5e_done
echo "DONE: step5e → output/$RUN/metric.json"
