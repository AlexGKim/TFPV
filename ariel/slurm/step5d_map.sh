#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J step5d_map
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step5d_map_%j.out

# Step 5d: MAP optimization → init_MAP.json
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5d_map.sh
# Requires: output/$RUN/input.json, init.json (step4 must be done)

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

echo "Step 5d: MAP optimize for run=$RUN"
./2color_g optimize \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv

python make_map_init.py --run $RUN

touch output/$RUN/.step5d_done
echo "DONE: step5d → output/$RUN/init_MAP.json"
