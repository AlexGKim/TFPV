#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J step4_data
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step4_data_%j.out

# Step 4: Convert FITS catalog to Stan JSON format.
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step4_data.sh

set -e

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p output/$RUN slurm/logs

echo "Step 4: desi_data.py for run=$RUN config=$CONFIG"
python desi_data.py --config $CONFIG

touch output/$RUN/.step4_done
echo "DONE: step4 → output/$RUN/input.json, init.json"
