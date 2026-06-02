#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J step8_predict
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step8_predict_%j.out

# Step 8: Posterior predictive catalogs and covariance matrices.
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step8_predict.sh
# Requires: output/$RUN/2color_?.csv (all 4 chains done)

set -e

CONFIG=${CONFIG:-configs/dr1_v6_2color.json}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

echo "Step 8: color_predict.py for run=$RUN"
python color_predict.py --config $CONFIG --model 2color --xonly

touch output/$RUN/.step8_done
echo "DONE: step8 → color_catalog.fits, color_xonly_catalog.fits, color_cov.fits"
