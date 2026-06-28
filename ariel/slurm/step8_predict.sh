#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:30:00
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
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

CONFIG=${CONFIG:-configs/dr1_v6_2color.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

echo "Step 8: color_predict.py for run=$RUN"
# DEBUG=1: skip the covariance (can take hours for large mock catalogs).
if [ "${DEBUG:-0}" = "1" ]; then
    python3 color_predict.py --config $CONFIG --model 2color --xonly --no-cov
else
    python3 color_predict.py --config $CONFIG --model 2color --xonly
fi

touch output/$RUN/.step8_done
echo "DONE: step8 → color_catalog.fits, color_xonly_catalog.fits, color_cov.fits"
