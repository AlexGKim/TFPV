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
# Requires: output/$RUN/${MODEL}_?.csv (all 4 chains done)

set -e
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

CONFIG=${CONFIG:-configs/dr1_v6_2color.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
source slurm/_model.sh   # resolves MODEL (env > config "model" > 2color)
mkdir -p slurm/logs

echo "Step 8: color_predict.py for run=$RUN model=$MODEL"
# x-only is the unconditional default (color_predict.py no longer takes
# --xonly -- pass --full to additionally get the full z/g-conditioned model).
# DEBUG=1: skip the covariance (can take hours for large mock catalogs).
# --model is deliberately NOT passed. color_predict.py's documented
# precedence is explicit flag > config "model" > "color", so omitting the flag
# makes the config's own "model" key authoritative -- which is what setting
# "model": "2color_v10" was always for. Passing --model $MODEL would work too
# but would re-introduce a second place the model name has to be kept correct.
if [ "${DEBUG:-0}" = "1" ]; then
    python3 color_predict.py --config $CONFIG --no-cov
else
    python3 color_predict.py --config $CONFIG
fi

# Residual diagnostics run here, after the predictions. Ordering it after
# color_predict.py also means a plotting failure can no longer destroy the
# run's science output: under `set -e` this script's catalog/covariance are
# already written and committed by the time these plots are attempted.
# Non-fatal, like step7's corner.py: this is a plotting step, and under
# `set -e` a failure here would stop `touch .step8_done` below from ever
# running -- which would make batch_submit.sh's idempotency skip and
# batch_status.sh report a finished 24 h run as incomplete forever. The
# catalogue and covariance from color_predict.py above are already written.
echo "Step 8b: explore_residuals.py for run=$RUN kind=$MODEL"
python3 explore_residuals.py --config $CONFIG --kind $MODEL \
    || echo "WARNING: explore_residuals.py failed (non-fatal)"

touch output/$RUN/.step8_done
echo "DONE: step8 → color_xonly_catalog.fits, color_xonly_cov.h5 (pass --full for color_catalog.fits/color_cov.h5), explore_residuals/"
