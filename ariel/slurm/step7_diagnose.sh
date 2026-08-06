#!/bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -J step7_diagnose
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step7_diagnose_%j.out

# Step 7: Stan diagnostics + corner plot.
# (explore_residuals.py runs in step8, after color_predict.py — see step8_predict.sh.)
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step7_diagnose.sh
# Requires: output/$RUN/2color_1.csv … 2color_4.csv (all 4 chains done)

set -e
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

CONFIG=${CONFIG:-configs/dr1_v6_2color.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

echo "Step 7: diagnostics for run=$RUN"
../../cmdstan/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt

echo "--- stansummary ---"
cat output/$RUN/stansummary.txt

python3 corner.py --run $RUN --model 2color

touch output/$RUN/.step7_done
echo "DONE: step7 → stansummary.txt, diagnose.txt, 2color.png"
