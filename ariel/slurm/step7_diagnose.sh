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
# Requires: output/$RUN/${MODEL}_1.csv … ${MODEL}_4.csv (all 4 chains done)

set -e
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

CONFIG=${CONFIG:-configs/dr1_v6_2color.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
source slurm/_model.sh   # resolves MODEL (env > config "model" > 2color)
mkdir -p slurm/logs

echo "Step 7: diagnostics for run=$RUN model=$MODEL"
# --sig_figs=6: stansummary defaults to 2 significant figures, which prints
# R_hat as "1.0"/"1.1" and cannot resolve the 1.02 convergence threshold
# check_rhat.py tests. Harmless to the one other parser of this file
# (finalize_v5b_2m.py:_stansummary_scalars), which float()s whitespace-split
# columns.
../../cmdstan/bin/stansummary --sig_figs=6 output/$RUN/${MODEL}_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/${MODEL}_?.csv > output/$RUN/diagnose.txt

echo "--- stansummary ---"
cat output/$RUN/stansummary.txt

# Convergence gate on the global parameters. CmdStan's `diagnose` above also
# reports R-hat (at 1.01), but it exits 0 even when it warns, so nothing in
# this chain ever acted on it. This writes output/$RUN/rhat_check.json with an
# explicit pass/fail that downstream decisions can read.
#
# Non-fatal by default: a failing fit still proceeds to step8, so the
# catalogue and predictions exist to diagnose WHY it failed. Set RHAT_STRICT=1
# to make a FAIL exit 1 instead, which under `set -e` holds step8 on its
# afterok dependency.
echo "--- R-hat gate (threshold ${RHAT_THRESHOLD:-1.02}) ---"
python3 check_rhat.py --run $RUN --threshold "${RHAT_THRESHOLD:-1.02}" \
    ${RHAT_STRICT:+--strict} \
    || { rc=$?; [ "${RHAT_STRICT:-0}" = "1" ] && exit $rc; \
         echo "WARNING: check_rhat.py exited $rc; continuing (RHAT_STRICT unset)"; }

# DEBUG=1 produces 15 unadapted draws, which chainconsumer cannot summarize
# (IndexError in get_parameter_summary_max) -- so in debug mode a corner.py
# failure is a warning, not a step failure. stansummary/diagnose above already
# ran and are what a plumbing test needs. Matches run_batch_local.sh.
if [ "${DEBUG:-0}" = "1" ]; then
    python3 corner.py --run $RUN --model $MODEL \
        || echo "WARNING: corner.py failed; tolerated because DEBUG=1 (too few draws to plot)"
else
    # --model is passed explicitly here, unlike step8: corner.py hard-defaults
    # to "tophat" (corner.py:292-293), so dropping the flag would change
    # behaviour rather than let the config decide.
    python3 corner.py --run $RUN --model $MODEL
fi

touch output/$RUN/.step7_done
echo "DONE: step7 → stansummary.txt, diagnose.txt, rhat_check.json, ${MODEL}.png"
