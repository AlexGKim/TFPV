#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J step5e_metric
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step5e_metric_%j.out

# Step 5e: Pathfinder-seeded dense_e warmup metric -> output/$RUN/pf_metric.json
#
# OPT-IN. Nothing submits this unless you ask for it:
#   sbatch --export=CONFIG=$CFG,MODEL=2color_v10 slurm/step5e_metric.sh
#   bash slurm/batch_submit.sh ... --metric pf        # inserts it in the chain
# With METRIC unset, step6 does not pass metric_file and the pipeline samples
# from the identity metric exactly as before.
#
# Why it is back
# -------------
# aaa52d3 removed step5e_metric.sh and make_metric.py, and the reasoning was
# sound FOR v5: under the rank-1 intrinsic covariance S = w w^T the warmup
# funnel that a pre-built metric existed to avoid is gone, a 100-draw np.cov
# builder was crude, and a metric from the wrong model or data type is actively
# harmful (wrong dimension, or a Cholesky failure).
#
# None of that transfers to 2color_v10, whose per-galaxy slope enters the
# selection integral (2color_v10.stan:823). On the identity metric v10's chains
# did not reach iteration 100 in 203 CPU-minutes, with 8x throughput variation
# between chains; with a Pathfinder metric they ran at ~11.7 s/iteration in
# lockstep. The dimension objection is handled by having one builder per model
# (below), and the "wrong data type" objection by the metric being built inside
# this run's own directory from this run's own Pathfinder draws.
#
# Pathfinder itself takes minutes, not hours -- the 2 h -q regular budget is
# slack, not an estimate. The argument list is NOT Stan's default: it is the
# one that actually works on this model, taken verbatim from run_v5_3.sh:57-66
# and run_v5b_2m.sh:48-66, which is how both delivered v5 catalogues were
# really fit.
#
# Requires: output/$RUN/input.json and init_MAP.json (step 4; or step 5d for a
# config with no fixed_init).
#
# PF_INIT: which init to start Pathfinder from, default init_MAP.json. Override
# with --export=...,PF_INIT=init.json -- run_v5_3.sh needed exactly that for
# the spiral population, whose MAP start made Pathfinder stall.

set -e
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
source slurm/_model.sh   # resolves MODEL (env > config "model" > 2color)
mkdir -p slurm/logs

PF_INIT=${PF_INIT:-init_MAP.json}
STEPSIZE=${STEPSIZE:-0.1}

# One builder per model: the metric is a covariance on Stan's UNCONSTRAINED
# scale, so its dimension and the order of its rows are properties of the
# model's parameters block. Using the wrong builder silently produces a metric
# of the wrong dimension, which step6 rejects (or, worse, mis-scales).
#   make_pf_metric.py      2color      13 params
#   make_pf_metric_v10.py  2color_v10  14 params (slope_std_0 + slope_std_1)
# Both assert N_bins == 1, which the DR2 2color workflow always has.
case "$MODEL" in
    2color)     METRIC_BUILDER=make_pf_metric.py ;;
    2color_v10) METRIC_BUILDER=make_pf_metric_v10.py ;;
    *)
        echo "ERROR: no Pathfinder metric builder for MODEL=$MODEL." >&2
        echo "       Builders present: $(ls make_pf_metric*.py | tr '\n' ' ')" >&2
        echo "       A metric built for the wrong parameters block is worse" >&2
        echo "       than no metric; refusing to guess." >&2
        exit 1
        ;;
esac

echo "Step 5e: Pathfinder metric for run=$RUN model=$MODEL builder=$METRIC_BUILDER init=$PF_INIT"

./${MODEL}_g pathfinder \
    num_paths=1 num_draws=200 num_elbo_draws=10 num_psis_draws=200 \
    history_size=30 max_lbfgs_iters=100 \
    data file=output/$RUN/input.json \
    init=output/$RUN/$PF_INIT \
    output file=output/$RUN/pathfinder.csv

python3 $METRIC_BUILDER --run $RUN --stepsize $STEPSIZE

# Hard-fail rather than fall through to the identity metric. A silently absent
# metric would make step6 sample with different preconditioning from its paired
# arm, and the v10/v5 comparison this exists to support would be confounded.
if [ ! -s output/$RUN/pf_metric.json ]; then
    echo "ERROR: output/$RUN/pf_metric.json not produced (or empty)." >&2
    exit 1
fi

touch output/$RUN/.step5e_done
echo "DONE: step5e → output/$RUN/pf_metric.json"
