#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -J step6_node
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step6_node_%j.out

# Step 6: all 4 MCMC chains in a single node-level job, one chain per physical
# GPU (CUDA_VISIBLE_DEVICES), instead of step6_chain.sh's 4 separate 1-GPU jobs.
#
# Why: sacct shows a "--gpus-per-task=1" step6_chain.sh job is allocated the
# WHOLE 4-GPU node anyway (gres/gpu:a100=4, cpu=128, node=1) -- Perlmutter's
# GPU nodes/QOS here aren't fractionally shared between jobs. So 4 separate
# step6_chain.sh jobs already reserve 4 whole nodes and use only 1 of 4 GPUs
# on each (3 idle GPUs + ~127 idle CPU cores per job, since num_threads=1
# means Stan is single-threaded per chain). Since one node already has 4 GPUs
# and comfortable headroom (257GB RAM vs. this model's tiny footprint), running
# all 4 chains as backgrounded processes within one job that already owns the
# node uses hardware that would otherwise sit idle, and collapses 4 job
# submissions into 1 -- avoiding per-job queue wait and (on constrained QOS
# like the debug queue, MaxSubmitPU=5) the chain+step5d submission cap that
# forced hand-threading single-run submissions through NERSC's debug queue.
#
# Writes the same output/$RUN/.step6_chain{1..4}_done sentinels as
# step6_chain.sh, so check_status.sh/batch_status.sh need no changes.
#
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step6_node.sh
# Requires: output/$RUN/input.json, init_MAP.json (step 4 done -- step4 writes
# init_MAP.json directly when the config sets "fixed_init"; step5d_map.sh is
# only needed for configs without a fixed_init).
# Each chain starts from the IDENTITY metric and adapts its own dense metric
# during warmup (metric=dense_e adapt save_metric=1). There is no pre-built
# metric step in this pipeline: the scripts that built one (step5e_metric.sh,
# make_metric.py) have been removed. A local CPU A/B test found that approach
# ~2.7x slower overall to obtain than identity+adapt with no quality gain, its
# builder was a crude 100-draw np.cov over a parameter list that no longer
# matches the model, and a metric from a different data type or model version
# is actively harmful (wrong dimension, or Cholesky failures).
#
# The local real-data workflow (run_dr2_onepop.sh) runs from identity too, with
# these same arguments -- the two paths sample identically, which is what lets
# mock-derived uncertainties calibrate the real measurement. Both rely on
# NUM_WARMUP=1000 being enough from identity, which the validated abacus run
# confirmed (warmup completed in ~4.4 h/chain on CPU without stalling at max
# treedepth). Pathfinder seeding was a rank-2-era workaround and is retired;
# make_pf_metric.py remains for manual experiments only.
#
# NUM_WARMUP defaults to 1000 (up from 250) and MAX_DEPTH defaults to 10
# (Stan's own default, up from a hardcoded 8): an abacus-mock run at the old
# 250/8 settings showed 14.75% of transitions hitting max treedepth and 2.5%
# divergent, alongside a systematic residual bias in the fit. Override either
# via --export=...,NUM_WARMUP=N,MAX_DEPTH=N.
#
# adapt delta=0.9 (up from Stan's default 0.8): the rank-2 S / Householder
# null-direction geometry produces tight posterior curvature that drives a
# non-trivial divergence/rejection rate at the default delta -- on NERSC this
# showed up as warmup getting stuck for hours near a boundary, spamming
# "lkj_corr_cholesky_lpdf: Random variable[2] is 0, but must be positive!"
# and never advancing past iteration 1. Matches the local DR2_TWOPOP.md fix
# (validated there against 1.9%-8.4% divergence rates at delta=0.8). delta=0.95
# was tried first but found impractically slow for the irregular population
# locally (~135 min/100 warmup iterations, ~20+ h/chain) -- its wall-clock
# cost is steeply non-linear, so 0.9 is the compromise: well above the 0.8
# default for divergence control, without the 0.95 blow-up. Override via
# --export=...,DELTA=N if a run still shows too many divergences at 0.9.
#
# DEBUG mode: set DEBUG=1 to run tiny 0+15 chains (no adaptation, fixed
# stepsize) -- see step6_chain.sh for the rationale. Submit with
# -q debug -t 0:20:00 for a fast plumbing test.

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

if [ "${DEBUG:-0}" = "1" ]; then
    NUM_WARMUP=${NUM_WARMUP:-0}
    NUM_SAMPLES=${NUM_SAMPLES:-15}
    ADAPT_ARGS="adapt engaged=0"
    ENGINE_ARGS="engine=nuts max_depth=1"
    STEPSIZE_ARG="stepsize=${STEPSIZE:-0.08}"
    echo "Step 6 (node): DEBUG mode — no adaptation, max_depth=1, fixed stepsize ${STEPSIZE:-0.08}, "\
"$NUM_SAMPLES samples (results not science-grade)"
else
    NUM_WARMUP=${NUM_WARMUP:-1000}
    NUM_SAMPLES=${NUM_SAMPLES:-1000}
    MAX_DEPTH=${MAX_DEPTH:-10}
    DELTA=${DELTA:-0.9}
    ADAPT_ARGS="adapt delta=$DELTA save_metric=1"
    ENGINE_ARGS="engine=nuts max_depth=$MAX_DEPTH"
    STEPSIZE_ARG=""
fi

# Stan's default refresh=100 prints iteration 1 and then nothing until 100, and
# save_warmup=false means the CSV stays empty for the whole warmup -- so a short
# run is completely opaque while it is in flight and you cannot tell a slow
# chain from a stuck one. Set REFRESH=1 (or 10) on a timing/debug run to get
# per-iteration progress. Unset keeps Stan's default, so production is unchanged.
REFRESH_ARG=""
[ -n "${REFRESH:-}" ] && REFRESH_ARG="refresh=$REFRESH"

echo "Step 6 (node): 4 chains for run=$RUN, one per GPU on $(hostname)"
PIDS=()
for CHAIN_ID in 1 2 3 4; do
    GPU_ID=$((CHAIN_ID - 1))
    (
        CUDA_VISIBLE_DEVICES=$GPU_ID ./2color_g sample num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES \
            $ADAPT_ARGS \
            algorithm=hmc $ENGINE_ARGS metric=dense_e $STEPSIZE_ARG \
            id=$CHAIN_ID \
            data file=output/$RUN/input.json \
            init=output/$RUN/init_MAP.json \
            output file=output/$RUN/2color_${CHAIN_ID}.csv $REFRESH_ARG \
        && touch output/$RUN/.step6_chain${CHAIN_ID}_done \
        && echo "DONE: step6 chain $CHAIN_ID (GPU $GPU_ID) → output/$RUN/2color_${CHAIN_ID}.csv"
    ) &
    PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done

if [ "$FAIL" = "1" ]; then
    echo "ERROR: one or more chains failed — check output/$RUN/.step6_chain{1..4}_done and this log"
    exit 1
fi

echo "DONE: step6 (node) → all 4 chains for run=$RUN"
