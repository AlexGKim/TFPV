#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 18:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J step6_chain
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step6_chain%a_%j.out

# Step 6: Single MCMC chain. Submit 4 of these independently via step6_submit.sh.
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json,CHAIN_ID=1 slurm/step6_chain.sh
# Requires: output/$RUN/input.json, init_MAP.json (step 4 done -- step4 writes
# init_MAP.json directly when the config sets "fixed_init"; step5d_map.sh is
# only needed for configs without a fixed_init). No pre-built metric file is
# needed or accepted -- see step6_node.sh's header comment.
#
# DEBUG mode: set DEBUG=1 (e.g. --export=...,DEBUG=1) to run a tiny 10+10 chain
# that still writes the standard 2color_${CHAIN_ID}.csv so step7/step8 can consume
# it. Intended for fast end-to-end plumbing tests; submit with `-q debug -t 0:05:00`.
# num_warmup/num_samples can also be overridden directly via NUM_WARMUP/NUM_SAMPLES.

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
CHAIN_ID=${CHAIN_ID:?'CHAIN_ID must be set (1-4)'}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

# Sampling depth: full (with adaptation) by default; in DEBUG mode skip adaptation
# entirely and sample at a fixed known-good stepsize so each iteration is fast and
# predictable. (With num_warmup<20 Stan disables adaptation and falls back to a
# tiny stepsize -> every transition hits max treedepth and a single iteration can
# take >10 min. A plumbing test must avoid that.)
if [ "${DEBUG:-0}" = "1" ]; then
    NUM_WARMUP=${NUM_WARMUP:-0}
    NUM_SAMPLES=${NUM_SAMPLES:-15}
    ADAPT_ARGS="adapt engaged=0"
    # max_depth=1 caps NUTS at 2 leapfrog steps/iteration (~1-2s each), so 15
    # samples completes in ~30s regardless of stepsize. Without this, NUTS picks
    # deep trees (treedepth 4-6, 64+ steps/iter) and a 20-min debug slot isn't
    # enough for even a single sample.
    ENGINE_ARGS="engine=nuts max_depth=1"
    STEPSIZE_ARG="stepsize=${STEPSIZE:-0.08}"
    echo "Step 6: DEBUG mode — no adaptation, max_depth=1, fixed stepsize ${STEPSIZE:-0.08}, "\
"$NUM_SAMPLES samples (results not science-grade)"
else
    # These MUST match step6_node.sh's non-debug defaults. This script exists to
    # resubmit a single failed chain alongside three that step6_node.sh already
    # produced, so different settings here would yield a chain that is not
    # comparable to its siblings and would corrupt the combined posterior. They
    # previously diverged (250 warmup / max_depth=8 here vs 1000 / 10 there).
    NUM_WARMUP=${NUM_WARMUP:-1000}
    NUM_SAMPLES=${NUM_SAMPLES:-1000}
    MAX_DEPTH=${MAX_DEPTH:-10}
    # delta=0.9 (up from Stan's default 0.8): tight posterior curvature drives a
    # non-trivial divergence/rejection rate at the default delta, and can stall
    # warmup for hours near a boundary (see step6_node.sh). delta=0.95 was tried
    # and found impractically slow (steeply non-linear wall-clock cost), so 0.9
    # is the compromise. Matches the local DR2_TWOPOP.md fix.
    DELTA=${DELTA:-0.9}
    ADAPT_ARGS="adapt delta=$DELTA save_metric=1"
    ENGINE_ARGS="engine=nuts max_depth=$MAX_DEPTH"
    STEPSIZE_ARG=""
fi

# See step6_node.sh: Stan's default refresh=100 makes a short run opaque while
# in flight (iteration 1, then nothing until 100, and no CSV rows during
# warmup). REFRESH=1 or 10 gives per-iteration progress; unset keeps Stan's
# default so production is unchanged.
REFRESH_ARG=""
[ -n "${REFRESH:-}" ] && REFRESH_ARG="refresh=$REFRESH"

echo "Step 6: MCMC chain $CHAIN_ID for run=$RUN"
./2color_g sample num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES \
    $ADAPT_ARGS \
    algorithm=hmc $ENGINE_ARGS metric=dense_e $STEPSIZE_ARG \
    id=$CHAIN_ID \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color_${CHAIN_ID}.csv $REFRESH_ARG

touch output/$RUN/.step6_chain${CHAIN_ID}_done
echo "DONE: step6 chain $CHAIN_ID → output/$RUN/2color_${CHAIN_ID}.csv"
