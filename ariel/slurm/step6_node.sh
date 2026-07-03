#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 18:00:00
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
# Requires: output/$RUN/input.json, init_MAP.json, metric.json (steps 5d+5e done)
#
# DEBUG mode: set DEBUG=1 to run tiny 0+15 chains (no adaptation, fixed
# stepsize) -- see step6_chain.sh for the rationale. Submit with
# -q debug -t 0:20:00 for a fast plumbing test.

set -e

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")
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
    NUM_WARMUP=${NUM_WARMUP:-250}
    NUM_SAMPLES=${NUM_SAMPLES:-1000}
    ADAPT_ARGS="adapt save_metric=1"
    ENGINE_ARGS=""
    STEPSIZE_ARG=""
fi

echo "Step 6 (node): 4 chains for run=$RUN, one per GPU on $(hostname)"
PIDS=()
for CHAIN_ID in 1 2 3 4; do
    GPU_ID=$((CHAIN_ID - 1))
    (
        CUDA_VISIBLE_DEVICES=$GPU_ID ./2color_g sample num_warmup=$NUM_WARMUP num_samples=$NUM_SAMPLES \
            $ADAPT_ARGS \
            algorithm=hmc $ENGINE_ARGS metric=dense_e $STEPSIZE_ARG \
            metric_file=output/$RUN/metric.json \
            id=$CHAIN_ID \
            data file=output/$RUN/input.json \
            init=output/$RUN/init_MAP.json \
            output file=output/$RUN/2color_${CHAIN_ID}.csv \
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
