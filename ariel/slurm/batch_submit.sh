#!/bin/bash
# Submit the full per-file 2COLOR pipeline for every config in a directory.
#
# For each configs/<dir>/<run>.json it submits a SLURM dependency chain:
#   step4 -> step5d -> step6 x4 -> step7 -> step8
# (step5e is NOT included here — run it separately once per data type to build
# output/<run>/metric.json, then reseed all run dirs with that metric before calling
# this script. make_batch_configs.py copies whichever metric.json you pass via
# --metric into each output/<run>/metric.json.)
#
# Files whose output/<run>/.step8_done sentinel exists are skipped (idempotent).
# Submissions are throttled so at most MAX_CONCURRENT files' chains are in the
# queue at once (each file = 4 GPU chains).
#
# Usage:
#   bash slurm/batch_submit.sh <config_dir> [MAX_CONCURRENT] [--debug]
# Examples:
#   bash slurm/batch_submit.sh configs/batch_v0.5.7 8
#   bash slurm/batch_submit.sh configs/batch_debug --debug
#
# --debug runs tiny 10+10 chains on the debug GPU queue (plumbing test only).

set -e

CONFIG_DIR=""
MAX_CONCURRENT=8
DEBUG=0
for arg in "$@"; do
    case "$arg" in
        --debug) DEBUG=1 ;;
        *[!0-9]*|"") if [ -z "$CONFIG_DIR" ]; then CONFIG_DIR="$arg"; fi ;;
        *) MAX_CONCURRENT="$arg" ;;
    esac
done

if [ -z "$CONFIG_DIR" ] || [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: pass a config directory. Usage: bash slurm/batch_submit.sh <config_dir> [MAX_CONCURRENT] [--debug]"
    exit 1
fi

mkdir -p slurm/logs batch
TRACKER="batch/job_tracker.csv"
if [ ! -f "$TRACKER" ]; then
    echo "run,config,step4,step5d,step6_1,step6_2,step6_3,step6_4,step7,step8,debug" > "$TRACKER"
fi

# step6 queue/time override for debug mode (command-line overrides in-script SBATCH).
STEP6_OVERRIDE=()
STEP6_EXPORT_EXTRA=""
if [ "$DEBUG" = "1" ]; then
    STEP6_OVERRIDE=(-q debug -t 00:20:00)
    STEP6_EXPORT_EXTRA=",DEBUG=1"
    echo "DEBUG mode: step6 chains sample 15 draws at fixed stepsize (no adaptation) on the debug GPU queue."
fi

n_submitted=0
n_skipped=0
for CFG in "$CONFIG_DIR"/*.json; do
    [ -e "$CFG" ] || { echo "No *.json configs in $CONFIG_DIR"; exit 1; }
    RUN=$(python3 -c "import json; print(json.load(open('$CFG'))['run'])")

    if [ -f "output/$RUN/.step8_done" ]; then
        echo "skip (done): $RUN"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # Throttle: wait until fewer than MAX_CONCURRENT files' worth of chains are queued.
    while true; do
        N_CHAINS=$(squeue -h -u "$USER" -n step6_chain 2>/dev/null | wc -l)
        if [ "$N_CHAINS" -lt "$((MAX_CONCURRENT * 4))" ]; then break; fi
        echo "  throttle: $N_CHAINS step6 chains queued (limit $((MAX_CONCURRENT * 4))), waiting 60s..."
        sleep 60
    done

    echo "Submitting chain for run=$RUN (config=$CFG)"
    JID4=$(sbatch --parsable --export=CONFIG=$CFG slurm/step4_data.sh)
    JID5D=$(sbatch --parsable --dependency=afterok:$JID4 \
            --export=CONFIG=$CFG slurm/step5d_map.sh)

    CHAIN_JIDS=()
    for CHAIN_ID in 1 2 3 4; do
        JID=$(sbatch --parsable --dependency=afterok:$JID5D \
              "${STEP6_OVERRIDE[@]}" \
              --export=CONFIG=$CFG,CHAIN_ID=$CHAIN_ID$STEP6_EXPORT_EXTRA \
              slurm/step6_chain.sh)
        CHAIN_JIDS+=($JID)
    done

    DEP6=$(IFS=:; echo "${CHAIN_JIDS[*]}")
    JID7=$(sbatch --parsable --dependency=afterok:$DEP6 \
           --export=CONFIG=$CFG slurm/step7_diagnose.sh)
    JID8=$(sbatch --parsable --dependency=afterok:$JID7 \
           --export=CONFIG=$CFG,DEBUG=$DEBUG slurm/step8_predict.sh)

    echo "$RUN,$CFG,$JID4,$JID5D,${CHAIN_JIDS[0]},${CHAIN_JIDS[1]},${CHAIN_JIDS[2]},${CHAIN_JIDS[3]},$JID7,$JID8,$DEBUG" >> "$TRACKER"
    echo "  step4=$JID4 step5d=$JID5D step6=${CHAIN_JIDS[*]} step7=$JID7 step8=$JID8"
    n_submitted=$((n_submitted + 1))
done

echo "---"
echo "Submitted: $n_submitted   Skipped(done): $n_skipped"
echo "Tracker:   $TRACKER"
echo "Monitor:   squeue -u \$USER   |   bash slurm/batch_status.sh $CONFIG_DIR"
