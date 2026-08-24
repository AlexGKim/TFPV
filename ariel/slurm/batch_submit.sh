#!/bin/bash
# Submit the full per-file 2COLOR pipeline for every config in a directory.
#
# For each configs/<dir>/<run>.json it submits a SLURM dependency chain:
#   step4 -> step6 (1 node, 4 chains/4 GPUs) -> step7 -> step8
# step4 writes init_MAP.json directly when the config sets "fixed_init"
# (frozen physical-unit init values, transformed into this run's own
# standardized coordinates) — step5d's GPU MAP-optimize job is no longer part
# of the chain in that case. step5d_map.sh remains available to run manually
# for any config that does NOT set "fixed_init".
# There is no metric-building step: every chain starts from the identity metric
# and adapts a dense one during warmup (see step6_node.sh).
#
# Files whose output/<run>/.step8_done sentinel exists are skipped (idempotent).
# Submissions are throttled so at most MAX_CONCURRENT runs' step6 nodes are
# queued at once (each run = 1 step6_node.sh job = 1 full 4-GPU node).
#
# Usage:
#   bash slurm/batch_submit.sh <config_dir> [MAX_CONCURRENT] [--debug]
# Examples:
#   bash slurm/batch_submit.sh configs/batch_v2.0.8 8
#   bash slurm/batch_submit.sh configs/batch_debug --debug
#   bash slurm/batch_submit.sh configs/batch_rlshift 8 --max-depth 8
#
# --debug          Run tiny chains on the debug GPU queue (plumbing test only).
# --max-depth N    NUTS max_depth for step6 (default: step6_node.sh's own 10).
#                  Set this when a mock set needs shallower trees to fit the
#                  24 h walltime -- see the note above the MAX_DEPTH block below.

set -e

CONFIG_DIR=""
MAX_CONCURRENT=8
DEBUG=0
MAX_DEPTH=""   # empty = inherit step6_node.sh's default (10)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      DEBUG=1; shift ;;
        --max-depth)  MAX_DEPTH="$2"; shift 2 ;;
        --*)          echo "ERROR: unknown option $1"; exit 1 ;;
        *)
            if [ -z "$CONFIG_DIR" ]; then
                CONFIG_DIR="$1"
            elif [[ "$1" =~ ^[0-9]+$ ]]; then
                MAX_CONCURRENT="$1"
            else
                echo "ERROR: unexpected argument '$1'"; exit 1
            fi
            shift ;;
    esac
done

if [ -z "$CONFIG_DIR" ] || [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: pass a config directory."
    echo "Usage: bash slurm/batch_submit.sh <config_dir> [MAX_CONCURRENT] [--debug] [--max-depth N]"
    exit 1
fi

mkdir -p slurm/logs batch
TRACKER="batch/job_tracker.csv"
if [ ! -f "$TRACKER" ]; then
    echo "run,config,step4,step6,step7,step8,debug" > "$TRACKER"
fi

# step6 queue/time override for debug mode (command-line overrides in-script SBATCH).
STEP6_OVERRIDE=()
STEP6_EXPORT_EXTRA=""
if [ "$DEBUG" = "1" ]; then
    STEP6_OVERRIDE=(-q debug -t 00:20:00)
    STEP6_EXPORT_EXTRA=",DEBUG=1"
    echo "DEBUG mode: step6 samples 15 draws at fixed stepsize (no adaptation) on the debug GPU queue."
fi

# Sampler cost is data-dependent, and a step6 that cannot finish inside the 24 h
# walltime writes NOTHING (save_warmup=false), so a whole batch can burn its
# allocation and produce no draws. The rlshift mock needed 102 s/iteration at
# max_depth=10 -- 56.8 h for 2000 iterations -- and finished in 11.9 h at
# max_depth=8, where the adapted sampler never exceeded treedepth 5 and no
# transition hit the cap (job 57376631). Measure the rate on ONE file before
# submitting a large set; if it is slow, pass --max-depth 8.
if [ -n "$MAX_DEPTH" ]; then
    STEP6_EXPORT_EXTRA="$STEP6_EXPORT_EXTRA,MAX_DEPTH=$MAX_DEPTH"
    echo "step6: max_depth=$MAX_DEPTH (overriding the default 10)."
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

    # Throttle: wait until fewer than MAX_CONCURRENT step6 nodes are queued.
    while true; do
        N_NODES=$(squeue -h -u "$USER" -n step6_node 2>/dev/null | wc -l)
        if [ "$N_NODES" -lt "$MAX_CONCURRENT" ]; then break; fi
        echo "  throttle: $N_NODES step6 nodes queued (limit $MAX_CONCURRENT), waiting 60s..."
        sleep 60
    done

    echo "Submitting chain for run=$RUN (config=$CFG)"
    JID4=$(sbatch --parsable --export=CONFIG=$CFG slurm/step4_data.sh)

    JID6=$(sbatch --parsable --dependency=afterok:$JID4 \
           "${STEP6_OVERRIDE[@]}" \
           --export=CONFIG=$CFG$STEP6_EXPORT_EXTRA \
           slurm/step6_node.sh)

    JID7=$(sbatch --parsable --dependency=afterok:$JID6 \
           --export=CONFIG=$CFG slurm/step7_diagnose.sh)
    JID8=$(sbatch --parsable --dependency=afterok:$JID7 \
           --export=CONFIG=$CFG,DEBUG=$DEBUG slurm/step8_predict.sh)

    echo "$RUN,$CFG,$JID4,$JID6,$JID7,$JID8,$DEBUG" >> "$TRACKER"
    echo "  step4=$JID4 step6=$JID6 step7=$JID7 step8=$JID8"
    n_submitted=$((n_submitted + 1))
done

echo "---"
echo "Submitted: $n_submitted   Skipped(done): $n_skipped"
echo "Tracker:   $TRACKER"
echo "Monitor:   squeue -u \$USER   |   bash slurm/batch_status.sh $CONFIG_DIR"
