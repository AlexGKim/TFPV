#!/bin/bash
# Submit the full per-file 2COLOR pipeline for every config in a directory.
#
# For each configs/<dir>/<run>.json it submits a SLURM dependency chain:
#   step4 -> [step5e] -> step6 (1 node, 4 chains/4 GPUs) -> step7 -> step8
# step4 writes init_MAP.json directly when the config sets "fixed_init"
# (frozen physical-unit init values, transformed into this run's own
# standardized coordinates) — step5d's GPU MAP-optimize job is no longer part
# of the chain in that case. step5d_map.sh remains available to run manually
# for any config that does NOT set "fixed_init".
# By default there is no metric-building step: every chain starts from the
# identity metric and adapts a dense one during warmup (see step6_node.sh).
# --metric pf inserts slurm/step5e_metric.sh between step4 and step6 and makes
# step6 consume its pf_metric.json. 2color_v10 needs this -- on identity its
# chains did not reach iteration 100 in 203 CPU minutes.
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
#   bash slurm/batch_submit.sh configs/batch_v213_v10 1 --model 2color_v10 --metric pf
#
# --debug          Run tiny chains on the debug GPU queue (plumbing test only).
# --max-depth N    NUTS max_depth for step6 (default: step6_node.sh's own 10).
#                  Set this when a mock set needs shallower trees to fit the
#                  24 h walltime -- see the note above the MAX_DEPTH block below.
# --model NAME     Stan model for steps 5e-8 (default: each config's own
#                  "model" key, else 2color). Needs a ${NAME}_g GPU binary from
#                  slurm/compile_gpu.sh.
# --metric pf      Insert step5e (Pathfinder dense_e metric) into the chain and
#                  have step6 use it. Use it for BOTH arms of a v10/v5 pair:
#                  run_v5_3.sh documents that v5's real procedure used a
#                  Pathfinder metric and that identity was ~5x slower, so a
#                  v5 control sampled on identity against a v10 twin sampled
#                  with a metric is a confounded comparison.

set -e

CONFIG_DIR=""
MAX_CONCURRENT=8
DEBUG=0
MAX_DEPTH=""   # empty = inherit step6_node.sh's default (10)
MODEL=""       # empty = let each config's "model" key decide (slurm/_model.sh)
METRIC=""      # "pf" = build and use a Pathfinder metric (step5e)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)      DEBUG=1; shift ;;
        --max-depth)  MAX_DEPTH="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --metric)     METRIC="$2"; shift 2 ;;
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
    echo "Usage: bash slurm/batch_submit.sh <config_dir> [MAX_CONCURRENT] [--debug] [--max-depth N] [--model NAME] [--metric pf]"
    exit 1
fi

mkdir -p slurm/logs batch
TRACKER="batch/job_tracker.csv"
TRACKER_HEADER="run,config,step4,step5e,step6,step7,step8,debug,model,metric"
if [ ! -f "$TRACKER" ]; then
    echo "$TRACKER_HEADER" > "$TRACKER"
elif [ "$(head -1 "$TRACKER")" != "$TRACKER_HEADER" ]; then
    # Nothing parses this file -- it is an append-only log of job ids -- so
    # widening the schema mid-file is harmless, but say so rather than leave
    # rows of two different widths unexplained.
    echo "note: $TRACKER has the pre-step5e header; rows appended from here on" \
         "carry the wider schema ($TRACKER_HEADER)."
fi

case "$METRIC" in
    ""|pf) ;;
    *) echo "ERROR: --metric takes 'pf' (or nothing), not '$METRIC'"; exit 1 ;;
esac
if [ -n "$MODEL" ]; then
    case "$MODEL" in
        *[!A-Za-z0-9_]*)
            echo "ERROR: --model '$MODEL' is not a plain identifier ([A-Za-z0-9_]+)"; exit 1 ;;
    esac
fi

# `sbatch --export=CONFIG=$CFG` without ALL RESTRICTS the job environment, so
# MODEL and METRIC do not reach the job unless they are appended to the
# --export string -- the same reason DEBUG and MAX_DEPTH are appended below.
# This suffix goes on every step that reads either one.
COMMON_EXPORT_EXTRA=""
[ -n "$MODEL" ]  && COMMON_EXPORT_EXTRA="$COMMON_EXPORT_EXTRA,MODEL=$MODEL"
[ -n "$METRIC" ] && COMMON_EXPORT_EXTRA="$COMMON_EXPORT_EXTRA,METRIC=$METRIC"
[ -n "$MODEL" ]  && echo "model: $MODEL (overriding each config's \"model\" key)"
[ -n "$METRIC" ] && echo "metric: Pathfinder (step5e inserted between step4 and step6)"

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
    JID4=$(sbatch --parsable --export=CONFIG=$CFG$COMMON_EXPORT_EXTRA slurm/step4_data.sh)

    # step5e only exists in the chain when --metric pf was passed; otherwise
    # step6 depends directly on step4 as before.
    if [ "$METRIC" = "pf" ]; then
        JID5E=$(sbatch --parsable --dependency=afterok:$JID4 \
                --export=CONFIG=$CFG$COMMON_EXPORT_EXTRA \
                slurm/step5e_metric.sh)
        STEP6_DEP=$JID5E
    else
        JID5E="-"
        STEP6_DEP=$JID4
    fi

    JID6=$(sbatch --parsable --dependency=afterok:$STEP6_DEP \
           "${STEP6_OVERRIDE[@]}" \
           --export=CONFIG=$CFG$COMMON_EXPORT_EXTRA$STEP6_EXPORT_EXTRA \
           slurm/step6_node.sh)

    JID7=$(sbatch --parsable --dependency=afterok:$JID6 \
           --export=CONFIG=$CFG$COMMON_EXPORT_EXTRA slurm/step7_diagnose.sh)
    JID8=$(sbatch --parsable --dependency=afterok:$JID7 \
           --export=CONFIG=$CFG,DEBUG=$DEBUG$COMMON_EXPORT_EXTRA slurm/step8_predict.sh)

    echo "$RUN,$CFG,$JID4,$JID5E,$JID6,$JID7,$JID8,$DEBUG,${MODEL:-<config>},${METRIC:-identity}" >> "$TRACKER"
    echo "  step4=$JID4 step5e=$JID5E step6=$JID6 step7=$JID7 step8=$JID8"
    n_submitted=$((n_submitted + 1))
done

echo "---"
echo "Submitted: $n_submitted   Skipped(done): $n_skipped"
echo "Tracker:   $TRACKER"
echo "Monitor:   squeue -u \$USER   |   bash slurm/batch_status.sh $CONFIG_DIR"
