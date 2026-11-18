#!/bin/bash
# Aggregate pipeline completion across every config in a batch directory.
# Reports, per run, which sentinel steps are done, and a summary count. The
# step list is 7 steps, or 8 for a run that used the opt-in step5e metric.
#
# Usage:
#   bash slurm/batch_status.sh <config_dir> [--verbose]
# Example:
#   bash slurm/batch_status.sh configs/batch_v2.0.8

CONFIG_DIR="${1:-}"
VERBOSE=0
[ "$2" = "--verbose" ] && VERBOSE=1
[ "$1" = "--verbose" ] && { VERBOSE=1; CONFIG_DIR="${2:-}"; }

if [ -z "$CONFIG_DIR" ] || [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: pass a config directory. Usage: bash slurm/batch_status.sh <config_dir> [--verbose]"
    exit 1
fi

# The default step list has no step5e: it is opt-in (batch_submit.sh
# --metric pf), and counting it unconditionally would make every finished v5
# batch -- which has no .step5e_done -- report MISSING forever. It is added
# per-run below, inside the loop, for runs that did use it.
STEPS_DEFAULT=(step4 step6_chain1 step6_chain2 step6_chain3 step6_chain4 step7 step8)
STEPS_PF=(step4 step5e step6_chain1 step6_chain2 step6_chain3 step6_chain4 step7 step8)

total=0; complete=0; inprogress=0; notstarted=0

for CFG in "$CONFIG_DIR"/*.json; do
    [ -e "$CFG" ] || { echo "No *.json configs in $CONFIG_DIR"; exit 1; }
    RUN=$(python3 -c "import json; print(json.load(open('$CFG'))['run'])")
    total=$((total + 1))

    if [ -f "output/$RUN/pf_metric.json" ] || [ -f "output/$RUN/.step5e_done" ]; then
        STEPS=("${STEPS_PF[@]}")
    else
        STEPS=("${STEPS_DEFAULT[@]}")
    fi
    NSTEPS=${#STEPS[@]}

    done_count=0
    missing=""
    for step in "${STEPS[@]}"; do
        if [ -f "output/$RUN/.${step}_done" ]; then
            done_count=$((done_count + 1))
        else
            missing="$missing $step"
        fi
    done

    if [ "$done_count" -eq "$NSTEPS" ]; then
        complete=$((complete + 1))
        state="COMPLETE"
    elif [ "$done_count" -eq 0 ]; then
        notstarted=$((notstarted + 1))
        state="NOT_STARTED"
    else
        inprogress=$((inprogress + 1))
        state="IN_PROGRESS"
    fi

    if [ "$VERBOSE" = "1" ] || [ "$state" != "COMPLETE" ]; then
        printf "  %-14s %s (%d/%d)%s\n" "$RUN" "$state" "$done_count" "$NSTEPS" \
               "${missing:+  missing:$missing}"
    fi
done

echo "---"
echo "Config dir : $CONFIG_DIR"
echo "Total runs : $total"
echo "Complete   : $complete"
echo "In progress: $inprogress"
echo "Not started: $notstarted"
