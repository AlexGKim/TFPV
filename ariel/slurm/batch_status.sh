#!/bin/bash
# Aggregate pipeline completion across every config in a batch directory.
# Reports, per run, which of the 7 sentinel steps are done, and a summary count.
#
# Usage:
#   bash slurm/batch_status.sh <config_dir> [--verbose]
# Example:
#   bash slurm/batch_status.sh configs/batch_v0.5.7

CONFIG_DIR="${1:-}"
VERBOSE=0
[ "$2" = "--verbose" ] && VERBOSE=1
[ "$1" = "--verbose" ] && { VERBOSE=1; CONFIG_DIR="${2:-}"; }

if [ -z "$CONFIG_DIR" ] || [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: pass a config directory. Usage: bash slurm/batch_status.sh <config_dir> [--verbose]"
    exit 1
fi

STEPS=(step4 step6_chain1 step6_chain2 step6_chain3 step6_chain4 step7 step8)
NSTEPS=${#STEPS[@]}

total=0; complete=0; inprogress=0; notstarted=0

for CFG in "$CONFIG_DIR"/*.json; do
    [ -e "$CFG" ] || { echo "No *.json configs in $CONFIG_DIR"; exit 1; }
    RUN=$(python3 -c "import json; print(json.load(open('$CFG'))['run'])")
    total=$((total + 1))

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
