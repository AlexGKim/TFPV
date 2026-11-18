#!/bin/bash
# Report which pipeline steps have completed for a given config.
# Usage: bash slurm/check_status.sh [configs/dr1_v6_2color.json]

CONFIG=${1:-configs/dr1_v6_2color.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")

echo "Status for run: $RUN"
echo "---"

# step5d is deliberately absent: it is skipped whenever the config sets
# "fixed_init" (step4 writes init_MAP.json itself), and listing it made a fully
# complete run report MISSING forever. step5e is absent by default for the same
# reason -- it is opt-in (--metric pf) -- but is added below for runs that did
# use it. Matches batch_status.sh.
STEPS=(step4 step6_chain1 step6_chain2 step6_chain3 step6_chain4 step7 step8)
# step5e is counted ONLY for runs that actually used it. Adding it
# unconditionally is the exact bug the comment above records: these scripts are
# run against finished v5 batches that have no .step5e_done, and every one of
# them would report MISSING forever. A run that used the Pathfinder metric
# leaves pf_metric.json (or .step5e_done) behind, so that is the gate.
if [ -f "output/$RUN/pf_metric.json" ] || [ -f "output/$RUN/.step5e_done" ]; then
    STEPS=(step4 step5e step6_chain1 step6_chain2 step6_chain3 step6_chain4 step7 step8)
fi

ALL_DONE=true
for step in "${STEPS[@]}"; do
    if [ -f "output/$RUN/.${step}_done" ]; then
        echo "  DONE    $step"
    else
        echo "  MISSING $step"
        ALL_DONE=false
    fi
done

echo "---"
if $ALL_DONE; then
    echo "All steps complete."
else
    # MODEL and METRIC must be carried on --export: `--export=CONFIG=...`
    # without ALL restricts the job environment, so a resubmitted chain that
    # omitted them would silently run 2color on the identity metric and produce
    # a chain that is not comparable to its three siblings. Both are resolved
    # here from the same places the job would resolve them -- the config's
    # "model" key, and whether this run left a pf_metric.json behind -- so the
    # printed command is one you can paste.
    RESUB_MODEL=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('model') or '2color')")
    RESUB_EXPORT="CONFIG=$CONFIG,CHAIN_ID=2,MODEL=$RESUB_MODEL"
    if [ -f "output/$RUN/pf_metric.json" ] || [ -f "output/$RUN/.step5e_done" ]; then
        RESUB_EXPORT="$RESUB_EXPORT,METRIC=pf"
    fi
    echo "To resubmit a missing step, e.g. chain 2:"
    echo "  sbatch --export=$RESUB_EXPORT slurm/step6_chain.sh"
fi
