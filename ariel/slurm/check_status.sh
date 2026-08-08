#!/bin/bash
# Report which pipeline steps have completed for a given config.
# Usage: bash slurm/check_status.sh [configs/dr1_v6_2color.json]

CONFIG=${1:-configs/dr1_v6_2color.json}
RUN=$(python -c "import json; print(json.load(open('$CONFIG'))['run'])")

echo "Status for run: $RUN"
echo "---"

# step5d and step5e are deliberately absent: step5d is skipped whenever the
# config sets "fixed_init" (step4 writes init_MAP.json itself) and step5e is
# vestigial (step6 never reads metric.json). Listing either made a fully
# complete run report MISSING forever. Matches batch_status.sh.
STEPS=(step4 step6_chain1 step6_chain2 step6_chain3 step6_chain4 step7 step8)
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
    echo "To resubmit a missing step, e.g. chain 2:"
    echo "  sbatch --export=CONFIG=$CONFIG,CHAIN_ID=2 slurm/step6_chain.sh"
fi
