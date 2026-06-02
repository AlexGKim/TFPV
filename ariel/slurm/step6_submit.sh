#!/bin/bash
# Submit 4 independent MCMC chain jobs, then gate step7 on all completing.
# Usage: bash slurm/step6_submit.sh [configs/dr1_v6_2color.json]

set -e

CONFIG=${1:-configs/dr1_v6_2color.json}

echo "Submitting 4 chains for config=$CONFIG"
JOB_IDS=()
for CHAIN_ID in 1 2 3 4; do
    JID=$(sbatch --parsable \
          --export=CONFIG=$CONFIG,CHAIN_ID=$CHAIN_ID \
          slurm/step6_chain.sh)
    JOB_IDS+=($JID)
    echo "  Chain $CHAIN_ID submitted as job $JID"
done

# Submit step7 to run after all 4 chains succeed
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
JID7=$(sbatch --parsable \
       --dependency=afterok:$DEP \
       --export=CONFIG=$CONFIG \
       slurm/step7_diagnose.sh)
echo "Step 7 (diagnose) submitted as job $JID7, depends on: $DEP"

# Submit step8 after step7
JID8=$(sbatch --parsable \
       --dependency=afterok:$JID7 \
       --export=CONFIG=$CONFIG \
       slurm/step8_predict.sh)
echo "Step 8 (predict) submitted as job $JID8, depends on: $JID7"

echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo "  bash slurm/check_status.sh $CONFIG"
