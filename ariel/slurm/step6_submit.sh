#!/bin/bash
# Submit the 4-chain MCMC job (step6_node.sh, one node/4 GPUs), then gate
# step7 on it completing.
# Usage: bash slurm/step6_submit.sh [configs/dr1_v6_2color.json]

set -e

CONFIG=${1:-configs/batch_test.json}

echo "Submitting step6 (4 chains, 1 node) for config=$CONFIG"
JID6=$(sbatch --parsable \
       --export=CONFIG=$CONFIG \
       slurm/step6_node.sh)
echo "  step6_node submitted as job $JID6"

# Submit step7 to run after all 4 chains succeed
JID7=$(sbatch --parsable \
       --dependency=afterok:$JID6 \
       --export=CONFIG=$CONFIG \
       slurm/step7_diagnose.sh)
echo "Step 7 (diagnose) submitted as job $JID7, depends on: $JID6"

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
