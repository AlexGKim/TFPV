#!/bin/bash
#SBATCH -A desi_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J step5d_map
#SBATCH --mail-type=FAIL
#SBATCH -o slurm/logs/step5d_map_%j.out

# Step 5d: MAP optimization → init_MAP.json
# Usage: sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5d_map.sh
# Requires: output/$RUN/input.json, init.json (step4 must be done)

set -e
source /global/common/software/desi/perlmutter/desiconda/20260227-2.3.1/conda/etc/profile.d/conda.sh
conda activate /global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV

module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
mkdir -p slurm/logs

echo "Step 5d: MAP optimize for run=$RUN"
# The free-covariance 2color MAP is near-singular (the achromatic mode drives the
# intrinsic correlations to ~0.99), which makes lbfgs line-search fail and Newton
# eventually overshoot into a non-PD region. Newton from the well-conditioned cold
# init.json climbs fast and cleanly for the first ~20 iterations, reaching
# essentially the MAP (validated: lp within ~25 of the true MAP), so cap it there.
# '|| true' keeps set -e from aborting if the optimizer still exits nonzero.
./2color_g optimize algorithm=newton iter=20 \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv || echo "step5d: optimizer exited nonzero (using best iterate / fallback)"

python3 make_map_init.py --run $RUN || true

# Guarantee a usable, finite init_MAP.json. If the optimizer diverged (non-finite
# S_scale/S_Lcorr) or make_map_init failed, fall back to the moderate init.json;
# NUTS warmup adapts from any reasonable start, so a converged MAP is not required.
if ! python3 -c "
import json, math, sys
d = json.load(open('output/$RUN/init_MAP.json'))
vals = list(d['S_scale']) + [x for row in d['S_Lcorr'] for x in row]
sys.exit(0 if all(math.isfinite(v) for v in vals) else 1)
" 2>/dev/null; then
    echo "step5d: init_MAP.json missing/non-finite -> falling back to init.json"
    cp output/$RUN/init.json output/$RUN/init_MAP.json
fi

touch output/$RUN/.step5d_done
echo "DONE: step5d → output/$RUN/init_MAP.json"
