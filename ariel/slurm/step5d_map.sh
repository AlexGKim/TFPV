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

# Guarantee a usable, well-conditioned init_MAP.json. On degenerate data (e.g.
# mocks whose per-band noise is perfectly correlated) the near-rank-1 MAP makes
# the optimizer diverge, driving S_scale to 0 (finite but SINGULAR -> step6's
# Cholesky fails). If init_MAP.json is missing, non-finite, or has a degenerate
# S_scale, replace it with a moderate well-conditioned start (S_scale=0.3,
# intrinsic correlation 0.7); NUTS warmup adapts from there, so a converged MAP
# is not required.
python3 - "$RUN" <<'PYEOF'
import json, math, sys, numpy as np
run = sys.argv[1]
mapf = f'output/{run}/init_MAP.json'
def bad():
    try:
        d = json.load(open(mapf)); s = d['S_scale']; L = d['S_Lcorr']
        vals = list(s) + [x for row in L for x in row]
        if not all(math.isfinite(v) for v in vals): return True
        if min(s) < 1e-2: return True          # degenerate / singular scatter
        return False
    except Exception:
        return True
if bad():
    print('step5d: init_MAP.json missing/non-finite/degenerate -> moderate fallback init')
    d = json.load(open(f'output/{run}/init.json'))
    R = np.full((3, 3), 0.7); np.fill_diagonal(R, 1.0)
    d['S_scale'] = [0.3, 0.3, 0.3]
    d['S_Lcorr'] = np.linalg.cholesky(R).tolist()
    json.dump(d, open(mapf, 'w'), indent=2)
else:
    print('step5d: MAP init_MAP.json OK (S_scale =', json.load(open(mapf))['S_scale'], ')')
PYEOF

touch output/$RUN/.step5d_done
echo "DONE: step5d → output/$RUN/init_MAP.json"
