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
# NOT part of the standard batch chain (slurm/batch_submit.sh) for any config
# that sets "fixed_init" -- step4 (desi_data.py) writes init_MAP.json directly
# in that case, from frozen physical-unit init values transformed into this
# run's own standardized coordinates, skipping this GPU job entirely. Kept
# here for manual/standalone use on any dataset that does NOT have a
# fixed_init (e.g. deriving a brand-new fixed_init in the first place).
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
# Plain optimize (LBFGS default, unbounded) -- matches what was validated
# locally on the current free-null rank-2 model (n_null/Sc_scale/Sc_Lcorr),
# which converges normally. The previous algorithm=newton iter=20 override
# here was tuned for an older model version (its comment described the
# achromatic mode driving correlations to ~0.99, which doesn't describe the
# current free-null parameterization) and was found on NERSC to sometimes
# produce zero usable output rows for this model, silently falling through
# to the generic fallback init below instead of an actual MAP.
# '|| true' keeps set -e from aborting if the optimizer still exits nonzero.
./2color_g optimize \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv || echo "step5d: optimizer exited nonzero (using best iterate / fallback)"

python3 make_map_init.py --run $RUN || true

# Guarantee a usable, well-conditioned init_MAP.json. On degenerate data (e.g.
# mocks whose per-band noise is perfectly correlated) the near-rank-1 MAP makes
# the optimizer diverge, driving Sc_scale to 0 (finite but SINGULAR -> step6's
# Cholesky fails). If init_MAP.json is missing, non-finite, or has a degenerate
# Sc_scale, replace it with a moderate well-conditioned start (Sc_scale=0.3,
# chromatic correlation 0.7); NUTS warmup adapts from there, so a converged MAP
# is not required.
python3 - "$RUN" <<'PYEOF'
import json, math, sys, numpy as np
run = sys.argv[1]
mapf = f'output/{run}/init_MAP.json'
def bad():
    try:
        d = json.load(open(mapf)); s = d['Sc_scale']; L = d['Sc_Lcorr']
        vals = list(s) + [x for row in L for x in row]
        if not all(math.isfinite(v) for v in vals): return True
        if min(s) < 1e-2: return True          # degenerate / singular scatter
        return False
    except Exception:
        return True
if bad():
    print('step5d: init_MAP.json missing/non-finite/degenerate -> moderate fallback init')
    d = json.load(open(f'output/{run}/init.json'))
    R = np.full((2, 2), 0.7); np.fill_diagonal(R, 1.0)
    d['Sc_scale'] = [0.3, 0.3]
    d['Sc_Lcorr'] = np.linalg.cholesky(R).tolist()
    json.dump(d, open(mapf, 'w'), indent=2)
else:
    print('step5d: MAP init_MAP.json OK (Sc_scale =', json.load(open(mapf))['Sc_scale'], ')')
PYEOF

touch output/$RUN/.step5d_done
echo "DONE: step5d → output/$RUN/init_MAP.json"
