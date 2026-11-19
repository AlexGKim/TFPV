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

source slurm/_gpu_runtime.sh   # modules + the LD_LIBRARY_PATH OpenCL fix

CONFIG=${CONFIG:-configs/batch_test.json}
RUN=$(python3 -c "import json; print(json.load(open('$CONFIG'))['run'])")
source slurm/_model.sh   # resolves MODEL (env > config "model" > 2color)
mkdir -p slurm/logs

echo "Step 5d: MAP optimize for run=$RUN model=$MODEL"
# Plain optimize (LBFGS default, unbounded) -- matches what was validated
# locally on the current free-null rank-2 model (n_null/Sc_scale/Sc_Lcorr),
# which converges normally. The previous algorithm=newton iter=20 override
# here was tuned for an older model version (its comment described the
# achromatic mode driving correlations to ~0.99, which doesn't describe the
# current free-null parameterization) and was found on NERSC to sometimes
# produce zero usable output rows for this model, silently falling through
# to the generic fallback init below instead of an actual MAP.
# '|| true' keeps set -e from aborting if the optimizer still exits nonzero.
./${MODEL}_g optimize \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv || echo "step5d: optimizer exited nonzero (using best iterate / fallback)"

python3 make_map_init.py --run $RUN || true

# Guarantee a usable, well-conditioned init_MAP.json. On degenerate data the
# MAP optimizer can drive the rank-1 loading w to ~0 -- finite, but a SINGULAR
# intrinsic covariance S = w w^T, so step6's Cholesky fails. If init_MAP.json
# is missing, non-finite, or has a collapsed |w|, replace it with a moderate
# well-conditioned start; NUTS warmup adapts from there, so a converged MAP is
# not required.
#
# This guard used to test Sc_scale and Sc_Lcorr, from the retired rank-2
# V Sigma_c V^T / free-null parameterization. The current model declares
# neither, so the lookup raised KeyError, the `except` branch declared EVERY
# MAP bad, and the script silently overwrote its own MAP with the plain
# polyfit init from init.json -- while logging "missing/non-finite/degenerate",
# which was never the real reason. It also injected Sc_scale/Sc_Lcorr keys no
# current model declares. Rewritten against the parameters the model has.
python3 - "$RUN" <<'PYEOF'
import json, math, sys
run = sys.argv[1]
mapf = f'output/{run}/init_MAP.json'
# A collapsed loading is the rank-1 analogue of the old Sc_scale -> 0 test:
# S = w w^T becomes singular as |w| -> 0.
W_NORM_MIN = 0.05
def bad():
    try:
        d = json.load(open(mapf))
    except Exception as e:
        print(f'step5d: cannot read {mapf} ({type(e).__name__}) -> fallback')
        return True
    w = d.get('w')
    if w is None:
        print('step5d: init_MAP.json has no w -> fallback')
        return True
    scalars = [v for v in d.values() if isinstance(v, (int, float))]
    if not all(math.isfinite(float(v)) for v in list(w) + scalars):
        print('step5d: init_MAP.json has non-finite values -> fallback')
        return True
    wn = math.sqrt(sum(float(x) ** 2 for x in w))
    if wn < W_NORM_MIN:
        print(f'step5d: |w| = {wn:.4g} < {W_NORM_MIN} (singular S) -> fallback')
        return True
    print(f'step5d: MAP init_MAP.json OK (|w| = {wn:.4g})')
    return False
if bad():
    d = json.load(open(f'output/{run}/init.json'))
    # Asymmetric by design: a symmetric w = [c,c,c] gives a degenerate gradient
    # for S = w w^T and can stall the optimizer. Same values desi_data.py uses.
    d['w'] = [0.15, 0.20, 0.25]
    json.dump(d, open(mapf, 'w'), indent=2)
    print('step5d: wrote moderate fallback init (w = [0.15, 0.20, 0.25])')
PYEOF

touch output/$RUN/.step5d_done
echo "DONE: step5d → output/$RUN/init_MAP.json"
