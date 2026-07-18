# DR2 Two-Color Run: Spiral / Irregular (Steps 1–8, self-contained)

This is the complete, self-contained command sequence for the DR2
two-population ("2color") TFR fit — the exact procedure that produced the
successful `DR2_v0_2color_spiral` run, merged from [DR1.md](DR1.md)'s
selection steps (1–3b) and [2COLOR.md](2COLOR.md)'s fit/predict steps (4–8),
plus two things found while actually running it that aren't reflected
upstream yet:

- Step 6 runs from the **identity metric directly** — no pre-built-metric
  "Step 5e" short run. An empirical head-to-head on this exact spiral
  dataset showed the short-run-then-warm-start approach was **~2.7x
  *slower* overall** than just letting Step 6 adapt from scratch (8344s for
  the short run + 5717s for Step 6's warmup = 14,061s, vs. 5232s running
  Step 6 directly from identity — both reached an equivalent final stepsize,
  ~0.22 vs ~0.21). The short run's `metric.json` was built from only 100
  post-warmup draws for 13+ parameters — too few for a reliable covariance
  estimate, so it wasn't actually a better starting point than identity.
  [2COLOR.md](2COLOR.md) still documents the short-run approach for
  reference/other datasets where it may behave differently; this doc omits
  it.
- Step 6 uses the **locally-run parallel-chains pattern** (backgrounded
  processes, one per core) as the primary method, not the sequential
  `num_chains=4` command — this is what was actually used.

This doc supersedes DR1.md Steps 1–3b + 2COLOR.md Steps 4–8 for the DR2
two-population 2color workflow specifically. DR1.md and 2COLOR.md remain
correct for other workflows (single-population runs, non-2color models, the
DR1 dataset) and are linked here as background only — you shouldn't need to
open them to run this procedure.

See also: [run_dr2.sh](run_dr2.sh), a script that automates everything below
end-to-end for one population per invocation (pausing only for Step 3's
interactive fiducial choice).

---

## Two-population setup

The official DR2 FITS file is fit as **two independent populations**, each
with its own selection, its own MCMC, and its own prediction run:

| Population | Predicate | Rows (DR2_v0 spiral run) |
|---|---|---|
| Spiral | `MORPHTYPE_AI == 'Spiral'` and not VI-rejected | 23,422 |
| Irregular | `MORPHTYPE_AI == 'Irregular'` and not VI-rejected | 8,409 |

`JOHN_VI` is a masked column whose only unmasked value is `'reject'`, so
"not VI-rejected" means `JOHN_VI.mask == True`. It applies to **both**
populations.

The population cut is applied **once, upstream**, by pre-filtering the
catalog into one FITS file per population, so the rest of the pipeline is an
ordinary single-FITS run and never needs to know about the population split:

```bash
python make_population_subsets.py --input data/<new_official_dr2>.fits
# writes data/<new_official_dr2>_spiral.fits, data/<new_official_dr2>_irregular.fits
```

Pass `--force` to overwrite existing subset files, `--outdir` to write
elsewhere (default: alongside `--input`).

---

## Setup (per population, env vars)

Run everything below **once per population**, exporting these first. RUN and
CONFIG are derived from the parent FITS filename so a new official file
never collides with an earlier run's output:

```bash
export PARENT_FITS=data/<new_official_dr2>.fits            # the file just split above
export POPULATION=spiral                                    # or irregular
export FITS=${PARENT_FITS%.fits}_${POPULATION}.fits
export RUN=$(basename ${PARENT_FITS%.fits})_2color_${POPULATION}
export CONFIG=configs/$(echo $RUN | tr '[:upper:]' '[:lower:]').json
```

(`run_dr2.sh` computes these the same way — see its `--fits`/`--population`
flags.)

---

## Step 1: Estimate the core distribution

```bash
python selection_ellipse.py --file $FITS --run $RUN --source DESI \
    --z_obs_min 0.01 --z_obs_max 0.065 --haty_min -23 --haty_max -18
```

Inspect:

```bash
open output/$RUN/selection_ellipse.png
```

---

## Step 2: MLE fit and pull-profile diagnostic

```bash
python select_v2.py --run $RUN --fits_file $FITS --exe ./tophat \
    --z_obs_min 0.01 --z_obs_max 0.065
```

`select_v2.py` always uses the `tophat` binary here regardless of which
final model you'll fit later — this step is a diagnostic 2D (x̂, ŷ) MLE fit
to produce the pull-profile plot, not the full quadrivariate 2color model
(passing `--exe ./2color` fails: 2color.stan's data block declares
variables like `z` that this step's data prep never populates).

Inspect the pull profile — this is what informs Step 3's choices:

```bash
open output/$RUN/select_v2_pull.png
```

---

## Step 3: Set fiducial selection criteria — the one interactive step

**This is the only step requiring a human judgment call.** Look at
`select_v2_pull.png` from Step 2 before running this.

```bash
python set_fiducial.py --run $RUN
```

Prompts for `n_sigma_perp`, `haty_min` (bright-end magnitude limit),
`haty_max` (dim-end magnitude limit), `z_obs_min`, `z_obs_max`. Writes
`output/$RUN/select_v2_fiducial.json`. Reference values used for the DR2_v0
spiral/irregular runs: `haty_min≈-22.0`, `haty_max≈-17.8` (spiral) /
`-17.5` (irregular), `z_obs_min=0.01`, `z_obs_max=0.065`, `n_sigma_perp=3.0`
— your new official file's optimal values may differ; choose based on the
plot, not these numbers.

Inspect the result:

```bash
open output/$RUN/select_v2_fiducial_pull.png
```

---

## Step 3b: Export run config

```bash
python export_config.py --run $RUN --out $CONFIG
```

Prompts for the remaining pipeline settings — stable values for this
workflow: `exe=2color`, `source=DESI`, `model=2color`, `n_sigma=3.0`.

`export_config.py` tries to pick up `fits_file` from `output/$RUN/config.json`,
but that file is only written by `desi_data.py` (Step 4) — which hasn't run
yet at this point unless you already ran it once before (e.g. via flags).
If `output/$RUN/config.json` doesn't exist yet, `export_config.py` silently
falls back to a hardcoded placeholder (`data/DESI-DR1_TF_pv_cat_v15.fits`),
which is wrong. **Always force-correct it after this step:**

```bash
python -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['fits_file'] = '$FITS'
json.dump(cfg, open('$CONFIG', 'w'), indent=2)
print('fits_file set to', cfg['fits_file'])
"
```

Commit the resulting `$CONFIG` to git.

**Optional**: add `"train_fraction": 0.4` to `$CONFIG` (see 2COLOR.md's
"Training sample size" section) to hold out an analysis sample distinct from
training; add `"dust_pickle": "data/<...>.pickle"` if a dust correction
pickle applies to this dataset. Neither is prompted for by `export_config.py`
— edit `$CONFIG` directly after this step if needed.

---

## Step 4: Prepare data

```bash
python desi_data.py --config $CONFIG
```

Inspect:

```bash
open output/$RUN/data.png
```

---

## Step 5: Compile Stan model (once)

```bash
cd ../../cmdstan
make ../TFPV/ariel/2color
cd ../TFPV/ariel
```

---

## Step 5d: Find MAP estimate (init_MAP.json)

```bash
./2color optimize \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv

python3 make_map_init.py --run $RUN
```

`make_map_init.py` floors any `sigma_int_*`/`log_sigma_int_*` parameter the
MAP drove near its 0 boundary (default floor 0.01, via `--sigma-floor`).

---

## Step 6: Run MCMC sampling — 4 parallel chains, identity metric

Runs directly from the identity metric with in-warmup dense-metric
adaptation (`adapt`, Stan's default) — **no pre-built metric / "Step 5e"**,
per the empirical result at the top of this doc. Launches 4 single-chain
processes in the background, one per core (adjust if you have fewer than 4
free cores):

```bash
PIDS=()
for CHAIN_ID in 1 2 3 4; do
    ./2color sample num_warmup=1000 num_samples=1000 \
        adapt save_metric=1 \
        algorithm=hmc engine=nuts max_depth=10 metric=dense_e \
        id=$CHAIN_ID \
        data file=output/$RUN/input.json \
        init=output/$RUN/init_MAP.json \
        output file=output/$RUN/2color_${CHAIN_ID}.csv &
    PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
done
[ "$FAIL" = "1" ] && echo "ERROR: one or more chains failed" || echo "DONE: all 4 chains"
```

`num_warmup=1000` (up from 250) and `max_depth=10` (Stan's default, up from
8): an initial abacus-mock experiment at 250/8 showed a non-trivial fraction
of transitions hitting the max-treedepth cap (14.75%) and diverging (2.5%),
alongside a systematic residual bias in the fit — raising warmup gives the
dense-metric adaptation more iterations to converge, and removing the
treedepth cap lets NUTS actually reach its natural trajectory length instead
of being cut off (at the cost of up to 1023 leapfrog steps/iteration in the
worst case, vs. 255 at depth 8, so this is meaningfully slower per iteration
— expect substantially longer runs than the timing note below, which
predates this change). `id=$CHAIN_ID` is required (distinct RNG seed/offset
per chain). This produces `2color_1.csv` … `2color_4.csv` in `output/$RUN/`,
matching the `2color_?.csv` glob used downstream.

For the spiral DR2_v0 run (at the original num_warmup=250/max_depth=8
settings): warmup ≈87 min per chain (all 4 run concurrently), so ~1.5–2
hours total including sampling — actual timing at num_warmup=1000/max_depth=10
will vary with N and core availability, but expect several times longer.

If you have fewer than 4 free cores, only run as many chains concurrently as
you have cores, or fall back to the sequential single-invocation form:

```bash
./2color sample num_warmup=1000 num_samples=1000 num_chains=4 \
    adapt save_metric=1 \
    algorithm=hmc engine=nuts max_depth=10 metric=dense_e \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color.csv
```
(slower — chains run one after another, not in parallel, since the CPU
binary isn't built with `STAN_THREADS`.)

---

## Step 7: Diagnose and visualize

```bash
../../cmdstan/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt

python corner.py --run $RUN --model 2color
```

Inspect:

```bash
cat output/$RUN/stansummary.txt
open output/$RUN/2color.png
```

Key parameters to check:
- `slope` — TFR slope
- `Sc_scale.1/2` — chromatic scatter scales
- `Sc_Lcorr.2.1` — the single free entry of the 2×2 chromatic correlation Cholesky
- `delta_c`, `delta_g` — color–velocity slopes (mean structure)
- `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` — band k-corrections

```bash
python explore_residuals.py --config $CONFIG --kind 2color
```

Residual plots land in `output/$RUN/explore_residuals/`.

---

## Step 8: Predict absolute magnitudes

```bash
python color_predict.py --config $CONFIG --model 2color --xonly
```

Key outputs:

| File | Description |
|------|-------------|
| `output/$RUN/color_catalog.fits` | DESI catalog with MU_TF, LOGDIST, MAIN, ANALYSIS (full model) |
| `output/$RUN/color_xonly_catalog.fits` | Same, x-only |
| `output/$RUN/color_cov.h5` | (G,G) covariance HDF5, datasets `cov`, `analysis` (full model) |
| `output/$RUN/color_xonly_cov.h5` | Same, x-only |

`MAIN` marks every galaxy passing selection cuts (training + analysis
union); `ANALYSIS` marks the non-training subset. Recover the analysis-only
covariance without re-deriving masks:

```python
import h5py, numpy as np
with h5py.File("output/$RUN/color_xonly_cov.h5", "r") as f:
    cov = f["cov"][:]
    analysis = f["analysis"][:]
cov_analysis = cov[np.ix_(analysis, analysis)]
```

Faster variants (skip covariance and/or catalog steps) and full column/output
documentation: see [2COLOR.md](2COLOR.md) Step 8 / Step 8 variants.

---

## File Reference

| File | Purpose |
|------|---------|
| [run_dr2.sh](run_dr2.sh) | Automates Steps 1–8 above, one population per invocation |
| [2COLOR.md](2COLOR.md) | Background: full 2color model docs, single-population form, Step 5e (not used here) |
| [DR1.md](DR1.md) | Background: original selection workflow this doc's Steps 1–3b are adapted from |
| `2color.stan` | Stan model: quadrivariate TFR with two independent color factors |
| `make_population_subsets.py` | Splits an official FITS file into per-population subsets |
| `color_predict.py --model 2color` | Posterior predictive computation |
| [Predict.md](Predict.md) | Full prediction-step argument reference |
