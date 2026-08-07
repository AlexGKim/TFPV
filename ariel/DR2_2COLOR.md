# DR2 Two-Color Run: Spiral / Irregular (Steps 1–8, self-contained)

This is the complete, self-contained command sequence for the DR2
two-population ("2color") TFR fit — the exact procedure that produced the
successful `DR2_v0_2color_spiral` run, merged from [DR1.md](DR1.md)'s
selection steps (1–3b) and [2COLOR.md](2COLOR.md)'s fit/predict steps (4–8),
plus two things found while actually running it that aren't reflected
upstream yet:

- Step 6 runs from a **Pathfinder-built dense warmup metric** (Step 5e below),
  not the identity metric. The 2color posterior is badly conditioned (~10^5
  spread in parameter SDs), so from the identity metric HMC warmup collapses to
  a tiny stepsize at max treedepth (~1023 leapfrog steps/iteration,
  ~60 s/iteration, ~4+ h/chain of warmup). Pathfinder estimates the full
  posterior covariance in ~1 minute; seeding `dense_e` with it removes the
  mis-conditioning from iteration 1 and keeps warmup off the treedepth cap
  (~3 s/iteration). This is **not** the old short-MCMC-run "Step 5e" that
  [2COLOR.md](2COLOR.md) documents and rejects — that built a covariance from
  only ~100 post-warmup draws (too few, ~2.7x slower overall). Pathfinder is
  cheap and purpose-built for this, so it is a default step here.
- The intrinsic (y,z,g) scatter is a **rank-1 covariance `S = w wᵀ`** (a single
  loading vector `w`), not the earlier rank-2 `V Σ_c Vᵀ` with a free
  `unit_vector` null direction. The data support one scatter axis (the rank-2
  second scale collapsed to ~0), and rank-1 also removes the warmup *funnel*:
  the rank-2 vanishing second scale and sphere-constrained null direction
  created nonlinear curvature that no fixed metric could precondition, so even
  a dense Pathfinder metric funneled back to max treedepth mid-warmup under
  rank-2; under rank-1 it does not.
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

| Population | Predicate | Rows (DR2_TF v5 run) |
|---|---|---|
| Spiral | `MORPHTYPE == 'Spiral'` and not VI-rejected | 24,813 |
| Irregular | `MORPHTYPE == 'Irregular'` and not VI-rejected | 6,555 |

The population split is on the authoritative **`MORPHTYPE`** column, *not*
`MORPHTYPE_AI`: the AI classification disagrees for a subset (it tags ~1,600
`MORPHTYPE == 'Spiral'` galaxies as Irregular/Undecided/etc.), which would
wrongly drop them from the spiral population. `JOHN_VI` is a masked column whose
only unmasked value is `'reject'`, so "not VI-rejected" means
`JOHN_VI.mask == True`. It applies to **both** populations.

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

**Required for DR2: `dust_pickle`.** The internal-dust slope uncertainty
`d_err_r` sets the dust off-diagonal term of the step-8 covariance
(`v_dust = d_err_r × (BA − 1)`, see `color_predict.py::_systematic_offdiag_terms`).
For the DR2 catalogs it must come from the loa internal-dust MCMC, **not** the
built-in iron default:

```json
"dust_pickle": "data/loa_internalDust_nokcorr_mcmc.pickle"
```

| Source | `d_err_r` |
|---|---|
| Built-in default (`_D_ERR_R`, iron) — used when the key is **absent** | 0.17680325 |
| `data/loa_internalDust_nokcorr_mcmc.pickle` (loa) — correct for DR2 | 0.21734862 |

Getting this wrong is silent: with the key absent, `color_predict.py` simply
uses the iron default and prints nothing. Both DR2 v5 populations were
originally run this way, leaving the dust contribution low by a factor of
`(0.2173/0.1768)² ≈ 1.5`. **Verify it took effect** — step 8 prints a
`Loaded d_err_r = 0.21734862 mag from …` line whenever the pickle is read, and
its absence from the log means the default was used:

```bash
grep "Loaded d_err_r" output/$RUN/step8.log
```

**Optional**: add `"train_fraction": 0.4` to `$CONFIG` (see 2COLOR.md's
"Training sample size" section) to hold out an analysis sample distinct from
training. Neither `dust_pickle` nor `train_fraction` is prompted for by
`export_config.py` — add them to `$CONFIG` by hand. As of the fix in this
repo, `export_config.py` *preserves* any key it does not itself manage on
re-export (and prints what it kept), so they survive a repeat of this step;
earlier versions dropped them silently, which is how the v5 covariances ended
up on the iron default.

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

## Step 5e: Build a warmup metric from Pathfinder

```bash
./2color pathfinder num_paths=1 history_size=30 max_lbfgs_iters=100 \
    num_draws=200 num_elbo_draws=10 num_psis_draws=200 \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/pathfinder.csv

python3 make_pf_metric.py --run $RUN
```

Pathfinder inits from `init.json` (the pre-MAP starting point), **not**
`init_MAP.json`: starting Pathfinder's L-BFGS exactly at the MAP mode (near-zero
gradient, very flat) can make every L-BFGS iteration fail with ELBO `-inf`
("None of the LBFGS iterations completed successfully" — seen on the spiral
population). Starting from `init.json` lets Pathfinder run its own optimization
normally.

The 2color posterior is badly conditioned — parameter standard deviations span
~10^5 across the 13 sampling dimensions (magnitudes ~1e-3, band k-corrections
~1e-1). Started from the identity metric, HMC fights this mis-conditioning for
most of warmup: the stepsize collapses to ~2e-4 and *every* transition hits the
max-treedepth cap (~1023 leapfrog steps/iteration, ~60 s/iteration), taking
~4+ h/chain of warmup before adaptation converges.

Pathfinder (Stan's L-BFGS variational method) produces an approximate posterior
in ~1 minute. `make_pf_metric.py` reads its draws, transforms each sampling
parameter to Stan's unconstrained scale (logit for the bounded `slope_std`,
`intercept_std`, `sigma_int_x`; the rank-1 loading `w` and the mean/k-correction
params are unconstrained-native), and writes a full **13×13 dense** `dense_e`
metric — the exact posterior covariance estimate, correlations included — to
`output/$RUN/pf_metric.json`. Seeding Step 6 with this removes the
mis-conditioning from iteration 1; warmup stays off the max-treedepth cap
(treedepth ~2–6, ~3 s/iteration) instead of funneling. (The rank-1 `S = w w^T`
parameterization is what makes the metric exact: every sampling dimension
transforms cleanly, unlike the earlier rank-2 `unit_vector` null direction,
which had a sphere-constrained radial dimension that no fixed metric could
precondition — the source of the warmup funnel this whole step exists to
avoid.) `make_pf_metric.py` assumes `N_bins == 1` (this workflow) and errors out
otherwise.

---

## Step 6: Run MCMC sampling — 4 parallel chains, Pathfinder metric

Runs from the Pathfinder-seeded metric (Step 5e) with in-warmup dense-metric
adaptation (`adapt`, Stan's default). Launches 4 single-chain processes in the
background, one per core (adjust if you have fewer than 4 free cores):

```bash
PIDS=()
for CHAIN_ID in 1 2 3 4; do
    ./2color sample num_warmup=1000 num_samples=1000 \
        adapt delta=0.9 save_metric=1 \
        algorithm=hmc engine=nuts max_depth=10 \
        metric=dense_e metric_file=output/$RUN/pf_metric.json \
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
predates this change). `adapt delta=0.9` (up from Stan's default 0.8): the
rank-2 `S` / Householder null-direction geometry produces tight posterior
curvature that drove a non-trivial divergence rate at the default (1.9% on
DR2_v0 spiral, 8.4% on DR2_TF_spirals_v5 spiral); a higher target acceptance
statistic shrinks the adapted stepsize and suppresses those divergences. This
comes at a real wall-clock cost, and the cost is steeply non-linear in the
irregular population: at `delta=0.95` the irregular warmup was so slow it was
impractical locally (~135 min per 100 warmup iterations, i.e. ~20+ h/chain),
so `0.9` is the compromise used here — well above the 0.8 default for
divergence control, without the 0.95 blow-up. If `diagnose.txt` still reports
too many divergences at `0.9`, raise toward `0.95`, but budget for the much
longer run (or move to slurm/NERSC). `id=$CHAIN_ID` is required (distinct RNG
seed/offset per chain). This produces `2color_1.csv` … `2color_4.csv` in `output/$RUN/`,
matching the `2color_?.csv` glob used downstream.

For the spiral DR2_v0 run (at the original num_warmup=250/max_depth=8
settings): warmup ≈87 min per chain (all 4 run concurrently), so ~1.5–2
hours total including sampling — actual timing at num_warmup=1000/max_depth=10
will vary with N and core availability, but expect several times longer.

If you have fewer than 4 free cores, only run as many chains concurrently as
you have cores, or fall back to the sequential single-invocation form:

```bash
./2color sample num_warmup=1000 num_samples=1000 num_chains=4 \
    adapt delta=0.9 save_metric=1 \
    algorithm=hmc engine=nuts max_depth=10 \
    metric=dense_e metric_file=output/$RUN/pf_metric.json \
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
- `w_norm` — magnitude of the rank-1 intrinsic-scatter loading (dominant scatter, mag)
- `w.1/2/3` — the rank-1 loading vector (S = w wᵀ); `w_report` gives the sign-fixed direction
- `scatter_angle_deg` — angle between the scatter direction and the achromatic
  axis; ~90° means purely chromatic scatter (achromatic/PV direction carries none)
- `delta_c`, `delta_g` — color–velocity slopes (mean structure)
- `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` — band k-corrections

---

## Step 8: Predict absolute magnitudes

```bash
python color_predict.py --config $CONFIG --model 2color
```

x-only (conditioning on x̂ and z_obs only, marginalizing ẑ and ĝ) is always
computed — it's the default model here, since it doesn't depend on the
z/g-band k-corrections and D-matrix coupling that the full model needs. On
the abacus-mock experiment, the full model showed a systematic ~0.05 mag
distance-modulus bias correlated with intrinsic dust (`A_INT_{G,R,Z}`); the
x-only model's bias was roughly half that (~0.027 mag) and *not*
significantly correlated with dust — pointing at the full model's z/g
machinery as the main (though not sole) source. Add `--full` to additionally
compute the full quadrivariate model's outputs for comparison:

```bash
python color_predict.py --config $CONFIG --model 2color --full
```

Key outputs (always written):

| File | Description |
|------|-------------|
| `output/$RUN/color_xonly_catalog.fits` | DESI catalog with MU_TF, LOGDIST, MAIN, ANALYSIS (x-only) |
| `output/$RUN/color_xonly_cov.h5` | (G,G) covariance HDF5, datasets `cov`, `analysis` (x-only) |
| `output/$RUN/color_grid_xonly.png` / `_full.png` | Mean residual on (x̂, ŷ) grid, x-only (MAIN / full sample) |

With `--full`, additionally:

| File | Description |
|------|-------------|
| `output/$RUN/color_catalog.fits` | Same, full model |
| `output/$RUN/color_cov.h5` | Same, full model |
| `output/$RUN/color_grid.png` / `_full.png` | Same, full model |

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

```bash
python explore_residuals.py --config $CONFIG --kind 2color
```

Residual plots land in `output/$RUN/explore_residuals/`.

---

## File Reference

| File | Purpose |
|------|---------|
| [run_dr2.sh](run_dr2.sh) | Automates Steps 1–8 above, one population per invocation |
| [2COLOR.md](2COLOR.md) | Background: full 2color model docs, single-population form, the old short-MCMC-run "Step 5e" (superseded here by Pathfinder) |
| [DR1.md](DR1.md) | Background: original selection workflow this doc's Steps 1–3b are adapted from |
| `2color.stan` | Stan model: quadrivariate TFR with two independent color factors |
| `make_map_init.py` | Step 5d: converts the MAP `optimize.csv` to `init_MAP.json` |
| `make_pf_metric.py` | Step 5e: builds the Pathfinder warmup metric `pf_metric.json` |
| `make_population_subsets.py` | Splits an official FITS file into per-population subsets |
| `color_predict.py --model 2color` | Posterior predictive computation |
| `data/loa_internalDust_nokcorr_mcmc.pickle` | Internal-dust MCMC for DR2; `"dust_pickle"` in `$CONFIG` points here to give `d_err_r = 0.21734862` instead of the iron default `0.17680325` (see Step 3b) |
| [Predict.md](Predict.md) | Full prediction-step argument reference |
