# Two-Color Run: SGA-2020

This document records the full command sequence for fitting and predicting using
the two-color TFR model on the `SGA-2020_iron_Vrot_VI_corr_v6.fits` dataset.

The two-color model extends the single-color (r–z) model by adding g-band as a
**second independent latent color factor** (g–r). The joint distribution becomes
quadrivariate (x̂, ŷ, ẑ, ĝ) with a 4×4 covariance matrix. The two color factors
ε_c (r–z) and ε_g (g–r) are uncorrelated. See `2color.stan` for the implementation.

## Full Workflow

| Phase | Steps | How |
|-------|-------|-----|
| A — Setup | 1–3b | Complete [DR1.md](DR1.md) Steps 1–3b (selection ellipse, pull profile, fiducial, export config) |
| B — 2color | 4–8 | Run Steps 4–8 below (data prep, compilation, MAP, metric, sampling, prediction) |

To run Phase B on NERSC Perlmutter as batch SLURM jobs, see [BATCH_NERSC.md](BATCH_NERSC.md).

---

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr_v6.fits   # input FITS catalog
export RUN=DR1_v6_2color                              # output directory name: output/$RUN/
export CONFIG=configs/dr1_v6_2color.json              # pipeline config
```

---

## Two-population runs (Spiral / Irregular)

`SGA-2020_loa_Vrot_VI_v0.fits` is fit as **two independent populations**, each
with its own selection, its own MCMC, and its own prediction run:

| Population | Predicate | Rows |
|---|---|---|
| Spiral | `MORPHTYPE_AI == 'Spiral'` and not VI-rejected | 23,422 |
| Irregular | `MORPHTYPE_AI == 'Irregular'` and not VI-rejected | 8,409 |

`JOHN_VI` is a masked column whose only unmasked value is `'reject'`, so
"not VI-rejected" means `JOHN_VI.mask == True`. It applies to **both**
populations.

The population cut is applied **once, upstream**, by pre-filtering the catalog
into one FITS file per population. Each population is then an ordinary
single-FITS run and the rest of the pipeline is unchanged. This is deliberate:
the validity mask is re-derived independently in `desi_data.py`,
`color_predict.py` (each catalog/covariance writer builds its own validity
mask, then calls the shared `_train_analysis_masks` helper for train/analysis
membership) and `explore_residuals.py`, so a population predicate threaded
through all of them could be applied in one place and missed in another —
silently fitting one population while validating against another. A
pre-filtered file cannot disagree with itself.

```bash
python make_population_subsets.py     # writes data/<stem>_spiral.fits, _irregular.fits
```

Then, **for each population**, run the full workflow with its own run/config:

```bash
export FITS=data/SGA-2020_loa_Vrot_VI_v0_spiral.fits   # or _irregular.fits
export RUN=DR2_v0_2color_spiral                        # or _irregular
export CONFIG=configs/dr2_v0_2color_spiral.json        # or _irregular
```

Each population needs **its own selection** — complete [DR1.md](DR1.md)
Steps 1–3b separately for each. The parallelogram fit to the blended sample is
centred on the spiral-dominated mix (it keeps 82% of z-selected spirals but only
72% of irregulars, which sit ~0.05 dex slower and ~0.3 mag fainter), and
`2color.stan` integrates over the selection region, so the config must describe
the cut actually applied to that population. Note `set_fiducial.py` and
`export_config.py` are interactive.

### Training sample size

The per-population configs set `train_fraction` rather than `n_objects`:

```json
"train_fraction": 0.40
```

`n_objects` is an absolute count, which silently stops meaning "40%" once the
selection is re-derived and the post-selection count moves. `train_fraction`
resolves against the post-selection count at data-prep time
(`n_objects = round(train_fraction * N_after_cuts)`) and the resolved value is
recorded in `input.json`. The two are mutually exclusive; if both are given,
`n_objects` wins and a warning is printed.

This only chooses *how many*. The galaxies chosen are still recorded explicitly
as `train_sga_ids` in `input.json` — `color_predict.py` never sees the
fraction, only the IDs.

`color_predict.py` predicts on the **union** of training and analysis
galaxies: `MAIN` marks every galaxy that passes selection cuts (training and
analysis both), and a separate boolean column `ANALYSIS` marks the subset
*not* in `train_sga_ids` (i.e. `MAIN & ANALYSIS` = analysis rows,
`MAIN & ~ANALYSIS` = training rows). This is written into
`color_catalog.fits`/`color_xonly_catalog.fits` by `write_desi_catalog_color`/
`write_desi_catalog_color_xonly`, and the same union is used for the
posterior-predictive covariance matrices (`write_cov_color`,
`write_cov_color_xonly`). See "Analysis-only covariance matrix" below for how
to recover the analysis-only covariance from the union output.

---

## Step 4: Prepare data

Convert the FITS file to Stan JSON format. Both z-band and g-band magnitudes are
loaded automatically when the FITS catalog contains the relevant columns.

```bash
python desi_data.py --config $CONFIG
```

`desi_data.py` writes `g`, `sigma_g`, and `c_bar_g_obs` to `input.json` in addition
to the z-band fields. The `init.json` includes starting values for all color and
g-band parameters.

Inspect the scatter plot:

```bash
open output/$RUN/data.png
```

---

## Step 5: Compile Stan model

Run from inside the `../../cmdstan/` directory:

```bash
cd ../../cmdstan
make ../TFPV/ariel/2color
cd ../TFPV/ariel
```

---

## Step 5d: Find MAP estimate (init_MAP.json)

The MAP provides a warm start near the posterior mode.

```bash
./2color optimize \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv

# Convert optimizer output to MCMC init file
python3 make_map_init.py --run $RUN
```

`make_map_init.py` also floors any `sigma_int_*` / `log_sigma_int_*` parameter
that the MAP drove near its 0 boundary (default floor: 0.01, via
`--sigma-floor`). MAP frequently collapses these to ~0, which starts HMC
warmup directly in the degenerate near-singular-covariance regime and slows
convergence; starting slightly off the boundary gives the sampler room to
explore. Pass `--sigma-floor 0` to disable and use the raw MAP value.

---

## Step 5e: Build initial metric from short run

The 2color model has highly varying parameter scales (condition number ~2.7M across
17 sampling parameters), so the default identity mass matrix leads to tiny stepsizes
(~0.002) and maximum treedepth (~10) on every step. Providing a pre-computed
covariance as the initial metric raises the stepsize ~40× and brings treedepth to
a practical range.

```bash
# Short 1-chain run — metric.json does NOT exist yet, so metric_file is
# omitted; Stan learns a dense metric from scratch (identity start) during
# warmup. metric.json is created by the Python snippet below, from this run's
# raw posterior draws — not from Stan's own adapted-metric output.
./2color sample num_warmup=100 num_samples=100 num_chains=1 \
    algorithm=hmc metric=dense_e \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color.csv

# Extract sample covariance as inverse mass matrix
python - <<'EOF'
import pandas as pd, numpy as np, json, os

RUN = os.environ['RUN']
df = pd.read_csv(f'output/{RUN}/2color.csv', comment='#')

# Declaration order in 2color.stan's parameters block (unconstrained dims):
#   slope_std(1), intercept_std[N_bins](1), sigma_int_x(1), n_null(3, unit_vector),
#   Sc_scale(2), Sc_Lcorr(1 free), delta_c(1), mu_c(1), delta_g(1), mu_g(1),
#   alpha_kcorr_r(1), alpha_kcorr_z(1), alpha_kcorr_g(1)  =  16 total.
# metric_file (dense_e) requires a matrix matching this full 16-dim count, or
# Step 6 fails with "dims declared=(16,16); dims found=(13,13)".
before_null = ['slope_std', 'intercept_std.1', 'sigma_int_x']
after_null = [
    'Sc_scale.1', 'Sc_scale.2',                        # chromatic scatter scales
    'Sc_Lcorr.2.1',                                    # 2x2 correlation Cholesky (1 free entry)
    'delta_c', 'mu_c', 'delta_g', 'mu_g',
    'alpha_kcorr_r', 'alpha_kcorr_z', 'alpha_kcorr_g',
]
sampling_params = before_null + after_null
X = df[sampling_params].values
cov13 = np.cov(X.T)

# Embed into a 16x16 identity, inserting an uninformative identity block for
# n_null's 3 unconstrained dims at its declared position (index 3:6). The CSV
# only has n_null's constrained unit-sphere coordinates, not its unconstrained
# values, so its true covariance can't be recovered here -- but n_null isn't
# the source of the ~2M condition number, so identity is a reasonable
# warm-start guess; NUTS's own warmup adapts it further in Step 6.
n_before = len(before_null)
new_idx = list(range(n_before)) + list(range(n_before + 3, 16))
cov = np.eye(16)
for i, oi in enumerate(new_idx):
    for j, oj in enumerate(new_idx):
        cov[oi, oj] = cov13[i, j]

metric = {"inv_metric": cov.tolist()}
with open(f'output/{RUN}/metric.json', 'w') as f:
    json.dump(metric, f)
print(f'Metric written to output/{RUN}/metric.json (16x16)')
print(f'Condition number: {np.linalg.cond(cov):.1f}')
EOF
```

The short run takes ~7 hours (N=4728 galaxies, 4×4 Cholesky per galaxy).

---

## Step 6: Run MCMC sampling

Pass the pre-computed metric via `metric_file`. The 250-step warmup refines it
further. With the metric, stepsize adapts to ~0.08 (vs ~0.002 without) and
treedepth stays at 4–6.

```bash
./2color sample num_warmup=250 num_samples=1000 num_chains=4 \
    adapt save_metric=1 \
    algorithm=hmc metric=dense_e \
    metric_file=output/$RUN/metric.json \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color.csv
```

This produces `2color_1.csv` … `2color_4.csv` in `output/$RUN/`. The adapted
per-chain metrics are saved as `2color_metric_1.json` … `2color_metric_4.json`.

Actual timing for DR1_v6_2color: warmup ~5.5 hours, sampling ~8.9 hours (4 chains).

### Running the 4 chains in parallel (local multi-core machines)

The `num_chains=4` invocation above runs the 4 chains **sequentially** within
one process — the CPU `2color` binary isn't built with `STAN_THREADS`, so
there's no internal parallelism across chains. On a machine with several free
cores, launch 4 separate single-chain processes in the background instead
(the same pattern `slurm/step6_node.sh` uses on NERSC, one chain per GPU
there — this is the CPU-only, no-SLURM equivalent, one chain per core):

```bash
PIDS=()
for CHAIN_ID in 1 2 3 4; do
    ./2color sample num_warmup=250 num_samples=1000 \
        adapt save_metric=1 \
        algorithm=hmc engine=nuts metric=dense_e \
        metric_file=output/$RUN/metric.json \
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

`id=$CHAIN_ID` is required — it gives each chain a distinct RNG seed/offset,
whereas 4 identical invocations without it risk seed collisions. Output files
are still `2color_1.csv` … `2color_4.csv`, matching the `2color_?.csv` glob
used by `stansummary`/`diagnose`/`corner.py`/`color_predict.py`, so nothing
downstream changes. Only run as many chains concurrently as you have free
cores — each chain is single-threaded, so 4-way parallelism needs ≥4 free
cores to actually be faster than the sequential command above.

---

## Step 7: Diagnose and visualize

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt

# Corner plot
python corner.py --run $RUN --model 2color
```

Inspect:

```bash
cat output/$RUN/stansummary.txt
open output/$RUN/2color.png
```

Key parameters to check:
- `slope` — TFR slope
- `Sc_scale.1/2` — chromatic scatter scales. The chromatic-only covariance
  S = V Σ_c Vᵀ excludes the achromatic (PV/distance-modulus) direction by
  construction; all achromatic scatter is owned by the downstream C_vv model.
- `Sc_Lcorr.2.1` — the single free entry of the 2×2 chromatic correlation
  Cholesky (controls the correlation between the two chromatic modes).
- `delta_c`, `delta_g` — color–velocity slopes (mean structure)
- `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` — band k-corrections

```bash
python explore_residuals.py --config $CONFIG --kind 2color
```

Residual plots (x-only prediction conditioned on x̂) are written to
`output/$RUN/explore_residuals/`.

---

## Step 8: Predict absolute magnitudes

x-only (conditioning on x̂ and z_obs, marginalizing ẑ and ĝ) is the
unconditional default — it doesn't depend on the z/g-band k-corrections and
D-matrix coupling that the full model needs, which were found (on an
abacus-mock experiment) to introduce a dust-correlated systematic bias
absent from the x-only predictions. Add `--full` to additionally compute the
full quadrivariate (x̂, ẑ, ĝ)-conditioned model for comparison:

```bash
python color_predict.py --config $CONFIG --model 2color            # x-only only
python color_predict.py --config $CONFIG --model 2color --full     # + full model
```

The script reads:
- `output/$RUN/config.json` — phase-space selection and FITS path
- `output/$RUN/input.json` — bounds, mean_x, z-band and g-band data
- `output/$RUN/2color_?.csv` — posterior MCMC draws

Outputs produced (always, x-only):

| File | Description |
|------|-------------|
| `output/$RUN/color_grid_xonly.png` | Mean residual on (x̂, ŷ) grid (MAIN sample, x-only) |
| `output/$RUN/color_grid_xonly_full.png` | Mean residual on (x̂, ŷ) grid (full sample, x-only) |
| `output/$RUN/redshift_color_xonly.png` | Residual vs. redshift (x-only) |
| `output/$RUN/gr_color_xonly.png` | Residual vs. g−r color (x-only) |
| `output/$RUN/variance_redshift_color_xonly.png` | Prediction variance vs. redshift (x-only) |
| `output/$RUN/redshift_grid_color.png` | Mean redshift on (x̂, ŷ) grid (data-space, independent of model) |
| `output/$RUN/color_xonly_catalog.fits` | DESI catalog with MU_TF, LOGDIST, MAIN, ANALYSIS (x-only) |
| `output/$RUN/color_xonly_cov.h5` | (G,G) covariance matrix HDF5, datasets `cov`, `analysis` (x-only) |

With `--full`, additionally:

| File | Description |
|------|-------------|
| `output/$RUN/color_grid.png` | Mean residual on (x̂, ŷ) grid (MAIN sample, full model) |
| `output/$RUN/color_grid_full.png` | Mean residual on (x̂, ŷ) grid (full sample, full model) |
| `output/$RUN/redshift_color.png` | Residual vs. redshift scatter (full model) |
| `output/$RUN/variance_redshift_color.png` | Prediction variance vs. redshift (full model) |
| `output/$RUN/color_catalog.fits` | DESI catalog with MU_TF, LOGDIST, MAIN, ANALYSIS (full model) |
| `output/$RUN/color_cov.h5` | (G,G) covariance matrix HDF5, datasets `cov`, `analysis` (full model) |

`MAIN` marks every galaxy passing selection cuts — the union of training and
analysis. `ANALYSIS` marks the non-training subset (`MAIN & ANALYSIS` =
analysis rows, `MAIN & ~ANALYSIS` = training rows). The (G,G) covariance
matrices are computed over the same union, in the row/col order of the
`MAIN`-selected rows; the `analysis` dataset (or, for non-2color models,
`color_cov_analysis.npy`/`color_xonly_cov_analysis.npy`) is a boolean array
in that same row/col order, so the analysis-only covariance is recoverable
without re-deriving any masks:

```python
import h5py, numpy as np
with h5py.File("output/$RUN/color_xonly_cov.h5", "r") as f:
    cov = f["cov"][:]                    # full (G, G) union covariance
    analysis = f["analysis"][:]          # bool, True = analysis (non-training) row
cov_analysis = cov[np.ix_(analysis, analysis)]
```

> **Note:** For the 2color model the covariance matrices are written as gzip-compressed
> HDF5 files (`color_cov.h5`, `color_xonly_cov.h5`), not FITS, to allow row-chunked
> writes that keep peak memory below ~1 GB.  Read with:
> ```python
> import h5py, numpy as np
> with h5py.File("output/$RUN/color_xonly_cov.h5", "r") as f:
>     cov = f["cov"][:]          # full matrix (G×G float32)
>     row = f["cov"][0, :]       # single row without loading all
> ```
> For non-2color models, the same matrices are written as FITS
> (`color_cov.fits`, `color_xonly_cov.fits`) with a sidecar
> `color_cov_analysis.npy`/`color_xonly_cov_analysis.npy` boolean array in
> place of the `analysis` dataset.

---

## Step 8 variants

x-only always runs; `--full` is purely additive (there's no flag to disable
x-only). To run diagnostics and catalogs only (no covariance — much faster):

```bash
python color_predict.py --config $CONFIG --model 2color --no-cov
```

To skip the catalog too (diagnostic plots only):

```bash
python color_predict.py --config $CONFIG --model 2color --no-cov --no-catalog
```

Add `--full` to any of the above to also get the full model's outputs
(`color_catalog.fits`, `color_cov.h5`, `color_grid.png`, etc.).

---

## File Reference

| File | Purpose |
|------|---------|
| `2color.stan` | Stan model: quadrivariate TFR with two independent color factors |
| `color_predict.py --model 2color` | Posterior predictive computation (shared with color model) |
| `desi_data.py` | Data preparation — writes g-band fields when FITS has G_MAG_SB26 |
| `corner.py` | Corner plot including γ_g, δ_g, μ_g, τ_g, α_{k,g} |
| `configs/dr1_v6_2color.json` | Pipeline config for this run |
| [COLOR.md](COLOR.md) | Single-color (r–z) workflow |
| [TOPHAT.md](TOPHAT.md) | Baseline tophat + k-correction workflow |
