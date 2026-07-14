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
# Short 1-chain run — metric.json does NOT exist yet; Stan learns dense metric
# from scratch during warmup. metric.json is created by the Python snippet below.
./2color sample num_warmup=100 num_samples=100 num_chains=1 \
    algorithm=hmc metric=dense_e \
    metric_file=output/$RUN/metric.json \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color.csv

# Extract sample covariance as inverse mass matrix
python - <<'EOF'
import pandas as pd, numpy as np, json, os

RUN = os.environ['RUN']
df = pd.read_csv(f'output/{RUN}/2color.csv', comment='#')
sampling_params = [
    'slope_std', 'intercept_std.1', 'sigma_int_x',
    'S_scale.1', 'S_scale.2', 'S_scale.3',            # intrinsic (y,z,g) scatter std
    'S_Lcorr.2.1', 'S_Lcorr.3.1', 'S_Lcorr.3.2',      # free correlation-Cholesky entries
    'delta_c', 'mu_c', 'delta_g', 'mu_g',
    'alpha_kcorr_r', 'alpha_kcorr_z', 'alpha_kcorr_g'
]
X = df[sampling_params].values
cov = np.cov(X.T)
metric = {"inv_metric": cov.tolist()}
with open(f'output/{RUN}/metric.json', 'w') as f:
    json.dump(metric, f)
print(f'Metric written to output/{RUN}/metric.json')
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
- `S_scale.1/2/3` — intrinsic (y,z,g) scatter std. The free intrinsic-covariance
  parameterization (replacing the old `gamma`/`tau_c`/`gamma_g`/`tau_g`/`sigma_int_*`
  product form) fits far better and reveals a near-rank-1, ~0.4-mag, ~99%-correlated
  achromatic (PV/distance-modulus) scatter mode (DR2 MAP: ~0.42/0.39/0.44).
- `S_Lcorr` — intrinsic-correlation Cholesky; implied band correlations are high
  (DR2 MAP: y-z ≈ y-g ≈ 0.99, z-g ≈ 0.97).
- `delta_c`, `delta_g` — color–velocity slopes (mean structure)
- `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` — band k-corrections

```bash
python explore_residuals.py --config $CONFIG --kind 2color
```

Residual plots (x-only prediction conditioned on x̂) are written to
`output/$RUN/explore_residuals/`.

---

## Step 8: Predict absolute magnitudes (full 2color model)

Run all outputs (diagnostic plots, catalog, covariance, and x-only variants):

```bash
python color_predict.py --config $CONFIG --model 2color --xonly
```

The script reads:
- `output/$RUN/config.json` — phase-space selection and FITS path
- `output/$RUN/input.json` — bounds, mean_x, z-band and g-band data
- `output/$RUN/2color_?.csv` — posterior MCMC draws

The full model conditions on (x̂, ẑ, ĝ) using the 3×3 D matrix to predict ŷ.
The `--xonly` flag additionally produces predictions conditioned on x̂ alone
(marginalizing ẑ and ĝ), with A₁₁ = γ²τ²_c + γ²_g τ²_g + σ²_{int,y} + σ²_{y,★}.

Outputs produced:

| File | Description |
|------|-------------|
| `output/$RUN/color_grid.png` | Mean residual on (x̂, ŷ) grid (MAIN sample) |
| `output/$RUN/color_grid_full.png` | Mean residual on (x̂, ŷ) grid (full sample) |
| `output/$RUN/redshift_color.png` | Residual vs. redshift scatter |
| `output/$RUN/redshift_color_xonly.png` | Residual vs. redshift (x-only) |
| `output/$RUN/gr_color_xonly.png` | Residual vs. g−r color (x-only) |
| `output/$RUN/variance_redshift_color.png` | Prediction variance vs. redshift |
| `output/$RUN/variance_redshift_color_xonly.png` | Prediction variance vs. redshift (x-only) |
| `output/$RUN/color_catalog.fits` | DESI catalog with MU_TF, LOGDIST (full model) |
| `output/$RUN/color_xonly_catalog.fits` | DESI catalog with MU_TF, LOGDIST (x-only) |
| `output/$RUN/color_cov.h5` | (G,G) covariance matrix HDF5, dataset `cov` (full model) |
| `output/$RUN/color_xonly_cov.h5` | (G,G) covariance matrix HDF5, dataset `cov` (x-only) |

> **Note:** For the 2color model the covariance matrices are written as gzip-compressed
> HDF5 files (`color_cov.h5`, `color_xonly_cov.h5`), not FITS, to allow row-chunked
> writes that keep peak memory below ~1 GB.  Read with:
> ```python
> import h5py, numpy as np
> with h5py.File("output/$RUN/color_xonly_cov.h5", "r") as f:
>     cov = f["cov"][:]          # full matrix (G×G float32)
>     row = f["cov"][0, :]       # single row without loading all
> ```

---

## Step 8 variants

To run diagnostics and catalogs only (no covariance — much faster):

```bash
python color_predict.py --config $CONFIG --model 2color --xonly --no-cov
```

To run only the x-only covariance (skip full cov and all catalogs):

```bash
python color_predict.py --config $CONFIG --model 2color --xonly --no-cov --no-catalog
# then separately if you also need the full cov:
python color_predict.py --config $CONFIG --model 2color --no-catalog
```

To run only the full covariance (skip x-only and catalog):

```bash
python color_predict.py --config $CONFIG --model 2color --no-catalog
```

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
