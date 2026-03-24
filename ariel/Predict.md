# Absolute Magnitude Prediction

## Introduction

After MCMC sampling (see `TFFit.md`), this stage uses the fitted TFR posterior
to infer absolute magnitudes for individual galaxies. For each galaxy with
observed rotation velocity x̂_* and uncertainty σ_{x,*}, the script computes
the posterior predictive mean and SD of the latent magnitude y_* by
marginalizing over MCMC draws.

---

## Step 1: Posterior Predictive Inference

Algorithm: `predict.py`

### 1. Posterior predictive models

The two Stan models give rise to two different posterior predictive computations.

**Normal model** (`--model normal`):

The latent true magnitude y_TF has a Gaussian prior and the posterior for
y_TF | x̂_* is computed analytically by Gaussian conjugacy:

- y_TF | x̂_* is Normal(μ_post, V_post), where the posterior variance V_post
  combines the prior variance τ² and the likelihood variance s²(σ²_{int,x} +
  σ²_{x,*}).
- y_* = y_TF + Normal(0, σ²_{int,y}), so E[y_* | x̂_*] = μ_post and
  Var[y_* | x̂_*] = V_post + σ²_{int,y}.

For each galaxy, mean and SD are computed vectorized over all M posterior
draws, then mixed:
- mean_y[g] = mean over draws of μ_post(g, m)
- sd_y[g] = sqrt( E[V_post + σ²_{int,y}] + Var(μ_post) ) — law of total variance

**Tophat model** (`--model tophat`):

The latent true magnitude y_TF has a uniform prior on [y_min, y_max].
The posterior for y_TF | x̂_* is a truncated normal with:

- Untruncated parameters: μ_L = c + s·x̂_*, σ²_L = s²(σ²_{int,x} + σ²_{x,*})
- Truncated to [y_min, y_max]; E[y_TF | x̂_*] and Var[y_TF | x̂_*] are
  computed from the truncated normal moments using the standard mills-ratio
  formula.
- y_* adds σ_{int,y} scatter: E[y_* | x̂_*] = E[y_TF | x̂_*], Var[y_*] =
  Var[y_TF] + σ²_{int,y}.

Mixture moments are computed over draws using the law of total variance, as
in the normal case.

### 2. Data sources

| `--source` | Data loaded | Extra output |
|------------|-------------|--------------|
| `fullmocks` | AbacusSummit FITS (`TF_extended_AbacusSummit_base_<run>_*.fits`) | truth-diff grid, high-pull scatter, redshift histograms |
| `DESI` | DESI FITS catalog (`data/DESI-DR1_TF_pv_cat_v15.fits`) | residual grid, redshift scatter |
| `ariel` | Ariel mock CSV | residual grid, redshift scatter |

For `--source fullmocks`, the script reads training selection cuts from
`output/<run>/input.json` and applies them (with optional offsets) to the
prediction sample.

### 3. Output files

**All sources:**

| File | Description |
|------|-------------|
| `output/<run>/{model}_grid.png` | mean_pred − ŷ_obs averaged on (x̂, ŷ) grid |
| `output/<run>/redshift_{model}.png` | pull vs. redshift scatter with weighted mean |

**`--source fullmocks` only:**

| File | Description |
|------|-------------|
| `output/<run>/{model}_truth_diff_grid.png` | (mean_pred − y_true) / σ_pred averaged on (x̂, ŷ) grid |
| `output/<run>/{model}_highpull.png` | (x̂, ŷ) scatter with pull > 4 highlighted in red |
| `output/<run>/redshift_hist_{model}.png` | pull histograms in 9 log-spaced redshift bins |

### Usage

```bash
# Fullmocks — reads FITS from --dir, compares predictions to R_ABSMAG_SB26_TRUE
python predict.py --run $RUN --model tophat --source fullmocks --dir $(dirname $FITS)

# Subsample galaxies used for prediction
python predict.py --run $RUN --model tophat --source fullmocks \
  --dir $(dirname $FITS) --n_objects 100000

# Looser prediction selection (expand windows relative to training cuts)
python predict.py --run $RUN --model tophat --source fullmocks \
  --dir $(dirname $FITS) --n_objects 100000 \
  --delta_haty_min -0.5 --delta_haty_max 0.5 \
  --delta_intercept_plane -0.05 --delta_intercept_plane2 0.05 \
  --delta_z_obs_min -0.03

# Predict on a different simulation realization than the one used for fitting
python predict.py --run $RUN --model tophat --source fullmocks \
  --dir $(dirname $FITS) --predict_run c000_ph000_r002

# DESI — predictions only
python predict.py --run DESI --model tophat --source DESI
python predict.py --run DESI --model normal --source DESI

# DESI — predictions + write augmented catalog FITS
python predict.py --run DESI --model tophat --source DESI \
  --catalog --input data/SGA-2020_iron_Vrot_VI_corr.fits

# Ariel mock
python predict.py --run ariel --model tophat --source ariel
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run NAME` | required | Run name; reads `output/<NAME>/` for chains and `input.json` |
| `--model` | `tophat` | Posterior predictive model: `tophat` or `normal` |
| `--source` | `DESI` | Data source: `DESI`, `ariel`, or `fullmocks` |
| `--n_objects INT` | all | Subsample size for prediction |
| `--dir DIR` | `data` | Directory containing FITS files (`fullmocks` only) |
| `--predict_run NAME` | same as `--run` | Simulation ID for the FITS file to predict on (`fullmocks` only) |
| `--catalog` | off | Write augmented catalog to `output/<run>/<model>_catalog.fits` (`DESI` only) |
| `--input PATH` | `data/SGA-2020_iron_Vrot_VI_corr.fits` | Input FITS path used by `--catalog` |

**Selection offset flags for `--source fullmocks`** (all default to 0, i.e.
match the training selection stored in `input.json`):

| Flag | Effect |
|------|--------|
| `--delta_haty_min FLOAT` | shift `haty_min` cut (negative = looser) |
| `--delta_haty_max FLOAT` | shift `haty_max` cut (positive = looser) |
| `--delta_z_obs_min FLOAT` | shift `z_obs_min` cut (negative = looser) |
| `--delta_z_obs_max FLOAT` | shift `z_obs_max` cut (positive = looser) |
| `--delta_intercept_plane FLOAT` | shift lower plane intercept (negative = looser) |
| `--delta_intercept_plane2 FLOAT` | shift upper plane intercept (positive = looser) |

The oblique plane cut is applied by default. Pass `--no_plane_cut` to disable it.

---

## Step 2: Posterior Predictive Covariance

Algorithm: `predict_cov.py`

The predicted magnitudes y\*[g] are covariant because they share the same
posterior draws θ_m. The full (G, G) covariance matrix is needed for
downstream peculiar velocity likelihood evaluation.

### Method

By the law of total covariance:

```
Cov(y*[g1], y*[g2]) = (1/M) Σ_m μ[m,g1]·μ[m,g2]  −  mean_y[g1]·mean_y[g2]
```

where μ[m,g] = E[y\* | x̂_g, θ_m] is the per-draw conditional mean
(Gaussian conjugacy for the normal model; truncated-normal mean for the
tophat model). The covariance is accumulated via chunked matrix multiply
(`mu_chunk.T @ mu_chunk`) to avoid allocating the full (M, G) matrix.

**Note:** The diagonal of `cov` is the between-draw variance of μ[m,g]
only. It does **not** include the within-draw variance E[σ²_{int,y}],
which is the extra term in `sd_y**2` returned by `predict.py`. Use `sd_y`
from `predict.py` for individual magnitude uncertainties; use `cov` here
for correlated structure across galaxies.

**Memory:** G = 9 474, M = 2 000, chunk\_size = 200 → ~15 MB peak
intermediate vs ~300 MB for the naive `np.cov` approach. Output (G, G)
matrix: ~720 MB float64 in both cases.

### Usage (Python API)

`predict_cov.py` is a library module, not a script. Import it after
running `predict.py` to obtain draws and galaxy arrays.

```python
import json
import numpy as np
from predict import read_cmdstan_posterior
from predict_cov import ystar_pp_cov_normal_vectorized, ystar_pp_cov_tophat_vectorized

# Load posterior draws
draws = read_cmdstan_posterior(
    "output/DESI/normal_?.csv",
    keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
    drop_diagnostics=True,
)

# Load galaxy data
with open("output/DESI/input.json") as f:
    data = json.load(f)
xhat_star    = np.array(data["x"])
sigma_x_star = np.array(data["sigma_x"])

# Normal model covariance
cov = ystar_pp_cov_normal_vectorized(draws, xhat_star, sigma_x_star)
# cov.shape == (G, G)

# Tophat model covariance (bounds from input.json)
draws_th = read_cmdstan_posterior(
    "output/DESI/tophat_?.csv",
    keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
    drop_diagnostics=True,
)
cov_th = ystar_pp_cov_tophat_vectorized(
    draws_th, xhat_star, sigma_x_star,
    bounds_json="output/DESI/input.json",   # supplies y_min, y_max
)

# Or pass bounds explicitly
cov_th = ystar_pp_cov_tophat_vectorized(
    draws_th, xhat_star, sigma_x_star,
    y_min=data["y_min"], y_max=data["y_max"],
)

# Save covariance + correlation image
from predict_cov import plot_cov
plot_cov(cov,    "output/DESI/normal_cov.png",  title="DESI normal model")
plot_cov(cov_th, "output/DESI/tophat_cov.png",  title="DESI tophat model")
```

### Function signatures

```
ystar_pp_cov_normal_vectorized(draws, xhat_star, sigma_x_star, chunk_size=200)
    draws       : DataFrame with columns slope, intercept.1,
                  sigma_int_x, sigma_int_y, mu_y_TF, tau
    → cov : (G, G) ndarray

ystar_pp_cov_tophat_vectorized(draws, xhat_star, sigma_x_star, *,
                                bounds_json=None, y_min=None, y_max=None,
                                on_bad_Z="floor", Z_floor=1e-300,
                                chunk_size=200)
    draws       : DataFrame with columns slope, intercept.1,
                  sigma_int_x, sigma_int_y
    bounds_json : JSON file path supplying y_min/y_max (ignored if
                  y_min and y_max are given directly)
    → cov : (G, G) ndarray

plot_cov(cov, output_path, *, title="Posterior predictive covariance", vmax=None)
    cov         : (G, G) ndarray from either covariance function above
    output_path : PNG file to write
    vmax        : colour-scale limit for covariance panel (default: 99th
                  percentile of |cov|); correlation panel always ±1
    Writes a two-panel PNG: left = covariance, right = correlation matrix.
```
