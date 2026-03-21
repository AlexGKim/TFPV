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
# DESI
python predict.py --run DESI --model tophat --source DESI
python predict.py --run DESI --model normal --source DESI

# Ariel mock
python predict.py --run ariel --model tophat --source ariel

# Fullmocks — reads FITS from --dir, compares predictions to R_ABSMAG_SB26_TRUE
python predict.py --run c000_ph000_r001 --model tophat --source fullmocks \
  --dir /path/to/mocks

# Subsample galaxies used for prediction
python predict.py --run c000_ph000_r001 --model tophat --source fullmocks \
  --dir /path/to/mocks --n_objects 100000

# Looser prediction selection (expand windows relative to training cuts)
python predict.py --run c000_ph000_r001 --model tophat --source fullmocks \
  --dir /path/to/mocks --n_objects 100000 \
  --delta_haty_min -0.5 --delta_haty_max 0.5 \
  --delta_intercept_plane -0.05 --delta_intercept_plane2 0.05 \
  --delta_z_obs_min -0.03

# Predict on a different simulation realization than the one used for fitting
python predict.py --run c000_ph000_r001 --model tophat --source fullmocks \
  --dir /path/to/mocks --predict_run c000_ph000_r002
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run NAME` | required | Run name; reads `output/<NAME>/` for chains and `input.json` |
| `--model` | `tophat` | Posterior predictive model: `tophat` or `normal` |
| `--source` | `DESI` | Data source: `DESI`, `ariel`, or `fullmocks` |
| `--n_objects INT` | all | Subsample size for prediction |
| `--dir DIR` | `data` | Directory containing FITS files (`fullmocks` only) |
| `--predict_run NAME` | same as `--run` | Simulation ID for the FITS file to predict on (`fullmocks` only) |

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
