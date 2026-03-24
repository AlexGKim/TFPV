# Tully-Fisher Peculiar Velocity (TFPV)

Bayesian statistical modeling of the Tully-Fisher Relation (TFR) for peculiar velocity analysis. The TFR is an empirical relation between galaxy rotation velocity (`x = log10(V/100 km/s)`) and absolute magnitude (`y`).

This project does three things:

1. **Sample selection**: Normal galaxies occupy a compact
region of the data phase space who obey a Tully-Fisher
relation with a specific slope.  There are a small
froction of other galaxies
that do not follow the Tully-Fisher relation that for the
most part occupy a different region of data phase space.
An analysis of the full sample would yield a different slope
with a poorer quality of fit.  The objective is to determine
sample selection criteria that gives high completeness
and purity of normal galaxies.
2. **Fit the TFR**: Infer TFR parameters from galaxy rotation velocity and absolute magnitude data, accounting for observational selection effects (magnitude limits and oblique plane cuts in x–y space).
3. **Infer absolute magnitudes**: Given fitted TFR parameters and observed rotation velocities, infer absolute magnitudes for individual galaxies.

Two Stan models are provided, differing in the assumed distribution of the latent true magnitude y_TF:

- **`tophat.stan`**: Uniform (tophat) prior on y_TF; latent magnitudes marginalized with Gauss-Legendre quadrature.
- **`normal.stan`**: Gaussian prior on y_TF; adds hyperparameters `mu_y_TF` and `tau`.

The tophat model gives better performance on mock data: the latent distribution is not accurately described by either model but the flat/cuspy central region is better captured by the tophat.

The statistical models are described in detail in `doc/model1.tex`, `doc/model2.tex`, and `doc/model3.tex`.

---

## Dependencies

- **CmdStan** — Stan compiler and sampler, expected at `../../cmdstan/` relative to this directory
- **Python** — numpy, scipy, pandas, matplotlib, astropy, chainconsumer, cmdstanpy

---

## Repository Layout

```
ariel/
  selection_ellipse.py               — noise- and truncation-corrected GMM fit to phase space
  ellipse_sweep.py                   — 3-D mag-split grid sweep; auto-selects fiducial cuts
  selection_criteria.py              — legacy plateau-detection (requires old ellipse_sweep.json)
  ariel_data.py                      — data prep for Ariel mock catalog
  desi_data.py                       — data prep for DESI survey data (reads mag_split_fiducial.json)
  fullmocks_data.py                  — data prep for AbacusSummit FITS mocks (reads cut_sweep_best_config.json)
  tophat.stan                        — Stan model (tophat prior)
  normal.stan                        — Stan model (normal prior)
  corner.py                          — posterior corner plots (uses chainconsumer)
  predict.py                         — infer absolute magnitudes from fitted TFR
  predict_cov.py                     — posterior predictive covariance library
  cut_sweep.py                       — fullmocks selection cut grid sweep (produces cut_sweep_best_config.json)
  plot_magnitude_predictions.py      — publication-quality magnitude prediction plots (fullmocks)
  plot_desi_magnitude_predictions.py — magnitude prediction plots for DESI
  figs/distributions.py              — figures of latent variable distributions (paper figures)
  doc/                               — LaTeX descriptions of the statistical models
  output/                            — all data products (gitignored)
```

---

## Output Directory Structure

All data products live under `output/<run>/` where `<run>` is a short name for the dataset and selection-cut combination (e.g. `ariel`, `DESI`, `c000_ph000_r000`):

```
output/<run>/
  selection_ellipse.json            — core Gaussian parameters and derived cuts
  selection_ellipse.png             — phase-space scatter with GMM ellipses and cuts
  mag_split_grid.json               — 3-D mag-split grid sweep results
  mag_split_grid.png                — heatmap of MLE slope over (n_σ_ŷmin, n_σ_ŷmax)
  fiducial_slope_hist.png           — histogram of MLE slopes at fiducial n_σ_perp
  mag_split_fiducial.json           — chosen cut values (DESI workflow)
  cut_sweep_best_config.json        — optimal cut parameters (fullmocks workflow)
  config.json                       — selection parameters used (for reproducibility)
  input.json                        — Stan data
  init.json                         — Stan initial conditions
  data.png                          — galaxy scatter plot with selection cuts
  tophat_1.csv … tophat_4.csv       — MCMC chains (tophat model)
  normal_1.csv  … normal_4.csv      — MCMC chains (normal model)
  tophat_metric_1.json … tophat_metric_4.json  — per-chain adapted mass matrices
  tophat.png                        — corner plot (tophat posterior)
  normal.png                        — corner plot (normal posterior)
  tophat_grid.png                   — mean_pred − yhat_obs on (x̂, ŷ) grid
  redshift_grid_tophat.png          — residual heat-map on (x̂, redshift) grid
  redshift_tophat.png               — pull vs. redshift scatter
  tophat_cov.fits                   — posterior predictive covariance matrix, float32, (G, G)
  tophat_cov.png                    — covariance + correlation matrix, two panels
  tophat_cov_sub.png                — same for a random subset of ≤512 galaxies
  tophat_catalog.fits               — augmented FITS catalog with predicted magnitudes (DESI)
  tophat_truth_diff_grid.png        — (mean_pred − y_true) / sd_pred on (x̂, ŷ) grid (fullmocks)
  redshift_hist_tophat.png          — pull histograms in 9 log-spaced redshift bins (fullmocks)
  tophat_highpull.png               — scatter with pull > 4 galaxies highlighted (fullmocks)
```

---

## Workflow

Each step produces plots and diagnostic statistics to `output/<run>/` for validation and paper inclusion. The checks in Step 7 should be applied after every step where outputs are produced, not only after MCMC.


### Steps 1–3: Sample Selection

Fit a noise- and truncation-corrected GMM to the observed phase space, sweep the
derived ellipse cuts to identify the slope-stability plateau, and automatically
choose the loosest stable cut value for each parameter.

→ See [Selection.md](Selection.md) for the full algorithm and usage.

---

### Steps 4–6: TFR Fitting

Prepare the Stan JSON input from the chosen selection cuts, compile the Stan tophat
and normal models, and run MCMC sampling to infer the TFR parameters.

→ See [TFFit.md](TFFit.md) for the full algorithm and usage.

---

### Step 7: Predict Absolute Magnitudes

Given the fitted TFR posterior, compute the posterior predictive mean and
uncertainty of the latent absolute magnitude y_* for each galaxy by
marginalizing over MCMC draws.

→ See [Predict.md](Predict.md) for the full algorithm and usage.

---


## RUN Procedures

Each run documents the exact command sequence used, the data set, and the
selection parameters chosen.

### DR1 — SGA-2020 (DESI DR1)

First end-to-end run on real DESI data using the
`SGA-2020_iron_Vrot_VI_corr.fits` catalog.

→ See [DR1.md](DR1.md) for the full command sequence and output file listing.

---

## NERSC Batch Jobs

Fitting can also be run as a SLURM batch job on NERSC:

```bash
sbatch test.sh
```

This workflow is still being refined. See `test.sh` for the current job configuration.
