# Tully-Fisher Peculiar Velocity (TFPV) — Ariel

Bayesian statistical modeling of the Tully-Fisher Relation (TFR) for peculiar velocity analysis. The TFR is an empirical relation between galaxy rotation velocity (`x = log10(V/100 km/s)`) and absolute magnitude (`y`).

This project does three things:
0. **Sample selection**: There are normal galaxies occupy a compact
region of the data phase space who obey a Tully-Fisher
relation with a specific slope.  There are a small
froction of other galaxies
that do not follow the Tully-Fisher relation that for the
most part occupy a different region of data phase space.
An analysis of the full sample would yield a different slope
with a poorer quality of fit.  The objective is to determine
sample selection criteria that gives high completeness
and purity of normal galaxies.
1. **Fit the TFR**: Infer TFR parameters from galaxy rotation velocity and absolute magnitude data, accounting for observational selection effects (magnitude limits and oblique plane cuts in x–y space).
2. **Infer absolute magnitudes**: Given fitted TFR parameters and observed rotation velocities, infer absolute magnitudes for individual galaxies.

Two Stan models are provided, differing in the assumed distribution of the latent true magnitude y_TF:

- **`tophat.stan`**: Uniform (tophat) prior on y_TF; latent magnitudes marginalized with Gauss-Legendre quadrature.
- **`normal.stan`**: Gaussian prior on y_TF; adds hyperparameters `mu_y_TF` and `tau`.

The statistical models are described in detail in `doc/model1.tex`, `doc/model2.tex`, and `doc/model3.tex`.

---

## Dependencies

- **CmdStan** — Stan compiler and sampler, expected at `../../cmdstan/` relative to this directory
- **Python** — numpy, scipy, pandas, matplotlib, astropy, chainconsumer, cmdstanpy

---

## Repository Layout

```
ariel/
  ariel_data.py                      — data prep for Ariel mock catalog
  desi_data.py                       — data prep for DESI survey data
  fullmocks_data.py                  — data prep for AbacusSummit FITS mocks
  tophat.stan                        — Stan model (tophat prior)
  normal.stan                        — Stan model (normal prior)
  corner.py                          — posterior corner plots (uses chainconsumer)
  predict.py                         — infer absolute magnitudes from fitted TFR
  cut_sweep.py                       — quantitative selection cut optimisation (grid sweep)
  plot_magnitude_predictions.py      — publication-quality magnitude prediction plots (fullmocks)
  plot_desi_magnitude_predictions.py — magnitude prediction plots for DESI
  figs/distributions.py              — figures of latent variable distributions (paper figures)
  doc/                               — LaTeX descriptions of the statistical models
  output/                            — all data products (gitignored)
```

**Possibly deprecated** (may no longer be needed; retained for reference):

```
  plot_dwarf.py     — diagnostic scatter for dwarf-galaxy outliers
  integral_test.py  — unit tests for bivariate-normal strip integral
```

---

## Output Directory Structure

All data products live under `output/<run>/` where `<run>` is a short name for the dataset and selection-cut combination (e.g. `ariel`, `DESI`, `c000_ph000_r000`):

```
output/<run>/
  config.json                       — selection parameters used (for reproducibility)
  input.json                        — Stan data
  init.json                         — Stan initial conditions
  data.png                          — galaxy scatter plot with selection cuts
  tophat_1.csv … tophat_4.csv       — MCMC chains (tophat model)
  normal_1.csv  … normal_4.csv      — MCMC chains (normal model)
  tophat_metric.json                — saved mass matrix (written by adapt save_metric=1)
  tophat.png                        — corner plot (tophat posterior)
  normal.png                        — corner plot (normal posterior)
  tophat_grid.png                   — mean_pred − yhat_obs on (x, y) grid
  tophat_truth_diff_grid.png        — (mean_pred − y_true) / sd_pred on (x, y) grid
  redshift_tophat.png               — pull vs. redshift scatter
  redshift_hist_tophat.png          — pull histograms in 9 log-spaced redshift bins
  tophat_highpull.png               — scatter in (x̂, ŷ) with pull > 4 galaxies highlighted
  cut_sweep.csv                     — grid sweep results (cut_sweep.py)
  cut_sweep_1d.png                  — 1-D slope profiles per selection parameter
  cut_sweep_2d_<p1>_<p2>.png       — 2-D slices: slope / volatility / log-likelihood
  cut_sweep_best_config.json        — optimal cut parameters from the sweep
```

---

## Fullmocks Quick-Start

End-to-end for a single AbacusSummit simulation file. Replace the path and run name
throughout.

```bash
export FITS=/path/to/TF_extended_AbacusSummit_base_c000_ph000_r000_z0.11.fits
export RUN=c000_ph000_r000

# Steps 1–3: Sample selection (see Selection.md)
python selection_ellipse.py --file $FITS --run $RUN
python ellipse_sweep.py --source fullmocks --fits_file $FITS --run $RUN
python selection_criteria.py --source fullmocks --fits_file $FITS --run $RUN

# Steps 4–6: TFR fitting (see TFFit.md)
python fullmocks_data.py --file $FITS --run $RUN <cuts from selection_criteria.json>
make -C ../../cmdstan ../TFPV/ariel/tophat ../TFPV/ariel/normal
./tophat sample ... data file=output/$RUN/input.json ...

# 6. Diagnose + corner plot
../../cmdstan/bin/stansummary output/$RUN/tophat_?.csv
python corner.py --run $RUN --model tophat

# 7. Predict
python predict.py --run $RUN --model tophat --source fullmocks --dir $(dirname $FITS)
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

### Step 6: Predict Absolute Magnitudes

Given the fitted TFR posterior, compute the posterior predictive mean and
uncertainty of the latent absolute magnitude y_* for each galaxy by
marginalizing over MCMC draws.

→ See [Predict.md](Predict.md) for the full algorithm and usage.

---

### Step 7: Diagnose and Visualize

These checks should be applied after every step that produces outputs (data prep, cut sweep, MCMC, predictions), not only at the end. At minimum, run them after MCMC and after prediction.

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/DESI/tophat_?.csv
../../cmdstan/bin/diagnose   output/DESI/tophat_?.csv

# Corner plots — writes output/<run>/tophat.png or normal.png
python corner.py --run DESI  --model tophat
python corner.py --run ariel --model tophat
```

---

## NERSC Batch Jobs

Fitting can also be run as a SLURM batch job on NERSC:

```bash
sbatch test.sh
```

This workflow is still being refined. See `test.sh` for the current job configuration.
