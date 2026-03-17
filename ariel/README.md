# Tully-Fisher Peculiar Velocity (TFPV) — Ariel

Bayesian statistical modeling of the Tully-Fisher Relation (TFR) for peculiar velocity analysis. The TFR is an empirical relation between galaxy rotation velocity (`x = log10(V/100 km/s)`) and absolute magnitude (`y`).

This project does two things:

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
  ariel_data.py          — data prep for Ariel mock catalog
  desi_data.py           — data prep for DESI survey data
  fullmocks_data.py      — data prep for AbacusSummit FITS mocks
  tophat.stan            — Stan model (tophat prior)
  normal.stan            — Stan model (normal prior)
  corner.py              — posterior corner plots (uses chainconsumer)
  predict.py             — infer absolute magnitudes from fitted TFR
  doc/                   — LaTeX descriptions of the statistical models
  output/                — all data products (gitignored)
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
```

---

## Workflow

### Step 1: Prepare Data

Each data source has a dedicated prep script. Use `--run <name>` to write all outputs to `output/<name>/` and save `config.json` recording the exact selection parameters used.

```bash
python ariel_data.py --run ariel
python desi_data.py  --run DESI

# Override selection parameters
python ariel_data.py --run ariel_tight --haty_max -18.0 --haty_min -24.0 \
  --slope_plane -8.5 --intercept_plane -20.5 --intercept_plane2 -19.1
python desi_data.py --run DESI_z01 --haty_max -19.0 --haty_min -22.0 --z_obs_min 0.01
```

Inspect `output/<run>/data.png` after each run and adjust selection parameters until satisfied.

For **AbacusSummit fullmocks**, each FITS file named
`TF_extended_AbacusSummit_base_c???_ph???_r???_z0.11.fits`
produces its own run directory named after the simulation ID extracted from the filename
(e.g. `c000_ph000_r000`). On NERSC, these files live under
`/global/cfs/cdirs/desi/science/td/pv/mocks/TF_mocks/fullmocks/v0.5.4/`.

**FITS columns used:**

| Column | Description | Stan field |
|---|---|---|
| `LOGVROT` | log10(V_rot / km/s) | `x = LOGVROT − 2` |
| `LOGVROT_ERR` | uncertainty on LOGVROT | `sigma_x` |
| `R_ABSMAG_SB26` | R-band absolute magnitude (SB26) | `y` |
| `R_ABSMAG_SB26_ERR` | uncertainty on R_ABSMAG_SB26 | `sigma_y` |
| `ZOBS` | observed redshift | `z_obs` |
| `MAIN` | boolean — only `MAIN=True` rows are used | — |

**Default selection parameters** (override any with the corresponding flag):

| Parameter | Default | Flag |
|---|---|---|
| `haty_max` | −20.0 | `--haty_max` |
| `haty_min` | −21.8 | `--haty_min` |
| `z_obs_min` | 0.03 | `--z_obs_min` |
| `z_obs_max` | 0.10 | `--z_obs_max` |
| `slope_plane` | −6.5 | `--slope_plane` |
| `intercept_plane` | −20.0 | `--intercept_plane` |
| `intercept_plane2` | −19.0 | `--intercept_plane2` |
| `n_objects` | 5000 | `--n_objects` |

```bash
# Process all matching FITS files in --dir
python fullmocks_data.py --dir /path/to/mocks

# Process a single specific FITS file
python fullmocks_data.py --file /path/to/mocks/TF_extended_AbacusSummit_base_c000_ph000_r000_z0.11.fits

# Process only the first matching file in --dir (for debugging)
python fullmocks_data.py --dir /path/to/mocks --one

# Override selection parameters or subsample size
python fullmocks_data.py --dir /path/to/mocks --n_objects 5000 --random_seed 42

# NERSC — all files
python fullmocks_data.py \
  --dir /global/cfs/cdirs/desi/science/td/pv/mocks/TF_mocks/fullmocks/v0.5.4

# NERSC — single file
python fullmocks_data.py \
  --file /global/cfs/cdirs/desi/science/td/pv/mocks/TF_mocks/fullmocks/v0.5.4/TF_extended_AbacusSummit_base_c000_ph000_r000_z0.11.fits
```

---

### Step 2: Compile Stan Models

Run from inside the `../../cmdstan/` directory:

```bash
make ../TFPV/ariel/tophat
make ../TFPV/ariel/normal
```

To compile with GPU support on NERSC:

```bash
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64
make STAN_OPENCL=TRUE ../TFPV/ariel/tophat
cp ../TFPV/ariel/tophat ../TFPV/ariel/tophat_g
```

---

### Step 3: Run MCMC Sampling

#### First run — adapt and save metric

On the first run (no prior metric available), let Stan adapt the mass matrix and step size, and save the resulting metric for reuse:

```bash
./tophat sample num_warmup=500 num_samples=500 num_chains=4 \
  adapt save_metric=1 \
  data file=output/ariel/input.json \
  init=output/ariel/init.json \
  output file=output/ariel/tophat.csv
```

This writes `tophat_metric.json` (containing `"inv_metric"`) alongside the chain CSVs in `output/ariel/`. The adapted step size is recorded in the CSV chain headers.

#### Subsequent runs — load saved metric and fixed step size

Once a metric and step size have been saved, subsequent runs can skip most of the warmup:

```bash
./tophat sample algorithm=hmc metric=diag_e \
  metric_file=output/ariel/tophat_metric.json \
  stepsize=0.11871086 \
  num_warmup=50 num_samples=500 num_chains=4 \
  data file=output/c000_ph000_r001/input.json \
  init=output/c000_ph000_r001/init.json \
  output file=output/c000_ph000_r001/tophat.csv
```

Notes:
- `metric_file` must be a JSON file containing only the key `"inv_metric"`.
- Retrieve the step size from the CSV chain header (search for `stepsize=`) or from `stansummary` output.
- `num_warmup=50` is sufficient when the metric and step size are pre-set.

#### DESI data example

```bash
./tophat sample num_warmup=500 num_samples=500 num_chains=4 \
  adapt save_metric=1 \
  data file=output/DESI/input.json \
  init=output/DESI/init.json \
  output file=output/DESI/tophat.csv
```

---

### Step 4: Diagnose and Visualize

Check chain quality and inspect the posterior:

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/DESI/tophat_?.csv
../../cmdstan/bin/diagnose   output/DESI/tophat_?.csv

# Corner plots — writes output/<run>/tophat.png or normal.png
python corner.py --run DESI  --model tophat
python corner.py --run ariel --model tophat
```

---

### Step 5: Predict Absolute Magnitudes

```bash
python predict.py --run DESI           --model tophat --source DESI
python predict.py --run ariel          --model tophat --source ariel

# fullmocks: reads FITS from --dir, compares predictions to R_ABSMAG_SB26_TRUE
python predict.py --run c000_ph000_r000 --model tophat --source fullmocks \
  --dir /path/to/mocks --n_objects 100000

# fullmocks with looser selection (expand magnitude and plane-cut windows)
python predict.py --run c000_ph000_r000 --model tophat --source fullmocks \
  --dir /path/to/mocks --n_objects 100000 \
  --delta_haty_min -0.5 --delta_haty_max 0.5 \
  --plane_cut --delta_intercept_plane -0.05 --delta_intercept_plane2 0.05 \
  --delta_z_obs_min -0.03
```

The `--source fullmocks` flag produces output plots in `output/<run>/`:
- `{model}_grid.png` — `mean_pred − yhat_obs` (observed residual on x–y grid)
- `{model}_truth_diff_grid.png` — `mean_pred − R_ABSMAG_SB26_TRUE` (prediction vs. simulation truth)
- `redshift_{model}.png` — pull vs. redshift scatter with weighted mean
- `redshift_hist_{model}.png` — pull histograms in 9 log-spaced redshift bins

**Selection flags for `--source fullmocks`** (all default to 0, i.e. training selection):

| Flag | Effect |
|---|---|
| `--delta_haty_min FLOAT` | shift `haty_min` cut (negative = looser) |
| `--delta_haty_max FLOAT` | shift `haty_max` cut (positive = looser) |
| `--delta_z_obs_min FLOAT` | shift `z_obs_min` cut (negative = looser) |
| `--delta_z_obs_max FLOAT` | shift `z_obs_max` cut (positive = looser) |
| `--plane_cut` | apply oblique plane cut from `input.json` (default: off) |
| `--delta_intercept_plane FLOAT` | shift lower plane intercept (negative = looser) |
| `--delta_intercept_plane2 FLOAT` | shift upper plane intercept (positive = looser) |

Use `--n_objects` (any source) to limit the number of galaxies used for prediction.

---

## NERSC Batch Jobs

Fitting can also be run as a SLURM batch job on NERSC:

```bash
sbatch test.sh
```

This workflow is still being refined. See `test.sh` for the current job configuration.
