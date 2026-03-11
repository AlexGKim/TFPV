# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian statistical modeling of the Tully-Fisher Relation (TFR) for peculiar velocity analysis. The TFR is an empirical relation between galaxy rotation velocity (x = log10(V/100 km/s)) and absolute magnitude (y). This project does two things:

1. **Fit the TFR**: Infer TFR parameters from galaxy rotation velocity and absolute magnitude data, accounting for observational selection effects (magnitude limits and oblique plane cuts in x-y space). Two models are fit, differing in the assumed distribution of the latent true magnitude y_TF: a tophat model (`tophat.stan`) and a normal model (`normal.stan`).

2. **Infer absolute magnitudes**: Given the fitted TFR parameters and observed rotation velocities, infer absolute magnitudes for individual galaxies.

## Documentation

The statistical models are described in the `doc/` directory:

- `doc/model1.tex`: The model for the Tully-Fisher relationship
- `doc/model2.tex`: The model to infer absolute magnitude from observed rotation velocity and fit the Tully-Fisher relationship
- `doc/model3.tex`: The general model extended with peculiar velocities

## Output Directory Structure

All data products live under `output/<run>/` where `<run>` is a short name for the
dataset and selection-cut combination (e.g. `ariel`, `DESI`, `ariel_tight`).
Every run directory uses the same standard filenames:

```
output/<run>/
  config.json          ← selection parameters used (for reproducibility)
  input.json           ← Stan data
  init.json            ← Stan initial conditions
  data.png             ← galaxy scatter plot with selection cuts
  tophat_1.csv … tophat_4.csv   ← MCMC chains (tophat model)
  normal_1.csv  … normal_4.csv  ← MCMC chains (normal model)
  tophat.png           ← corner plot (tophat posterior)
  normal.png           ← corner plot (normal posterior)
```

## Workflow

### 1. Prepare Data

Each data source has a dedicated prep script following the `{source}_data.py` convention.
Use `--run <name>` to write all outputs to `output/<name>/` and save `config.json`
recording the exact selection parameters used.

This step is iterative: inspect `output/<run>/data.png`, adjust selection parameters,
and rerun until satisfied.

```bash
python ariel_data.py --run ariel
python desi_data.py  --run DESI

# Override selection parameters
python ariel_data.py --run ariel_tight --haty_max -18.0 --haty_min -24.0 \
  --slope_plane -8.5 --intercept_plane -20.5 --intercept_plane2 -19.1
python desi_data.py --run DESI_z01 --haty_max -19.0 --haty_min -22.0 --z_obs_min 0.01
```

For **fullmocks** (AbacusSummit simulations), use `fullmocks_data.py`. Each FITS file
`TF_extended_AbacusSummit_base_c???_ph???_r???_z0.11.fits` produces its own run directory
named after the simulation ID (e.g. `c000_ph000_r000`):

```bash
# Process all matching FITS files in --dir
python fullmocks_data.py --dir /path/to/mocks

# Process only the first file (debugging)
python fullmocks_data.py --dir /path/to/mocks --one

# Override selection parameters or subsample size
python fullmocks_data.py --dir /path/to/mocks --n_objects 5000 --random_seed 42
```

### 2. Compile Stan Models

Stan models must be compiled with CmdStan before running. Run from inside the `../../cmdstan/` directory:

```bash
make ../TFPV/ariel/tophat
make ../TFPV/ariel/normal
```

To compile on GPU on NERSC requires
```bash
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64
make STAN_OPENCL=TRUE ../TFPV/ariel/tophat; cp ../TFPV/ariel/tophat ../TFPV/ariel/tophat_g
```

### 3. Run MCMC Sampling

#### Local Execution

Chains go into `output/<run>/` using standard filenames:

```bash
# Ariel mock data, 4 chains
  ./tophat sample num_warmup=500 num_samples=500  num_chains=4 adapt save_metric=1 data file=output/ariel/input.json init=output/ariel/init.json \
    output file=output/ariel/tophat.csv

# DESI data, 4 chains
  ./tophat sample num_warmup=500 num_samples=500 num_chains=4 adapt save_metric=1 data file=output/DESI/input.json init=output/DESI/init.json \
    output file=output/DESI/tophat.csv
```

#### NERSC Batch Jobs

Fitting can also be run as a batch job on NERSC using SLURM. This workflow is still being refined.

```bash
sbatch test.sh
```

### 4. Diagnose and Visualize Fit

After fitting, check chain quality and inspect the posterior:

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/DESI/tophat_?.csv
../../cmdstan/bin/diagnose output/DESI/tophat_?.csv

# Corner plot — writes output/<run>/tophat.png
python corner.py --run DESI --model tophat
python corner.py --run ariel --model tophat
```

### 5. Predict Absolute Magnitudes

```bash
python predict.py --run DESI --model tophat --source DESI
python predict.py --run ariel --model tophat --source ariel

# fullmocks: reads FITS from --dir, compares predictions to R_ABSMAG_SB26_TRUE
python predict.py --run c000_ph000_r000 --model normal --source fullmocks
python predict.py --run c000_ph000_r000 --model normal --source fullmocks \
  --n_objects 5000 --dir /path/to/mocks
```

`--source fullmocks` produces two grid plots in `output/<run>/`:
- `{model}_grid.png` — mean_pred − yhat_obs (observed residual)
- `{model}_truth_diff_grid.png` — mean_pred − R_ABSMAG_SB26_TRUE (prediction vs simulation truth)

Use `--n_objects` (any source) to limit the number of galaxies used for prediction.

## Architecture

### Data Flow

```
Raw Data → {source}_data.py ⇄ (inspect plot, adjust cuts) → JSON → Stan MCMC → CSV chains → diagnose/stansummary/corner.py → predict.py
```

### Stan Model Design

Two models are fit to the data, differing in the assumed distribution of the latent true magnitude parameter y_TF:

**`tophat.stan`** is the tophat model. Key design decisions:
- Latent true magnitudes (y_TF) are marginalized over with Gauss-Legendre quadrature
- Selection probability is computed as `P_binormal_strip`: the bivariate normal probability mass in the strip between two oblique parallel planes in (x, y) space
- Owen's T function is used for bivariate normal CDF evaluation
- Parameters: `slope`, `intercept[N_bins]`, `sigma_int_x`, `sigma_int_y`

**`normal.stan`** is the normal model.
- Parameters: `slope`, `intercept[N_bins]`, `sigma_int_x`, `sigma_int_y`, `mu_y_TF`, `tau`

### JSON Data Format

All `{source}_data.py` scripts produce a standardized JSON consumed by all Stan models. Key fields:
- `x`, `sigma_x`: observed log rotation velocities and uncertainties
- `y`, `sigma_y`: observed absolute magnitudes and uncertainties
- `z` *(optional)*: observed redshift
- `haty_max`, `haty_min`: magnitude selection limits
- `slope_plane`, `intercept_plane`, `intercept_plane2`: oblique cut parameters
- `mu_y_TF`, `tau`: prior mean and std on true magnitudes
- `N_bins`: number of redshift bins for `intercept[N_bins]`

### Dependencies

- **CmdStan**: Stan compiler and sampler (expected at `../../cmdstan/`)
- **Python**: numpy, scipy, pandas, matplotlib, astropy (FITS), chainconsumer (corner plots), cmdstanpy