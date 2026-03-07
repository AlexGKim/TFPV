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

## Workflow

### 1. Prepare Data

Each data source has a dedicated prep script following the `{source}_data.py` convention. The script reads source-specific CSV or FITS files, applies selection cuts, and produces a Stan input JSON and a plot of the galaxy data in (x, y) space with selection cut boundaries overlaid.

This step is iterative: inspect the plot, adjust selection cut parameters (`haty_max`, `haty_min`, `slope_plane`, `intercept_plane`, `intercept_plane2`), and rerun until satisfied.

```bash
python ariel_data.py
python desi_data.py

# Selection parameters can be passed as command-line arguments
python ariel_data.py --haty_max -18.0 --haty_min -24.0 --slope_plane -8.5 \
  --intercept_plane -20.5 --intercept_plane2 -19.1
python desi_data.py --haty_max -19.0 --haty_min -22.0 --z_obs_min 0.01
```

### 2. Compile Stan Models

Stan models must be compiled with CmdStan before running. Run from inside the `../../cmdstan/` directory:

```bash
make ../TFPV/ariel/tophat
make ../TFPV/ariel/normal
```

### 3. Run MCMC Sampling

#### Local Execution

```bash
# Mock data
./tophat sample data file=ariel_input.json init=ariel_init.json \
  output file=ariel_tophat.csv

# DESI data
./tophat sample data file=DESI_input.json init=DESI_init.json \
  output file=DESI_tophat.csv
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
../../cmdstan/bin/stansummary DESI_tophat_?.csv
../../cmdstan/bin/diagnose DESI_tophat_?.csv

# Corner plot of posterior
python corner.py 'DESI_tophat_?.csv' --output DESI_tophat.png
```

### 5. Predict Absolute Magnitudes

The second stage of processing infers absolute magnitudes from measured rotation velocities given the fitted TFR model parameters. Details of inputs, outputs, and usage are to be determined.

```bash
python predict.py
```

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