# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian statistical modeling of the Tully-Fisher Relation (TFR) for peculiar velocity analysis. The TFR is an empirical relation between galaxy rotation velocity (x = log10(V/100 km/s)) and absolute magnitude (y). The models infer TFR parameters while accounting for observational selection effects (magnitude limits and oblique plane cuts in x-y space).

## Commands

### Compile Stan Models

Stan models must be compiled with CmdStan before running. From the repo root (or parent directory containing `cmdstan/`):

```bash
# Compile base model (truncated bivariate normal with selection effects)
make ../TFPV/ariel/base

# Compile normal model (simplified, unbounded)
make ../TFPV/ariel/normal
```

### Run MCMC Sampling

```bash
# Mock data
./base sample data file=MOCK_n10000_input.json init=MOCK_n10000_init.json \
  output file=MOCK_n10000_base.csv

# DESI data (4 chains)

./base sample data file=DESI_input.json init=DESI_init.json \
  output file=DESI_base.csv &

```

### Diagnose and Summarize Chains

```bash
../cmdstan/bin/stansummary DESI_base_?.csv
../cmdstan/bin/diagnose DESI_base_?.csv
```

### Prepare Data

```bash
# Mock data from CSV
python ariel_data.py

# DESI real data from FITS
python desi_data.py
```

### Analysis and Visualization

```bash
# Corner plot of posterior chains
python corner.py 'DESI_base_?.csv' --output DESI_base.png

# Posterior predictions
python predict.py

```

## Architecture

### Data Flow

```
Raw Data → Python prep scripts → JSON (Stan input) → Stan MCMC → CSV chains → Analysis scripts
```

1. **Data preparation** (`ariel_data.py`, `desi_data.py`): Reads CSV or FITS data, applies coordinate transforms, encodes selection cuts, outputs JSON for Stan initial conditions.

2. **Stan models** (`base.stan`, `normal.stan`): Compiled to executables `./base` and `./normal`. Take JSON input/init, output CSV chains.

3. **Analysis** (`corner.py`, `predict.py`, `plot_*.py`): Read Stan CSV output for posteriors and predictions.

### Stan Model Design

**`base.stan`** is the tophat model for the latent parameters. Key design decisions:
- Latent true magnitudes (y_TF) are marginalized over with Gauss-Legendre quadrature
- Selection probability is computed as `P_binormal_strip`: the bivariate normal probability mass in the strip between two oblique parallel planes in (x, y) space
- Owen's T function is used for bivariate normal CDF evaluation
- Parameters: `slope`, `intercept[N_bins]`, `sigma_int_x`, `sigma_int_y`

**`normal.stan`** is a normal model for the latent parameter distribution.
- Parameters: `slope`, `intercept[N_bins]`, `sigma_int_x`, `sigma_int_y`, `mu_y_TF`, `tau`

### JSON Data Format

Key fields in Stan input JSON:
- `x`, `sigma_x`: observed log velocities and uncertainties
- `y`, `sigma_y`: observed magnitudes and uncertainties
- `haty_max`, `haty_min`: magnitude selection limits
- `slope_plane`, `intercept_plane`, `intercept_plane2`: oblique cut parameters
- `mu_y_TF`, `tau`: prior mean and std on true magnitudes
- `N_bins`: number of redshift bins for `intercept[N_bins]`

### Dependencies

- **CmdStan**: Stan compiler and sampler (expected at `../cmdstan/`)
- **Python**: numpy, scipy, pandas, matplotlib, astropy (FITS), chainconsumer (corner plots), cmdstanpy
