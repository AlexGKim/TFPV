# Color-Correction Run: SGA-2020

This document records the full command sequence for fitting and predicting using
the color-correction TFR model on the `SGA-2020_iron_Vrot_VI_corr_v5.fits` dataset.

The color-correction model (Appendix C of the paper) extends the baseline tophat
fit by including z-band absolute magnitudes (ẑ) and modeling the luminosity–color
correlation. See [COLOR.md](COLOR.md) for detailed workflow.

## Full Workflow

The color run reuses the phase-space selection from the baseline tophat fit
([DR1.md](DR1.md), Steps 1–3b) but requires running Steps 4 (with color data),
5 (compilation), and new steps 5d–8c (MAP initialization, sampling, prediction).

### Workflow Summary

| Phase | Steps | How |
|-------|-------|-----|
| A — Setup | 1–3b | Complete [DR1.md](DR1.md) Steps 1–3b first (selection ellipse, pull profile, fiducial) |
| B — Color | 3b–8c | Run Steps 3b–8c below (config export, color data prep, compilation, sampling, prediction) |

---

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr_v5.fits   # input FITS catalog
export RUN=DR1_v5_color                               # output directory name: output/$RUN/
export CONFIG=configs/dr1_v5_color.json               # pipeline config
```

---

## Step 1: Estimating the core distribution

Fit a noise- and truncation-corrected 2-component GMM to the (x, y) phase
space to estimate the TFR core selection boundary.

```bash
# via config
python selection_ellipse.py --config $CONFIG

# via flags
python selection_ellipse.py --file $FITS --run $RUN --source DESI \
    --z_obs_min 0.03 --z_obs_max 0.08 --haty_min -23 --haty_max -18
```

Inspect the output:

```bash
open output/$RUN/selection_ellipse.png
```

---

## Step 2: MLE fit and pull-profile diagnostic

Run Stan MAP optimisation on the 3σ-ellipse selection and produce a pull
profile over all catalog objects. Use the plot to guide the choice of the
final magnitude window.

```bash
# via config (--exe must still be passed explicitly)
python select_v2.py --config $CONFIG --exe ./tophat

# via flags
python select_v2.py --run $RUN --fits_file $FITS --exe ./tophat \
    --z_obs_min 0.03 --z_obs_max 0.08
```

Inspect the pull profile:

```bash
open output/$RUN/select_v2_pull.png
```

---

## Step 3: Set fiducial selection criteria

Based on the pull profile, interactively choose the perpendicular cut width
(in σ units) and the magnitude window, then write
`output/$RUN/select_v2_fiducial.json`.

```bash
python set_fiducial.py --run $RUN
```

The script prints the 1σ reference values and prompts for `n_sigma_perp`,
`haty_min`, `haty_max`, `z_obs_min`, and `z_obs_max`.

Inspect the pull profile with the cuts:

```bash
open output/$RUN/select_v2_fiducial_pull.png
```

---

## Step 3b: Export run config

After completing the interactive fiducial step, capture all parameter choices
in a portable config file:

```bash
python export_config.py --run $RUN --out $CONFIG
```

The script reads `output/$RUN/select_v2_fiducial.json` (including the
interactively chosen cuts) and prompts for the remaining pipeline settings
(`exe`, `source`, `model`, `n_sigma`). The `fits_file` is taken automatically
from `output/$RUN/config.json` so it matches the file actually used. Commit
the resulting JSON to git — it is the permanent version record for this run.

---

## Step 4: Prepare data (color variant)

Convert the FITS file to Stan JSON format, including z-band magnitudes and color info.

```bash
# via config
python desi_data.py --config $CONFIG

# via flags
python desi_data.py --input $FITS --run $RUN \
    --haty_min -21.5 --haty_max -19.0 \
    --slope_plane -6.386925076468424 --intercept_plane -20.74814050932727 --intercept_plane2 -18.31309635087515 \
    --z_obs_min 0.03 --z_obs_max 0.08
```

`desi_data.py` always loads z-band data when available:
- Loads z-band magnitudes from the FITS catalog (columns Z_MAG_SB26_CORR or Z_MAG_SB26)
- Computes z-band absolute magnitudes and uncertainties
- Writes z-band data and sample mean color c_bar_obs to `output/$RUN/input.json`

Inspect the scatter plot to verify the selection looks correct:

```bash
open output/$RUN/data.png
```

---

## Step 5: Compile Stan models

Run from inside the `../../cmdstan/` directory:

```bash
cd ../../cmdstan
make ../TFPV/ariel/tophat
make ../TFPV/ariel/normal
make ../TFPV/ariel/color
cd ../TFPV/ariel
```

---

## Step 5d: Find MAP estimate (init_MAP.json)

The MAP (maximum a posteriori) estimate provides a warm start near the posterior
mode. Starting MCMC from the MAP rather than a hand-set `init.json` reduces
warmup requirements and avoids the sampler spending time finding the basin.

```bash
# Find MAP estimate — typically completes in < 5 minutes
./color optimize \
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

## Step 6: Run MCMC sampling (color model)

`algorithm=hmc metric=dense_e` learns the full posterior covariance during
warmup, absorbing parameter correlations (e.g. slope ↔ α_kcorr, δ_c ↔ μ_c)
and reducing the leapfrog steps needed per effective sample.

```bash
./color sample num_warmup=250 num_samples=1000 num_chains=4 \
    adapt save_metric=1 \
    algorithm=hmc metric=dense_e \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/color.csv
```

Sampling may take longer than tophat due to the additional z-band data and
color parameters (γ, δ, μ_c, τ_c).

---

## Step 7: Diagnose and visualize (color model)

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/$RUN/color_?.csv > output/$RUN/stansummary_color.txt
../../cmdstan/bin/diagnose    output/$RUN/color_?.csv > output/$RUN/diagnose_color.txt

# Corner plots
python corner.py --run $RUN --model color
```

Inspect:

```bash
cat output/$RUN/stansummary_color.txt
open output/$RUN/color.png
```

Key parameters to check:
- `gamma` — luminosity–color slope (expected: negative; redder → brighter)
- `delta_c` — color–velocity slope
- `mu_c` — mean color at x̄
- `tau_c` — intrinsic color scatter at fixed velocity

---

## Step 8: Predict absolute magnitudes (color model)

x-only is the unconditional default (no `--xonly` flag needed/accepted
anymore); pass `--full` to additionally get the full model's outputs:

```bash
python color_predict.py --run-dir output/$RUN            # x-only only
python color_predict.py --run-dir output/$RUN --full      # + full model
```

The script reads:
- `output/$RUN/config.json` — phase-space selection and FITS path
- `output/$RUN/input.json` — bounds (y_min, y_max), mean_x, and z-band data
- `output/$RUN/color_?.csv` — posterior MCMC draws

Per run, the following diagnostic plots are written (always, x-only):

| File | Description |
|------|-------------|
| `output/$RUN/color_grid_xonly.png` | mean residual on (x̂, ŷ) grid — MAIN sample, x-only |
| `output/$RUN/color_grid_xonly_full.png` | mean residual on (x̂, ŷ) grid — full input, x-only |
| `output/$RUN/redshift_grid_color.png` | mean redshift on (x̂, ŷ) grid (data-space, model-independent) |
| `output/$RUN/redshift_color_xonly.png` | residual vs. redshift scatter, x-only |
| `output/$RUN/variance_redshift_color_xonly.png` | prediction variance vs. redshift, x-only |

With `--full`, additionally:

| File | Description |
|------|-------------|
| `output/$RUN/color_grid.png` | mean residual on (x̂, ŷ) grid — MAIN sample, full model |
| `output/$RUN/color_grid_full.png` | mean residual on (x̂, ŷ) grid — full input, full model |
| `output/$RUN/redshift_color.png` | residual vs. redshift scatter, full model |
| `output/$RUN/variance_redshift_color.png` | prediction variance vs. redshift, full model |
| `output/$RUN/variance_xhat_color.png` | prediction variance vs. x̂, full model |

See [color_predict.py](color_predict.py) for implementation details.

---

## Comparison with baseline

To compare color-correction residuals against the baseline tophat model:

```bash
# After running both tophat (Step 8 in DR1.md) and color (Step 8c above)
python -c "
import numpy as np
import matplotlib.pyplot as plt
from predict import load_xy_and_uncertainties_from_desi, read_cmdstan_posterior, ystar_pp_mean_sd_tophat_vectorized
from color_predict import ystar_pp_mean_sd_color_vectorized, load_xyz_and_uncertainties_from_desi

# Load data
xhat, sigma_x, yhat, sigma_y, zhat, sigma_z, zobs = load_xyz_and_uncertainties_from_desi('$FITS')

# Tophat predictions
draws_tophat = read_cmdstan_posterior('output/$RUN/tophat_?.csv', 
    keep=['slope', 'intercept.1', 'sigma_int_x', 'sigma_int_y'], drop_diagnostics=True)
mean_tophat, sd_tophat = ystar_pp_mean_sd_tophat_vectorized(
    draws_tophat, xhat, sigma_x, y_min=-22.35, y_max=-18.0)

# Color predictions
draws_color = read_cmdstan_posterior('output/$RUN/color_?.csv',
    keep=['slope', 'intercept.1', 'sigma_int_x', 'sigma_int_y', 'gamma', 'delta_c', 'mu_c', 'tau_c'],
    drop_diagnostics=True)
import json
with open('output/$RUN/input.json') as f:
    input_data = json.load(f)
x_bar = input_data['mean_x']
mean_color, sd_color = ystar_pp_mean_sd_color_vectorized(
    draws_color, xhat, sigma_x, zhat, sigma_z, x_bar=x_bar, y_min=-22.35, y_max=-18.0)

# Residual scatter
resid_tophat = mean_tophat - yhat
resid_color = mean_color - yhat

plt.scatter(zobs, resid_tophat, alpha=0.1, s=1, label='Tophat')
plt.scatter(zobs, resid_color, alpha=0.1, s=1, label='Color')
plt.xscale('log')
plt.axhline(0, color='k', ls='--', lw=0.8)
plt.xlabel(r'\$z_{\\mathrm{obs}}\$')
plt.ylabel(r'\$M_{\\mathrm{pred}} - M_{\\mathrm{obs}}\$ (mag)')
plt.legend()
plt.tight_layout()
plt.savefig('output/$RUN/tophat_vs_color_residuals.png', dpi=150)
print(f'Tophat  σ(resid): {np.std(resid_tophat):.3f}')
print(f'Color   σ(resid): {np.std(resid_color):.3f}')
print(f'Color mean bias: {np.mean(resid_color - resid_tophat):.3f}')
"
```

---

## Running a variant with color data

To run the color-correction pipeline with custom parameters, create a config
file and pass it to the pipeline:

```bash
cp configs/dr1_v5.json configs/dr1_v5_color.json
```

Edit `configs/dr1_v5_color.json` — change `"run"` to `"dr1_v5_color"` and
ensure selection parameters match. Then run the full pipeline:

```bash
python run_pipeline.py configs/dr1_v5_color.json
```

Or run only specific steps (e.g. re-do data prep and sampling after adjusting
parameters):

```bash
python run_pipeline.py configs/dr1_v5_color.json --steps 4-8
```

---

## Implementation Notes

The color-correction model (Appendix C of the paper) extends the baseline tophat fit
by:
1. Including z-band absolute magnitudes (ẑ) as an observed quantity
2. Adding four parameters: γ (luminosity–color slope), δ (color–velocity correlation),
   μ_c (mean color), τ_c (intrinsic color scatter)
3. Computing predictions via a trivariate bivariate-Gaussian regression (Eqs. C.33–C.39)

Key assumptions:
- Color residuals are approximately Gaussian at fixed velocity
- Color scatter is independent of velocity (captured by the structural constraint in the model)
- The z-band observation improves predictions primarily when γ ≠ 0 (i.e., when color
  correlates with luminosity at fixed velocity)

---

## File Reference

| File | Purpose |
|------|---------|
| `color.stan` | Stan model implementing the color-correction likelihood |
| `color_predict.py` | Posterior predictive computation and diagnostics |
| `desi_data.py --color` | Data preparation with z-band magnitudes |
| [COLOR.md](COLOR.md) | This file — color-correction workflow |
