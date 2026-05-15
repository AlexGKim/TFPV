# Color-Correction Run: SGA-2020

This document records the full command sequence for fitting and predicting using
the color-correction TFR model on the `SGA-2020_iron_Vrot_VI_corr_v5.fits` dataset.

**Prerequisites:** This run requires prior completion of Steps 1–4 of [DR1.md](DR1.md),
which establish the phase-space selection region and produce `output/$RUN/input.json`
with x, y, z, sigma_x, sigma_y, sigma_z, and c_bar_obs (computed by `desi_data.py`
with the `--color` flag).

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr_v5.fits   # input FITS catalog
export RUN=DR1_v5_color                               # output directory name: output/$RUN/
export CONFIG=configs/dr1_v5_color.json               # pipeline config
```

The color run shares the phase-space selection from the baseline tophat fit.
Only Steps 4 (data prep with color), 5c (color model compilation), and 6+ change.

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

## Step 5c: Compile Stan models (color variant)

Run from inside the `../../cmdstan/` directory:

```bash
cd ../../cmdstan
make ../TFPV/ariel/color
cd ../TFPV/ariel
```

---

## Step 6c: Run MCMC sampling (color model)

```bash
./color sample num_warmup=500 num_samples=1000 num_chains=4 \
    adapt save_metric=1 \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/color.csv
```

Sampling may take longer than tophat due to the additional z-band data and
color parameters (γ, δ, μ_c, τ_c).

---

## Step 7c: Diagnose and visualize (color model)

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/$RUN/color_?.csv > output/$RUN/stansummary_color.txt
../../cmdstan/bin/diagnose    output/$RUN/color_?.csv > output/$RUN/diagnose_color.txt

# Corner plots
python corner.py --run $RUN
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

## Step 8c: Predict absolute magnitudes (color model)

```bash
# via config
python color_predict.py --run-dir output/$RUN --xonly

# via flags (if no config available)
python color_predict.py --run-dir output/$RUN --xonly
```

The script reads:
- `output/$RUN/config.json` — phase-space selection and FITS path
- `output/$RUN/input.json` — bounds (y_min, y_max), mean_x, and z-band data
- `output/$RUN/color_?.csv` — posterior MCMC draws

Per run, the following diagnostic plots are written:

| File | Description |
|------|-------------|
| `output/$RUN/color_grid.png` | mean residual on (x̂, ŷ) grid — MAIN sample |
| `output/$RUN/color_grid_full.png` | mean residual on (x̂, ŷ) grid — full input |
| `output/$RUN/redshift_grid_color.png` | mean redshift on (x̂, ŷ) grid |
| `output/$RUN/redshift_color.png` | residual vs. redshift scatter |
| `output/$RUN/variance_redshift_color.png` | prediction variance vs. redshift |
| `output/$RUN/variance_xhat_color.png` | prediction variance vs. x̂ |

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
