# DR2 Run: SGA-2020 loa

This document records the full command sequence for the DR2 run on the
`SGA-2020_loa_Vrot_VI_v0.fits` dataset using the two-color TFR model.

## Setup

```bash
export FITS=data/SGA-2020_loa_Vrot_VI_v0.fits   # input FITS catalog
export RUN=DR2_v0_2color                          # output directory: output/$RUN/
export CONFIG=configs/dr2_v0_2color.json          # pipeline config
```

---

## Step 1: Estimating the core distribution

Fit a noise- and truncation-corrected 2-component GMM to the (x, y) phase
space to estimate the TFR core selection boundary.

```bash
python selection_ellipse.py --config $CONFIG

# via flags
python selection_ellipse.py --file $FITS --run $RUN --source DESI \
    --z_obs_min 0.01 --z_obs_max 0.065 --haty_min -22 --haty_max -18.5
```

Inspect:

```bash
open output/$RUN/selection_ellipse.png
```

---

## Step 2: MLE fit and pull-profile diagnostic

```bash
python select_v2.py --config $CONFIG --exe ./tophat

# via flags
python select_v2.py --run $RUN --fits_file $FITS --exe ./tophat \
    --z_obs_min 0.01 --z_obs_max 0.065
```

Inspect:

```bash
open output/$RUN/select_v2_pull.png
```

---

## Step 3: Set fiducial selection criteria

```bash
python set_fiducial.py --run $RUN
```

**Fiducial parameters chosen for DR2_v0_2color:**

| Parameter | Value |
|-----------|-------|
| `haty_min` | −22.0 |
| `haty_max` | −18.5 |
| `z_obs_min` | 0.01 |
| `z_obs_max` | 0.065 |
| `slope_plane` | −6.643332400024041 |
| `intercept_plane` | −20.358756228023335 |
| `intercept_plane2` | −18.053911959498492 |
| `n_sigma_perp` | 3.0 |

Inspect:

```bash
open output/$RUN/select_v2_fiducial_pull.png
```

---

## Step 3b: Export run config

```bash
python export_config.py --run $RUN --out $CONFIG
```

> **Note:** `export_config.py` does not preserve the `dust_pickle` key.
> After running it, re-add the following to `$CONFIG`:
> ```json
> "dust_pickle": "data/loa_internalDust_nokcorr_mcmc.pickle"
> ```

---

## Steps 4–8: Two-color model fit and prediction

Follow **[2COLOR.md](2COLOR.md) Steps 4–8** with:

```bash
export FITS=data/SGA-2020_loa_Vrot_VI_v0.fits
export RUN=DR2_v0_2color
export CONFIG=configs/dr2_v0_2color.json
```

The `dust_pickle` key in `$CONFIG` ensures the covariance matrix uses
`d_err_r ≈ 0.2173 mag` (from `data/loa_internalDust_nokcorr_mcmc.pickle`)
rather than the iron default.

---

## File Reference

| File | Purpose |
|------|---------|
| `data/SGA-2020_loa_Vrot_VI_v0.fits` | Input FITS catalog (35,666 galaxies) |
| `data/loa_internalDust_nokcorr_mcmc.pickle` | Internal dust MCMC (d_err_r ≈ 0.2173 mag) |
| `configs/dr2_v0_2color.json` | Pipeline config for this run |
| `2color.stan` | Stan model: quadrivariate TFR with two color factors |
| `color_predict.py --model 2color` | Posterior predictive computation |
| [2COLOR.md](2COLOR.md) | Two-color workflow (Steps 4–8) |
| [DR1.md](DR1.md) | DR1 iron run record (Steps 1–3b reference) |
