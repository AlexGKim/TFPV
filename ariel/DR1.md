# DR1 Run: SGA-2020

This document records the full command sequence for the DR1 run on the
`SGA-2020_iron_Vrot_VI_corr.fits` dataset.

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr.fits
export RUN=DR1
```

---

## Step 1: Estimating the core distribution

Fit a noise- and truncation-corrected 2-component GMM to the (x, y) phase
space to estimate the TFR core selection boundary.

```bash
python selection_ellipse.py --file $FITS --run $RUN --source DESI
```

Inspect the output:

```bash
open output/$RUN/selection_ellipse.png
```

The printed "Derived selection cuts" block provides starting estimates for
`haty_min`, `haty_max`, `slope_plane`, `intercept_plane`, and
`intercept_plane2` to use in the steps below.

---

## Step 2: Selection plateau search

Scan a 3D grid over (n_σ_perp, n_σ_lo, n_σ_hi) using Stan MAP optimization
to find where the MLE TFR slope is stable.  The fiducial cuts are selected
automatically as the max-N cell within the 16–84 percentile band of converged
slopes, and written to `output/$RUN/mag_split_fiducial.json`.

```bash
python ellipse_sweep.py \
    --source DESI \
    --fits_file $FITS \
    --run $RUN
```

Inspect the heatmap and slope histogram:

```bash
open output/$RUN/mag_split_grid.png
open output/$RUN/fiducial_slope_hist.png
```

Replot from saved results without rerunning Stan:

```bash
python ellipse_sweep.py --source DESI --fits_file $FITS --run $RUN --mag_split_plot
```

---

## Step 3: Selection criteria

Cut parameters are read automatically from `output/$RUN/mag_split_fiducial.json`
when `--run` is passed to `desi_data.py`.  No manual step required; proceed to
Step 4.  Individual parameters can still be overridden on the command line if
needed.

---

## Step 4: Prepare data

Convert the FITS file to Stan JSON format using the selection parameters
chosen in Step 3.

```bash
python desi_data.py --input $FITS --run $RUN --z_obs_min 0.01
```

Selection parameters (`haty_min`, `haty_max`, `slope_plane`, `intercept_plane`,
`intercept_plane2`) are loaded automatically from
`output/$RUN/mag_split_fiducial.json` written in Step 3.

Inspect the scatter plot to verify the selection looks correct:

```bash
open output/$RUN/data.png
```

Iterate: adjust parameters and rerun until satisfied, then proceed.

---

## Step 5: Compile Stan models (once)

Run from inside the `../../cmdstan/` directory:

```bash
cd ../../cmdstan
make ../TFPV/ariel/tophat
make ../TFPV/ariel/normal
cd ../TFPV/ariel
```

---

## Step 6: Run MCMC sampling

```bash
./tophat sample num_warmup=500 num_samples=500 num_chains=4 \
    adapt save_metric=1 \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/tophat.csv

./normal sample num_warmup=500 num_samples=500 num_chains=4 \
    adapt save_metric=1 \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/normal.csv
```

---

## Step 7: Diagnose and visualize

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/$RUN/tophat_?.csv
../../cmdstan/bin/diagnose    output/$RUN/tophat_?.csv

../../cmdstan/bin/stansummary output/$RUN/normal_?.csv
../../cmdstan/bin/diagnose    output/$RUN/normal_?.csv

# Corner plots — writes output/$RUN/tophat.png and output/$RUN/normal.png
python corner.py --run $RUN --model tophat
python corner.py --run $RUN --model normal
```

Inspect:

```bash
open output/$RUN/tophat.png
open output/$RUN/normal.png
```

---

## Step 8: Predict absolute magnitudes

```bash
python predict.py --run $RUN --model tophat --source DESI \
    --catalog --input $FITS
python predict.py --run $RUN --model normal  --source DESI \
    --catalog --input $FITS
```

`--catalog` writes an augmented FITS catalog to
`output/$RUN/{tophat,normal}_catalog.fits` containing the input columns plus
predicted mean magnitude and uncertainty.

Per run, the following output files are written:

| File | Description |
|------|-------------|
| `output/$RUN/{model}_grid.png` | mean_pred − ŷ_obs on (x̂, ŷ) grid |
| `output/$RUN/redshift_grid_{model}.png` | residual heat-map on (x̂, redshift) grid |
| `output/$RUN/redshift_{model}.png` | pull vs. redshift scatter |
| `output/$RUN/{model}_catalog.fits` | augmented FITS catalog with predicted magnitudes |
| `output/$RUN/{model}_cov.fits` | posterior predictive covariance matrix, float32, (G, G) |
| `output/$RUN/{model}_cov.png` | covariance + correlation matrix, two panels |
| `output/$RUN/{model}_cov_sub.png` | same for a random subset of ≤512 galaxies |

See [Predict.md](Predict.md) for full argument reference and covariance
computation details.
