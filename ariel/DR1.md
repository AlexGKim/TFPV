# DR1 Run: SGA-2020

This document records the full command sequence for the DR1 run on the
`SGA-2020_iron_Vrot_VI_corr_v3.fits` dataset.

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr_v3.fits   # input FITS catalog
export RUN=DR1_v3                                         # output directory name: output/$RUN/
export CONFIG=configs/dr1_v3.json                      # pipeline config (parameters + file paths)
```

`FITS` is only needed for the interactive Phase A steps (1–3b) that accept explicit file flags.
`CONFIG` is the primary input for all pipeline scripts and `run_pipeline.py`.
`RUN` is still used directly by `set_fiducial.py`, `corner.py`, and diagnostic commands.

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
profile over all catalog objects.  Use the plot to guide the choice of the
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

> `set_fiducial.py` has no `--config` option: it is the script that
> *creates* the fiducial parameters that later go into the config.

Inspect the pull profile with the cuts:

```bash
open output/$RUN/select_v2_fiducial_pull.png
```

---

## Step 3b: Export run config

After completing the interactive fiducial step, capture all parameter choices
in a portable config file:

```bash
python export_config.py --run $RUN --out configs/dr1_v3.json
```

The script reads `output/$RUN/select_v2_fiducial.json` (including the
interactively chosen cuts) and prompts for the remaining pipeline settings
(`exe`, `source`, `model`, `n_sigma`).  The `fits_file` is taken automatically
from `output/$RUN/config.json` (written by `desi_data.py`) so it matches the
file actually used.  Commit the resulting JSON to git — it is the permanent
version record for this run.

> `export_config.py` has no `--config` option: it is the script that
> *produces* the config file.

---

## Step 4: Prepare data

Convert the FITS file to Stan JSON format using the selection parameters
chosen in Step 3.

```bash
# via config
python desi_data.py --config $CONFIG

# via flags
python desi_data.py --input $FITS --run $RUN \
    --haty_min -21.5 --haty_max -19.0 \
    --slope_plane -6.386925076468424 --intercept_plane -20.74814050932727 --intercept_plane2 -18.31309635087515 \
    --z_obs_min 0.03 --z_obs_max 0.08
```

Inspect the scatter plot to verify the selection looks correct:

```bash
open output/$RUN/data.png
```

Iterate: adjust parameters and rerun until satisfied, then proceed.

`desi_data.py` writes `output/$RUN/config.json` recording the selection cuts
and `fits_file` used.  This file is read by `export_config.py` (Step 3b) and
as a fallback by `predict.py` (Step 8).

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

Stan is invoked directly; there is no config or flags abstraction here.

```bash
./tophat sample num_warmup=500 num_samples=1000 num_chains=4 \
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
# Convergence diagnostics (flags only — no config equivalent)
../../cmdstan/bin/stansummary output/$RUN/tophat_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/tophat_?.csv > output/$RUN/diagnose.txt

../../cmdstan/bin/stansummary output/$RUN/normal_?.csv
../../cmdstan/bin/diagnose    output/$RUN/normal_?.csv

# Corner plots
# via config
python corner.py --config $CONFIG
python corner.py --config $CONFIG --model normal

# via flags
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
# via config
python predict.py --config $CONFIG --catalog
python predict.py --config $CONFIG --model normal --catalog

# via flags
python predict.py --run $RUN --model tophat --source DESI --input $FITS --catalog
python predict.py --run $RUN --model normal  --source DESI --input $FITS --catalog
```

`--catalog` writes an augmented FITS catalog to
`output/$RUN/{tophat,normal}_catalog.fits` containing the input columns plus
predicted mean magnitude and uncertainty.

Per run, the following output files are written:

| File | Description |
|------|-------------|
| `output/$RUN/{model}_grid.png` | mean residual on (x̂, ŷ) grid — main selection |
| `output/$RUN/{model}_grid_full.png` | mean residual on (x̂, ŷ) grid — full input sample |
| `output/$RUN/{model}_grid_variance.png` | average prediction variance per (x̂, ŷ) bin |
| `output/$RUN/{model}_grid_sigma_x.png` | mean σ_x̂ per (x̂, ŷ) bin |
| `output/$RUN/redshift_grid_{model}.png` | mean redshift on (x̂, ŷ) grid |
| `output/$RUN/redshift_{model}.png` | residual vs. redshift scatter |
| `output/$RUN/variance_redshift_{model}.png` | prediction variance vs. redshift |
| `output/$RUN/variance_xhat_{model}.png` | prediction variance vs. x̂ |
| `output/$RUN/{model}_catalog.fits` | augmented FITS catalog with predicted magnitudes |
| `output/$RUN/{model}_cov.fits` | posterior predictive covariance matrix, float32, (G, G) |
| `output/$RUN/{model}_cov.png` | covariance + correlation matrix, two panels |
| `output/$RUN/{model}_cov_sub.png` | same for a random subset of ≤512 galaxies |

See [Predict.md](Predict.md) for full argument reference and covariance
computation details.

---

## Step 9: Explore residual bias

Correlate magnitude residuals with additional galaxy properties from the FITS
catalogue (morphology, colour, size, inclination, environment).  Results land
in `output/$RUN/explore_residuals/`.

```bash
# via config
python explore_residuals.py --config $CONFIG --kind tophat

# via flags
python explore_residuals.py --run $RUN --kind tophat
```

Output plots:

| File | Description |
|------|-------------|
| `resid_vs_ba.png` | Residual vs. axis ratio b/a (inclination proxy) |
| `resid_vs_d26_kpc.png` | Residual vs. physical diameter D₂₆ (kpc) |
| `resid_vs_g_r.png` | Residual vs. g − r colour |
| `resid_vs_r_z.png` | Residual vs. r − z colour |
| `resid_vs_g_z.png` | Residual vs. g − z colour |
| `resid_vs_sma_sb26.png` | Residual vs. angular semi-major axis at SB26 |
| `resid_vs_sma_ratio.png` | Residual vs. SMA_SB26 / SMA_SB22 (concentration proxy) |
| `resid_vs_{g,r,z}_sma50.png` | Residual vs. per-band half-light radius |
| `resid_by_morphtype.png` | Mean residual per morphological type |
| `resid_by_photsys.png` | Residual distributions for PHOTSYS N vs. S |
| `resid_by_group.png` | Isolated (GROUP_MULT=1) vs. group galaxies |
| `correlation_summary.png` | Pearson r summary for all continuous parameters |

---

## Running a parameter variant

To run the pipeline with different parameters (e.g. a wider redshift range),
copy an existing config, edit the desired values, and run non-interactively:

```bash
cp configs/dr1_default.json configs/dr1_zmax015.json
```

Edit `configs/dr1_zmax015.json` — change `"run"` (the output directory name)
and any parameters you want to vary, e.g.:

```json
{
  "run": "dr1_zmax015",
  "z_obs_max": 0.15,
  ...
}
```

Then run the full pipeline:

```bash
python run_pipeline.py configs/dr1_zmax015.json
```

Or run only a subset of steps (e.g. re-do data prep and onwards after
changing selection cuts):

```bash
python run_pipeline.py configs/dr1_zmax015.json --steps 1-4
```

Step 5 (Stan sampling) is not executed automatically — `run_pipeline.py`
prints the sampling command for you to run manually.  All outputs land in
`output/<run>/` as usual.

**Workflow summary:**

| Phase | Steps | How |
|-------|-------|-----|
| A — Discovery | 1–3 + 3b | Manual (interactive `set_fiducial.py` + `export_config.py`) |
| B — Variants  | 1–7 | `run_pipeline.py configs/<variant>.json` |
