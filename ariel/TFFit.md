# TFR Fitting

## Introduction

After selection cuts are finalised in `output/<run>/selection_criteria.json`
(see `Selection.md`), this stage prepares the Stan JSON input, compiles the Stan
tophat and normal models, and runs MCMC sampling to infer TFR parameters.

---

## Step 1: Prepare Data with Optimal Cuts

Algorithm: `fullmocks_data.py`

### 1. Reading selection cuts from `cut_sweep_best_config.json`

When `output/<run>/cut_sweep_best_config.json` is present, `fullmocks_data.py`
loads the cut values automatically — no flags needed for the cut parameters.
Cut values may also be passed explicitly on the command line (CLI flags take
precedence over the JSON).

### 2. FITS columns used

| Column | Description | Stan field |
|--------|-------------|-----------|
| `LOGVROT` | log10(V_rot / km/s) | `x = LOGVROT − 2` |
| `LOGVROT_ERR` | uncertainty on LOGVROT | `sigma_x` |
| `R_ABSMAG_SB26` | R-band absolute magnitude (SB26) | `y` |
| `R_ABSMAG_SB26_ERR` | uncertainty on R_ABSMAG_SB26 | `sigma_y` |
| `ZOBS` | observed redshift | `z_obs` |
| `MAIN` | boolean — only `MAIN=True` rows are used | — |

### 3. Selection cuts applied

Galaxies are retained if they satisfy all of:

- `haty_min ≤ y ≤ haty_max` — horizontal magnitude window
- `z_obs_min < z_obs ≤ z_obs_max` — redshift window
- `y ≥ slope_plane · x + intercept_plane` — lower oblique plane cut
- `y ≤ slope_plane · x + intercept_plane2` — upper oblique plane cut

### 4. Stan JSON construction

From the selected sample, the script computes:

- `x`, `sigma_x` — log rotation velocity and uncertainty
- `y`, `sigma_y` — absolute magnitude and uncertainty
- `z_obs` — observed redshift
- `haty_max`, `haty_min`, `slope_plane`, `intercept_plane`, `intercept_plane2` — cut parameters passed through to Stan for the selection-correction integrals
- `mu_y_TF`, `tau` — prior mean and std on true magnitudes (derived from the selected sample)
- `N_bins` — number of redshift bins for `intercept[N_bins]`

OLS on the selected sample initialises `slope` and `intercept` for Stan's
`init.json`.

### 5. Output

| File | Description |
|------|-------------|
| `output/<run>/input.json` | Stan data |
| `output/<run>/init.json` | OLS-initialised Stan parameters |
| `output/<run>/data.png` | scatter plot with selection cuts overlaid |
| `output/<run>/config.json` | exact parameters used (reproducibility) |

### Usage

```bash
python fullmocks_data.py --file $FITS --run $RUN
```

Cut parameters are loaded automatically from
`output/c000_ph000_r001/selection_criteria.json` (written by Step 3 of
`Selection.md`). Pass cut flags explicitly to override individual values.

| Argument | Default | Description |
|----------|---------|-------------|
| `--file FILE` | — | Path to a single FITS file |
| `--dir DIR` | `data/` | Directory searched for FITS files (used when `--file` is omitted) |
| `--run RUN` | required | Output subdirectory under `output/` |
| `--haty_max` | −20.0 | Dim-end (upper) magnitude cut |
| `--haty_min` | −22.2 | Bright-end (lower) magnitude cut |
| `--slope_plane` | −7.5 | Oblique cut slope |
| `--intercept_plane` | −21.0 | Lower oblique cut intercept |
| `--intercept_plane2` | −19.2 | Upper oblique cut intercept |
| `--z_obs_min` | 0.03 | Minimum redshift |
| `--z_obs_max` | 0.10 | Maximum redshift |
| `--n_objects` | 5000 | Subsample size (0 = use all) |
| `--random_seed` | — | Random seed for subsampling |

---

## Step 2: Compile Stan Models

Algorithm: `make` from `../../cmdstan/`

### 1. Standard compilation

```bash
# From ../../cmdstan/
make ../TFPV/ariel/tophat
make ../TFPV/ariel/normal
```

Binaries are written in-place as `tophat` and `normal` in the `ariel/` directory.
This step is only needed once (or after modifying a `.stan` file).

### 2. GPU compilation on NERSC

```bash
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64
make STAN_OPENCL=TRUE ../TFPV/ariel/tophat
cp ../TFPV/ariel/tophat ../TFPV/ariel/tophat_g
```

The GPU binary is copied to `tophat_g` so the CPU binary is preserved for
single-chain or diagnostic use.

---

## Step 3: Run MCMC Sampling

Algorithm: CmdStan `tophat` / `normal` executables

### 1. First run — adapt and save metric

On the first run (no prior metric available), let Stan adapt the mass matrix and
step size, and save the resulting metric for reuse:

```bash
./tophat sample num_warmup=500 num_samples=500 num_chains=4 \
  adapt save_metric=1 \
  data file=output/$RUN/input.json \
  init=output/$RUN/init.json \
  output file=output/$RUN/tophat.csv
```

This writes `tophat_1.csv` … `tophat_4.csv` and `tophat_metric.json` (containing
the adapted mass matrix) in `output/c000_ph000_r001/`. The adapted step size is
recorded in each CSV chain header.

### 2. Subsequent runs — load saved metric and step size

Once a metric and step size have been saved, subsequent runs can skip most of the
warmup:

```bash
./tophat sample algorithm=hmc metric=diag_e \
  metric_file=output/$RUN/tophat_metric.json \
  stepsize=0.11871086 \
  num_warmup=50 num_samples=500 num_chains=4 \
  data file=output/$RUN/input.json \
  init=output/$RUN/init.json \
  output file=output/$RUN/tophat.csv
```

Notes:
- Retrieve the adapted step size from the CSV chain header:
  `grep stepsize= output/$RUN/tophat_1.csv`
- `metric_file` must be a JSON file containing only the key `"inv_metric"`.
- `num_warmup=50` is sufficient when the metric and step size are pre-set.
- Replace `tophat` with `normal` to run the Gaussian prior model.

---

## Step 4: Posterior Corner Plot

Algorithm: `corner.py`

Reads the Stan CSV chain files from `output/<run>/` and produces a corner plot
of the TFR posterior parameters using ChainConsumer. Parameters present in the
CSV are plotted; missing parameters (e.g. `mu_y_TF` and `tau` are absent in the
tophat model) are silently skipped.

### 1. Parameters plotted

| Label | Stan CSV column(s) | Model |
|-------|--------------------|-------|
| `slope` | `slope` | both |
| `intercept` | `intercept.1` / `intercept[1]` / `intercept` | both |
| `$\sigma_{\rm int,x}$` | `sigma_int_x` | both |
| `$\sigma_{\rm int,y}$` | `sigma_int_y` | both |
| `mu_{y_TF}` | `mu_y_TF` | normal only |
| `tau` | `tau` | normal only |

### 2. Output

| File | Description |
|------|-------------|
| `output/<run>/<model>.png` | corner plot of posterior parameters |

Mean, std, and median for each plotted parameter are also printed to stdout,
along with ChainConsumer's 1D summary (16th/50th/84th percentiles).

### Usage

```bash
# Standard usage with --run (reads output/$RUN/<model>_?.csv)
python corner.py --run $RUN --model tophat
python corner.py --run $RUN --model normal

# Overlay two models on the same plot
python corner.py \
    "output/$RUN/tophat_?.csv" \
    "output/$RUN/normal_?.csv" \
    --output output/$RUN/compare.png

# Compare two runs
python corner.py \
    "output/$RUN/tophat_?.csv" \
    "output/c000_ph000_r002/tophat_?.csv" \
    --name $RUN --name c000_ph000_r002 \
    --output output/compare_runs.png

# Overlay true parameter values (e.g. from simulation)
python corner.py --run $RUN --model tophat \
    --truth slope=-8.3 intercept=-20.1 '$\sigma_{\rm int,x}$=0.03' '$\sigma_{\rm int,y}$=0.5'
```

| Argument | Default | Description |
|----------|---------|-------------|
| `PATTERN …` | — | Glob patterns for Stan CSV files (quote each to prevent shell expansion) |
| `--run NAME` | — | Run name; reads `output/<NAME>/<model>_?.csv`, writes `output/<NAME>/<model>.png` |
| `--model` | `tophat` | Model to plot when using `--run`: `tophat` or `normal` |
| `--output FILE` | `corner_plot.png` | Output PNG path (overrides the `--run` default) |
| `--name LABEL` | — | Legend label for a chain (repeat once per pattern, in order) |
| `--truth PARAM=VALUE …` | — | True parameter values to overlay as vertical/horizontal lines |

---

## Summary of Generated Files

| File | Script | Description |
|------|--------|-------------|
| `output/<run>/input.json` | `fullmocks_data.py` / `desi_data.py` | Stan data: galaxy observables (x, y, σ_x, σ_y, z_obs), selection cut parameters, prior hyperparameters, redshift bin count |
| `output/<run>/init.json` | `fullmocks_data.py` / `desi_data.py` | OLS-derived initial values for Stan parameters (slope, intercept, scatter) |
| `output/<run>/data.png` | `fullmocks_data.py` / `desi_data.py` | (x, y) scatter plot of the full sample and the selected subsample with selection cuts overlaid |
| `output/<run>/config.json` | `fullmocks_data.py` / `desi_data.py` | Exact selection parameter values and sample size used, for reproducibility |
| `output/<run>/tophat_1.csv … tophat_4.csv` | CmdStan `tophat` | MCMC chain files for the tophat model; each contains parameter draws, log-probability, and generated quantities |
| `output/<run>/normal_1.csv … normal_4.csv` | CmdStan `normal` | MCMC chain files for the normal model |
| `output/<run>/tophat_metric_1.json … tophat_metric_4.json` | CmdStan `tophat` | Per-chain adapted mass matrices saved by `adapt save_metric=1`; used to warm-start subsequent runs |
| `output/<run>/tophat.png` | `corner.py` | Corner plot of the tophat model posterior (slope, intercept, σ_int_x, σ_int_y) |
| `output/<run>/normal.png` | `corner.py` | Corner plot of the normal model posterior (adds mu_y_TF, τ) |
