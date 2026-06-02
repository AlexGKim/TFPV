# Running the 2COLOR Pipeline on NERSC Perlmutter

This runbook covers Phase B of the 2COLOR workflow — the batch steps that run on
Perlmutter. Phase A (selection ellipse, fiducial setting, config export) is done
locally and results in a committed config file such as `configs/dr1_v6_2color.json`.

---

## Prerequisites

### 1. Pull the batch branch

```bash
cd $SCRATCH/TFPV/ariel        # or wherever you cloned the repo on NERSC
git fetch origin
git checkout batch
git pull
```

### 2. Load modules

Add the following to your `~/.bashrc` or run before any job submission:

```bash
module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64
```

The `LIBRARY_PATH` line is a required workaround for the OpenCL CUDA link error
on Perlmutter (otherwise `make` fails with a missing `libOpenCL` error).

### 3. Set your config

```bash
export CONFIG=configs/dr1_v6_2color.json
```

### 4. Data file prerequisites

| File | In repo? | Contents | Notes |
|------|----------|----------|-------|
| `configs/dr1_v6_2color.json` | yes | Selection cuts (magnitude window, redshift range, color limit, ellipse geometry) produced by Phase A | Already committed |
| `data/SGA-2020_iron_Vrot_VI_corr_v6.fits` | **no** | Raw DESI galaxy catalog (~9 MB) | Must be present on NERSC scratch; only needed if re-running step 4 |
| `output/DR1_v6_2color/input.json` | yes | Stan data arrays: x, σ_x, y, σ_y, z_obs, g, σ_g for ~4728 selected galaxies, plus bounds and mean_x/sd_x | Output of step 4; committed to allow skipping step 4 |
| `output/DR1_v6_2color/init_MAP.json` | yes | Stan parameter starting values at the MAP optimum: slope, intercept, scatter terms, color slopes, k-corrections | Output of step 5d; committed to allow skipping step 5d |
| `output/DR1_v6_2color/metric.json` | yes | 17×17 inverse mass matrix (parameter covariance) for HMC | Output of step 5e; committed to allow skipping step 5e (~7h) |
| `2color_g` (binary) | **no** | Compiled Stan GPU executable | Must be built on Perlmutter; see One-Time Setup below |

For the initial test run with `DR1_v6_2color`, the three committed output files mean **steps 4, 5d, and 5e can all be skipped** — go straight to step 6.

### Obtaining the FITS file

The FITS file is not in the repo. Transfer it to NERSC scratch before running step 4:

```bash
# From your local Mac:
scp data/SGA-2020_iron_Vrot_VI_corr_v6.fits \
    perlmutter.nersc.gov:$SCRATCH/TFPV/ariel/data/
```

Or if it is already on NERSC from a previous run, confirm it is in place:

```bash
ls -lh $SCRATCH/TFPV/ariel/data/SGA-2020_iron_Vrot_VI_corr_v6.fits
```

### Obtaining the config file (Phase A)

The config file encodes the selection cuts chosen interactively on your local Mac.
For `DR1_v6_2color` this is already done and committed — no action needed.

For a **new dataset**, Phase A must be run locally first:

```bash
# On local Mac, from ariel/:
export FITS=data/<new_catalog>.fits
export RUN=<new_run_name>

python selection_ellipse.py --config configs/<base>.json  # Step 1: fit selection ellipse
python select_v2.py --config configs/<base>.json --exe ./tophat  # Step 2: MLE + pull profile
python set_fiducial.py --run $RUN                         # Step 3: interactive cut selection
python export_config.py --run $RUN --out configs/<new>.json  # Step 3b: export config
git add configs/<new>.json && git commit -m "add config for <new_run>"
git push
```

Then on NERSC: `git pull` to get the new config before submitting jobs.

---

## One-Time Setup

### Compile the GPU binary

Run once per checkout (or after any `.stan` file changes):

```bash
sbatch slurm/compile_2color_gpu.sh
```

Wait for it to complete, then verify:

```bash
ls -lh 2color_g
```

### Build the metric (one-time, reusable)

The metric captures the parameter covariance and dramatically improves HMC
efficiency. It only needs to be built once — the result can be copied to any
new run directory.

```bash
# First complete step 4 and 5d (see below), then:
sbatch --export=CONFIG=$CONFIG slurm/step5e_metric.sh
```

This takes ~7 hours. Once done, `output/DR1_v6_2color/metric.json` exists and
can be reused for future runs:

```bash
cp output/DR1_v6_2color/metric.json output/<new_run>/metric.json
```

---

## Per-Run Workflow

All jobs are submitted from the `ariel/` directory. Jobs write logs to `slurm/logs/`.

### Step 4 — Prepare data

```bash
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
```

Produces: `output/$RUN/input.json`, `output/$RUN/init.json`

### Step 5d — MAP optimization

Requires step 4 to be complete.

```bash
sbatch --export=CONFIG=$CONFIG slurm/step5d_map.sh
```

Produces: `output/$RUN/optimize.csv`, `output/$RUN/init_MAP.json`

### Step 5e — Build metric (if not reusing)

Requires step 5d to be complete. Skip if you are copying an existing `metric.json`.

```bash
sbatch --export=CONFIG=$CONFIG slurm/step5e_metric.sh
```

Produces: `output/$RUN/metric.json`  (~7 hours)

### Step 6 — MCMC sampling (4 independent chains)

Requires steps 5d and 5e to be complete. This script submits all 4 chains and
automatically chains step 7 and step 8 as dependencies:

```bash
bash slurm/step6_submit.sh $CONFIG
```

Each chain runs independently (~14 hours each, all 4 run in parallel).
Step 7 (diagnostics) starts automatically after all 4 chains complete.
Step 8 (predictions) starts automatically after step 7.

### Monitoring

```bash
squeue -u $USER                          # live SLURM queue
bash slurm/check_status.sh $CONFIG      # sentinel-file status
cat slurm/logs/step6_chain1_*.out        # tail a chain log
```

---

## Resubmitting Failed Steps

Each step is idempotent — re-running it overwrites outputs cleanly.

### Single chain failure

```bash
sbatch --export=CONFIG=$CONFIG,CHAIN_ID=2 slurm/step6_chain.sh
```

### Step 4 / 5d / 5e failure

```bash
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
sbatch --export=CONFIG=$CONFIG slurm/step5d_map.sh
sbatch --export=CONFIG=$CONFIG slurm/step5e_metric.sh
```

### Step 7 / 8 failure (after all chains done)

```bash
sbatch --export=CONFIG=$CONFIG slurm/step7_diagnose.sh
sbatch --export=CONFIG=$CONFIG slurm/step8_predict.sh
```

### Check which steps are missing

```bash
bash slurm/check_status.sh $CONFIG
```

---

## Expected Outputs

After the full pipeline completes:

| File | Description |
|------|-------------|
| `output/$RUN/input.json` | Stan data (N galaxies, z- and g-band) |
| `output/$RUN/init_MAP.json` | MAP warm start for MCMC |
| `output/$RUN/metric.json` | Pre-computed inverse mass matrix |
| `output/$RUN/2color_1.csv` … `2color_4.csv` | Posterior MCMC draws (4 chains) |
| `output/$RUN/stansummary.txt` | Convergence summary (R̂, ESS) |
| `output/$RUN/diagnose.txt` | Divergence and treedepth diagnostics |
| `output/$RUN/2color.png` | Corner plot |
| `output/$RUN/color_catalog.fits` | Catalog with MU_TF, LOGDIST (full model) |
| `output/$RUN/color_xonly_catalog.fits` | Catalog (x-only model) |
| `output/$RUN/color_cov.fits` | (G,G) posterior predictive covariance |

---

## Expected Runtimes

| Step | Queue | Time |
|------|-------|------|
| compile | debug | ~5 min |
| step4 (data prep) | debug | <5 min |
| step5d (MAP) | debug | ~5 min |
| step5e (metric) | regular | ~7 hours |
| step6 (4 chains) | regular | ~14 hours each (parallel) |
| step7 (diagnose) | debug | ~15 min |
| step8 (predict) | regular | ~1–4 hours |
