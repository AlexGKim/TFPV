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
