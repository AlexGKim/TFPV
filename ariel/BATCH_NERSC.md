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
| `output/DR1_v6_2color/metric.json` | yes | 17×17 inverse mass matrix (parameter covariance) for HMC | Output of step 5e — **not actually read by `step6_node.sh`/`step6_chain.sh`** (neither passes `metric_file=`; both run from the identity metric with in-warmup adaptation). Step 5e is optional/vestigial, kept only for `step6_chain.sh --debug`-style manual experiments. |
| `2color_g` (binary) | **no** | Compiled Stan GPU executable | Must be built on Perlmutter; see One-Time Setup below |

For the initial test run with `DR1_v6_2color`, the two committed output files mean **steps 4 and 5d can be skipped** — go straight to step 6. Step 5e was never a real prerequisite (see above).

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
python select_v2.py --config configs/<base>.json --exe ./tophat  # Step 2: MLE + pull profile (tophat, not 2color)
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

### Build the metric — optional, not part of the standard workflow

**`step6_node.sh` and `step6_chain.sh` do not read `metric.json`** — both run
every chain from the identity metric with in-warmup adaptation
(`metric=dense_e`, no `metric_file=`). Step 5e (below) and this section are
kept only for manual experimentation; skip them for a normal run. An
abacus-mock A/B test found the identity-start approach ~2.7x faster overall
than a Step-5e-built metric with no quality loss (the metric's own
short-chain/`np.cov` method is fragile — see the warning this section used
to lead with, now folded into `step5e_metric.sh`'s own comments).

If you do want to experiment with a pre-built metric manually (not needed
for `step6_node.sh`):

```bash
sbatch --export=CONFIG=$CONFIG slurm/step5e_metric.sh   # ~7h
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

### Step 5e — Build metric (optional, not needed for step6_node.sh)

Skip this — `step6_node.sh` doesn't read `metric.json` (see "Build the
metric" above). Kept only for manual experimentation:

```bash
sbatch --export=CONFIG=$CONFIG slurm/step5e_metric.sh
```

### Step 6 — MCMC sampling (4 independent chains, 1 node)

Requires only step 5d to be complete (not step 5e — see above). This script
submits a single `step6_node.sh` job (1 node, 4 GPUs, one chain per GPU via
`CUDA_VISIBLE_DEVICES`) and automatically chains step 7 and step 8 as
dependencies:

```bash
bash slurm/step6_submit.sh $CONFIG
```

`sacct` shows a `--gpus-per-task=1` job is allocated the *whole* 4-GPU node
regardless (`gres/gpu:a100=4, cpu=128, node=1`), and each chain runs
single-threaded (`num_threads=1`), so 4 separate 1-GPU jobs (the old
`step6_chain.sh`×4 pattern) reserved 4 whole nodes and used only 1 GPU on
each — 3 idle GPUs and ~127 idle CPU cores per job. `step6_node.sh` runs all
4 chains as backgrounded processes within the one node SLURM already grants
it, so it's both faster (one queue wait instead of four, and avoids QOS
submit-count limits when running many runs in a batch — see `BATCH_MOCKS.md`)
and uses hardware that would otherwise sit idle. Each chain runs
independently, in parallel on its own GPU — timing depends on
`NUM_WARMUP`/`MAX_DEPTH` (now default 1000/10, up from 250/8; re-measure
actual wall-clock for your dataset rather than trusting older ~14h figures
quoted for the previous 250/8 defaults).
Step 7 (diagnostics) starts automatically after all 4 chains complete.
Step 8 (predictions) starts automatically after step 7.

If a single chain fails while the other 3 succeed, `step6_chain.sh` (1
chain, 1 job) is still available to resubmit just that chain — see
"Resubmitting Failed Steps" below.

### Monitoring

```bash
squeue -u $USER                          # live SLURM queue
bash slurm/check_status.sh $CONFIG      # sentinel-file status
cat slurm/logs/step6_node_*.out          # tail the step6 log (all 4 chains)
```

---

## Resubmitting Failed Steps

Each step is idempotent — re-running it overwrites outputs cleanly.

### All 4 chains

```bash
sbatch --export=CONFIG=$CONFIG slurm/step6_node.sh
```

### Single chain failure

If only one of the 4 chains failed, resubmit just that one instead of the
whole node:

```bash
sbatch --export=CONFIG=$CONFIG,CHAIN_ID=2 slurm/step6_chain.sh
```

### Step 4 / 5d failure

```bash
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
sbatch --export=CONFIG=$CONFIG slurm/step5d_map.sh
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
| `output/$RUN/2color_1.csv` … `2color_4.csv` | Posterior MCMC draws (4 chains) |
| `output/$RUN/stansummary.txt` | Convergence summary (R̂, ESS) |
| `output/$RUN/diagnose.txt` | Divergence and treedepth diagnostics |
| `output/$RUN/2color.png` | Corner plot |
| `output/$RUN/explore_residuals/` | Residual-vs-galaxy-property diagnostic plots |
| `output/$RUN/color_xonly_catalog.fits` | Catalog (x-only model, the default — pass `--full` to step8 for `color_catalog.fits`/`color_cov.fits` too) |
| `output/$RUN/color_xonly_cov.h5` | (G,G) posterior predictive covariance (x-only) |

`output/$RUN/metric.json` is **not** produced by the standard workflow
(Step 5e is optional/skipped — see above).

---

## Expected Runtimes

| Step | Queue | Time |
|------|-------|------|
| compile | debug | ~5 min |
| step4 (data prep) | debug | <5 min |
| step5d (MAP) | debug | ~5 min |
| step6 (4 chains, 1 node) | regular | depends on dataset size and the `NUM_WARMUP=1000`/`MAX_DEPTH=10` defaults — re-measure per dataset rather than assuming a fixed figure (walltime is set to 24h; adjust if insufficient) |
| step7 (diagnose) | debug | ~15 min |
| step8 (predict) | regular | ~1–4 hours |

Step 5e is optional (not part of the standard workflow) — skip it.
