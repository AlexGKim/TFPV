# Running the 2COLOR Pipeline on AbacusSummit Mocks (Batch)

This document covers running the 2COLOR Tully-Fisher pipeline over a **set of
AbacusSummit mock FITS files**. It is both a runbook (for a person submitting the
jobs) and an initial-condition brief (for a future Claude session resuming this
work). For the mechanics of the individual pipeline steps, see `BATCH_NERSC.md`
(runbook) and `BATCH_CLAUDE.md` (step map, failure modes); this file only covers
what is *different* for the mock batch and does not duplicate them.

---

## Mock Batch Overview (design decisions + rationale)

The mock batch is deliberately a stripped-down version of the full per-dataset
workflow. The following decisions are fixed; do not re-derive them:

1. **Mocks are treated like real data.** The mock FITS files carry `MAIN`/`DWARF`
   columns, but we **do not use them** — selection is purely from observables.
   `desi_data.py` already ignores those columns, so no code change is needed.
   (Confirmed on the test file: `MAIN` is a galaxy-type flag and the exact
   complement of `DWARF`; both populations span the full redshift range, so `MAIN`
   is *not* a redshift selector. Redshift is cut separately by `z_obs_min/max`.
   Only 3 rows have `LOGVROT==0` — negligible.)

2. **Selection cuts are frozen and shared by every mock.** They come from
   `configs/abacus_2color.json` (`haty_min/max`, `slope_plane`,
   `intercept_plane`, `intercept_plane2`, `z_obs_min/max`, `random_seed=42`,
   `n_sigma`, `n_sigma_perp`). The hands-on Phase A (selection ellipse /
   fiducial selection) was already done to produce these values and is **not
   part of this pipeline**.

3. **Step 5e (metric build, ~7h) must be run once per mock dataset, then
   reused across every file *and every subset* in that dataset.** The DR1
   metric (`output/DR1_v6_2color/metric.json`) is **not** transferable to
   mocks — confirmed empirically: using it caused repeated `cholesky_decompose`
   failures in the sampler, the chains ran at ~50s/iteration (vs. ~1–2s with
   a good metric), and 18h was not enough for 1000 samples. Build the metric
   once on any representative mock subset, then copy it to all other run dirs.
   The reusable mock metric lives at `output/abacus_2color/metric.json` once
   built.

4. **Every mock file is split into `n_subsets` disjoint subsets** (see "Subset
   Partition Mode" below) — this isn't just for the one oversized 169k-galaxy
   test file, it's the standard mode `make_batch_configs.py` uses for every
   file in the batch. **Run name** is the `c<NN>_ph<NN>_r<NN>` file token
   (regex `c\d+_ph\d+_r\d+`) plus a `_s<NN>` subset suffix, e.g.
   `TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits` → runs
   `c000_ph000_r001_s00` … `c000_ph000_r001_s04` (for `n_subsets=5`). Outputs
   go to `output/<run>/`, one independent run dir per (file, subset) pair.

5. **Per-(file, subset) chain:**
   `step4 → step5d → step5e (first subset of first file only) → step6 ×4 →
   step7 → step8`. Every other (file, subset) pair copies the metric from the
   first and skips step5e. **Cost implication:** total step6 chains submitted
   = `n_files × n_subsets × 4`, not `n_files × 4` — factor in `n_subsets`
   (default 5) when estimating GPU-hours (see "Runtime / cost notes" below).

**Target mock set:** `v0.5.7`
(`/global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7/`).
The batch driver takes a `--dir`, so it processes whatever files are present when
more arrive. (Other populated sets exist: `DR2/.../v0.5.6/` 675 files, and older
`mocks/TF_mocks/fullmocks/v0.5.1–4/` 675 each.)

---

## File / Command Map

| Artifact | Role |
|----------|------|
| `configs/abacus_2color.json` | Base config: frozen selection cuts + test fits file. |
| `output/abacus_2color/metric.json` | Reusable HMC metric (seed; copied into every run dir). |
| `make_batch_configs.py` | Generate per-(file, subset) configs from a mock dir (`--n-subsets`, default 5); seed each `output/<run>/metric.json`. |
| `slurm/batch_submit.sh` | Submit the full dependency chain per (file, subset) run (`--debug` for plumbing test). |
| `slurm/batch_status.sh` | Aggregate sentinel completion across all runs in a config dir. |
| `slurm/step6_node.sh` | All 4 MCMC chains, 1 node/4 GPUs (`CUDA_VISIBLE_DEVICES`); honors `DEBUG=1` (15 samples, no adaptation). `step6_chain.sh` still exists for resubmitting a single failed chain. |
| `batch/job_tracker.csv` | Appended log of submitted SLURM job IDs per run. |
| `slurm/step{4,5d,6,7,8}_*.sh` | The underlying step scripts (see `BATCH_NERSC.md`). |

---

## One-Time Setup

```bash
cd $SCRATCH/TFPV/ariel        # or wherever the repo lives on NERSC
git checkout batch && git pull
module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

# GPU binary 2color_g must exist (already compiled). If not:
#   sbatch slurm/compile_2color_gpu.sh

# Seed the reusable metric (once):
cp output/DR1_v6_2color/metric.json output/abacus_2color/metric.json
```

---

## Step 0 — Validate on a single mock (recommended before any batch)

Run the standalone test on `configs/abacus_2color.json` to confirm the reused
metric is adequate before fanning out:

```bash
sbatch --export=CONFIG=configs/abacus_2color.json slurm/step4_data.sh
# verify output/abacus_2color/input.json (N up to 5000, sane ranges) and data.png
sbatch --export=CONFIG=configs/abacus_2color.json slurm/step5d_map.sh
# verify init_MAP.json (finite, no NaN); metric.json already seeded -> skip step5e
bash slurm/step6_submit.sh configs/abacus_2color.json   # 4 chains + auto step7/8
```

After it completes, check `output/abacus_2color/stansummary.txt` (R̂ < 1.01,
ESS > 100/chain) and that `stepsize__` in the chain CSVs is ~0.08 (not ~0.002).

**Metric-adequacy fallback:** if `stepsize__` is tiny / transitions hit max
treedepth, the seeded DR1 metric is inadequate for mocks. Build it once on the
mock and reuse *that*:

```bash
sbatch --export=CONFIG=configs/abacus_2color.json slurm/step5e_metric.sh   # ~7h, once
# then re-seed batch runs from output/abacus_2color/metric.json
```

---

## Debug Mode — fast end-to-end plumbing test

Full chains cost ~14h each, so before a real batch confirm the *plumbing*
(config generation, metric copy, dependencies, sentinels, output FITS) with short
chains on the debug queue. Debug chains skip adaptation and sample at a fixed
known-good stepsize (0.08) with the seeded metric, so each completes in a few
minutes (step6 gets `-t 00:20:00` on the debug GPU queue). Results are **not**
science-grade. (Do not use `num_warmup<20` with adaptation: Stan then disables
adaptation, falls back to a tiny stepsize, and a single iteration can exceed
10 min.)

```bash
python3 make_batch_configs.py \
    --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
    --base configs/abacus_2color.json \
    --outdir configs/batch_debug \
    --metric output/abacus_2color/metric.json \
    --n-subsets 5 --n-objects 5000
bash slurm/batch_submit.sh configs/batch_debug --debug
watch bash slurm/batch_status.sh configs/batch_debug   # all 8 sentinels in ~10-15 min
```

This generates `n_files × n_subsets` configs/runs (5 per file by default), so a
debug run against even a handful of mock files fans out to dozens of tiny
plumbing-test chains — that's expected and still fast on the debug queue.

Success = every `.step*_done` sentinel appears and `output/<run>/color_catalog.fits`
is written. Then drop `--debug` for the real run.

---

## Full Batch Run

```bash
# 1. Generate n_subsets configs per fits file + seed each run's metric:
python3 make_batch_configs.py \
    --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
    --base configs/abacus_2color.json \
    --outdir configs/batch_v0.5.7 \
    --metric output/abacus_2color/metric.json \
    --n-subsets 5 --n-objects 5000

# 2. Submit (throttle so at most N (file, subset) runs' chains are queued at once):
bash slurm/batch_submit.sh configs/batch_v0.5.7 8

# 3. Monitor:
squeue -u $USER
bash slurm/batch_status.sh configs/batch_v0.5.7            # summary
bash slurm/batch_status.sh configs/batch_v0.5.7 --verbose  # per-run detail
```

`batch_submit.sh` skips any run whose `.step8_done` sentinel already exists, so it
is safe to re-run to pick up failed/incomplete (file, subset) runs.

---

## Expected Outputs (per run, in `output/<run>/`)

`input.json`, `init.json`, `init_MAP.json`, `metric.json` (copied),
`2color_1.csv`…`2color_4.csv`, `stansummary.txt`, `diagnose.txt`,
`color_catalog.fits`, `color_xonly_catalog.fits`, `color_cov.fits`, and the
`.step*_done` sentinels.

---

## Resubmitting failures

```bash
# A single chain (after step5d done):
sbatch --export=CONFIG=configs/batch_v0.5.7/c000_ph000_r001_s00.json,CHAIN_ID=2 slurm/step6_chain.sh
# Re-run one (file, subset) run's remaining steps: clear its sentinel(s) and re-submit:
rm output/c000_ph000_r001_s00/.step8_done
bash slurm/batch_submit.sh configs/batch_v0.5.7 8
# Per-run status:
bash slurm/check_status.sh configs/batch_v0.5.7/c000_ph000_r001_s00.json
```

---

## Runtime / cost notes

- step4 (CPU debug, <5 min), step5d (GPU debug, ~5 min): cheap, per (file, subset).
- **step6 dominates GPU-hours:** 4 chains × ~14h each (run in parallel, 1 per GPU
  on a single `step6_node.sh` node) per (file, subset) run. With `n_files` files
  and `n_subsets` subsets each (default 5), total step6 chains = `n_files ×
  n_subsets × 4` — **5× the GPU-hours of a naive one-run-per-file estimate.**
- **step6 job-submission count is now `n_files × n_subsets`, not ×4.**
  `step6_node.sh` runs all 4 chains as backgrounded processes on the 4 GPUs of
  one already-allocated node (`sacct` confirms a `--gpus-per-task=1` job gets
  the whole 4-GPU node anyway — nothing here is fractionally shared), instead
  of `batch_submit.sh` submitting 4 separate `step6_chain.sh` jobs per run. This
  matters most on constrained QOS like debug (`MaxSubmitPU=5`): fanning out 5
  subsets × 4 old-style chain jobs could only fit ~1 run's GPU jobs in the
  queue at once and required manually threading each run's submission through
  as slots freed — with `step6_node.sh` each run needs only 1 debug-GPU slot
  (+1 for step5d), so far more runs fit in flight simultaneously. Use the
  `MAX_CONCURRENT` throttle (now counts `step6_node` jobs, i.e. runs, directly)
  and mind the NERSC regular-GPU QOS limits for the real (non-debug) batch.
- step7 (CPU debug, ~15 min), step8 (CPU, fast for 5000 objects) — per (file, subset).
- **step5e** (~7h) is run once total (on the first subset of the first mock file)
  to build a mock-specific metric, then that metric is reused for every other
  file *and* every other subset. The DR1 metric is **not** transferable to mocks
  (see decision #3 above).

---

## Subset Partition Mode (mechanism, and a single-file manual walkthrough)

When a mock FITS file is too large for the prediction step to fit in memory (e.g.
169k galaxies → OOM on the O(M×G) posterior predictive computation), split it into
disjoint subsets. Each subset trains on 5,000 galaxies (matching DR2) and predicts
on the remaining holdout within the subset. **This is not an opt-in special case
for oversized outliers** — decisions #3–#5 above establish it as the standard mode
for every file in the real batch, and `make_batch_configs.py --n-subsets 5` (used
by "Full Batch Run" above) generates it automatically for every file it finds. The
rest of this section documents the underlying mechanism and walks through it by
hand for one file — useful for understanding what's happening, debugging a single
run, or validating locally before a real multi-file batch (which is exactly how
`configs/abacus_subsets/` and `run_subsets.sh` were built and exercised).

### How it works

`desi_data.py` supports three config fields for partitioning:

```json
{
  "n_subsets": 5,       // total number of disjoint partitions
  "subset_index": 0,    // which partition (0-indexed)
  "n_objects": 5000     // training sample size within the subset
}
```

When `n_subsets` and `subset_index` are present, `desi_data.py`:

1. Apply all selection cuts (magnitude window, redshift, plane cuts) → N_after_cuts
2. Permute all N_after_cuts indices with `random_seed` (shared across subsets)
3. Split the permuted array into `n_subsets` contiguous chunks
4. Select chunk `subset_index` (~18,759 galaxies per subset)
5. Subsample `n_objects` from the chunk for training (5,000)
6. Record both `subset_sga_ids` (full chunk) and `train_sga_ids` (training sample)

This guarantees **zero overlap** between subsets and full coverage of all post-cut
galaxies. Predictions are made on the holdout within the subset (~13,759 galaxies
not used for training).

### Config generation

5 configs live at `configs/abacus_subsets/abacus_2color_s00.json` through `s04.json`.
They share all selection parameters from `configs/abacus_2color.json` but differ in:
- `"run": "abacus_2color_s00"` … `"abacus_2color_s04"`
- `"subset_index": 0` … `4`
- `"n_subsets": 5`
- `"n_objects": 5000`
- `"fits_file"`: local path `data/TF_AbacusSummit_…_appmag.fits`

To regenerate:

```python
import json, os
base = json.load(open('configs/abacus_2color.json'))
base['fits_file'] = 'data/TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11_appmag.fits'
os.makedirs('configs/abacus_subsets', exist_ok=True)
for i in range(5):
    cfg = dict(base)
    cfg['run'] = f'abacus_2color_s{i:02d}'
    cfg['subset_index'] = i
    cfg['n_subsets'] = 5
    cfg['n_objects'] = 5000
    with open(f'configs/abacus_subsets/abacus_2color_s{i:02d}.json', 'w') as f:
        json.dump(cfg, f, indent=2)
```

### Running all 5 subsets (NERSC / SLURM)

`batch_submit.sh` handles the full chain for every config in a directory. Pass
`--metric` so it seeds each run dir before submitting:

```bash
bash slurm/batch_submit.sh configs/abacus_subsets \
    --metric output/abacus_2color/metric.json
```

This submits `step4 → step5d → step6×4 → step7 → step8` for all 5 subsets with
SLURM dependencies, throttled to 8 concurrent files (20 chains) by default.

### Running one subset end-to-end (manual, NERSC / SLURM)

```bash
export CONFIG=configs/abacus_subsets/abacus_2color_s00.json

sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
# After step4 done:
cp output/abacus_2color/metric.json output/abacus_2color_s00/metric.json
sbatch --export=CONFIG=$CONFIG slurm/step5d_map.sh
# After step5d done:
bash slurm/step6_submit.sh $CONFIG
```

### Running locally without SLURM

`run_subsets.sh` runs subsets s01–s04 end-to-end (step4 → step5d → step6 → step7
→ step8) directly via the `./2color` binary, for machines without SLURM access.
It intentionally skips `s00`, which is meant to be run standalone first (e.g. via
the manual commands above with `CONFIG=configs/abacus_subsets/abacus_2color_s00.json`,
substituting direct `python`/`./2color` calls for the `sbatch` wrappers):

```bash
zsh run_subsets.sh
```

### Predictions (plots only, no covariance)

For local runs where the covariance matrix would OOM:

```bash
python color_predict.py --run-dir output/abacus_2color_s00 --model 2color --no-cov --no-catalog
```

This produces diagnostic plots (`redshift_color.png`, `redshift_color_xonly.png`,
etc.) without computing the O(G²) covariance matrix. The `--xonly` flag is now
default (produces x-only plots alongside full-model plots). Predictions are made
on the ~13,759 holdout galaxies within the subset.

### Output structure

Each subset produces a full independent run:

```
output/abacus_2color_s00/
  ├── config.json, input.json, init.json
  ├── 2color_{1..4}.csv          (MCMC chains)
  ├── stansummary.txt, diagnose.txt
  ├── redshift_color.png         (full model residuals vs z)
  ├── redshift_color_xonly.png   (x-only residuals vs z)
  └── color_catalog.fits         (if --no-catalog not set)
output/abacus_2color_s01/
  └── ...
```

### Subset sizes

With `n_subsets=5` and 93,796 galaxies passing cuts:
- Subsets 0–3: 18,759 galaxies each (93796 // 5)
- Subset 4: 18,760 galaxies (remainder)
- Training: 5,000 per subset
- Holdout (for prediction): ~13,759 per subset
