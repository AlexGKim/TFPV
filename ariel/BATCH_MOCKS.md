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
   `intercept_plane`, `intercept_plane2`, `z_obs_min/max`, `n_objects=5000`,
   `random_seed=42`, `n_sigma`, `n_sigma_perp`). The hands-on Phase A (selection
   ellipse / fiducial selection) was already done to produce these values and is
   **not part of this pipeline**. Each per-file config differs from the base only
   in `run` and `fits_file`.

3. **Step 5e (metric build, ~7h) must be run once per mock dataset, then
   reused across all files in that dataset.** The DR1 metric
   (`output/DR1_v6_2color/metric.json`) is **not** transferable to mocks —
   confirmed empirically: using it caused repeated `cholesky_decompose`
   failures in the sampler, the chains ran at ~50s/iteration (vs. ~1–2s with
   a good metric), and 18h was not enough for 1000 samples. Build the metric
   once on any representative mock file, then copy it to all other run dirs.
   The reusable mock metric lives at `output/abacus_2color/metric.json` once
   built.

4. **Per-file run name** is the `c<NN>_ph<NN>_r<NN>` token from the filename
   (regex `c\d+_ph\d+_r\d+`), e.g.
   `TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits` → run
   `c000_ph000_r001`. Outputs go to `output/<run>/`.

5. **Per-file chain:**
   `step4 → step5d → step5e (first file only) → step6 ×4 → step7 → step8`.
   Subsequent files copy the metric from the first file and skip step5e.

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
| `make_batch_configs.py` | Generate per-file configs from a mock dir; seed each `output/<run>/metric.json`. |
| `slurm/batch_submit.sh` | Submit the full dependency chain per file (`--debug` for plumbing test). |
| `slurm/batch_status.sh` | Aggregate sentinel completion across all runs in a config dir. |
| `slurm/step6_chain.sh` | One MCMC chain; honors `DEBUG=1` (10+10 samples, standard CSV name). |
| `batch/job_tracker.csv` | Appended log of submitted SLURM job IDs per run. |
| `slurm/step{4,5d,6,7,8}_*.sh` | The underlying step scripts (unchanged; see `BATCH_NERSC.md`). |

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
    --metric output/abacus_2color/metric.json
bash slurm/batch_submit.sh configs/batch_debug --debug
watch bash slurm/batch_status.sh configs/batch_debug   # all 8 sentinels in ~10-15 min
```

Success = every `.step*_done` sentinel appears and `output/<run>/color_catalog.fits`
is written. Then drop `--debug` for the real run.

---

## Full Batch Run

```bash
# 1. Generate one config per fits file + seed each run's metric:
python3 make_batch_configs.py \
    --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
    --base configs/abacus_2color.json \
    --outdir configs/batch_v0.5.7 \
    --metric output/abacus_2color/metric.json

# 2. Submit (throttle so at most N files' chains are queued at once):
bash slurm/batch_submit.sh configs/batch_v0.5.7 8

# 3. Monitor:
squeue -u $USER
bash slurm/batch_status.sh configs/batch_v0.5.7            # summary
bash slurm/batch_status.sh configs/batch_v0.5.7 --verbose  # per-run detail
```

`batch_submit.sh` skips any run whose `.step8_done` sentinel already exists, so it
is safe to re-run to pick up failed/incomplete files.

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
sbatch --export=CONFIG=configs/batch_v0.5.7/c000_ph000_r001.json,CHAIN_ID=2 slurm/step6_chain.sh
# Re-run a whole file's remaining steps: clear its sentinel(s) and re-submit:
rm output/c000_ph000_r001/.step8_done
bash slurm/batch_submit.sh configs/batch_v0.5.7 8
# Per-run status:
bash slurm/check_status.sh configs/batch_v0.5.7/c000_ph000_r001.json
```

---

## Runtime / cost notes

- step4 (CPU debug, <5 min), step5d (GPU debug, ~5 min): cheap.
- **step6 dominates:** 4 chains × ~14h each (run in parallel) per file. With N
  files this is the GPU-hours driver — use the `MAX_CONCURRENT` throttle and mind
  the NERSC regular-GPU QOS limits.
- step7 (CPU debug, ~15 min), step8 (CPU, fast for 5000 objects).
- **step5e** (~7h) is run once on the first mock to build a mock-specific metric,
  then that metric is reused for all other files. The DR1 metric is **not**
  transferable to mocks (see decision #3 above).

---

## Subset Partition Mode (single file → 17 disjoint subsets)

When a single mock FITS file is too large for the prediction step to fit in memory
(e.g. 169k galaxies → OOM on the O(M×G) posterior predictive computation), split
it into disjoint subsets that each match the DR1 sample size (~10k galaxies).

### How it works

`desi_data.py` supports two new config fields:

```json
{
  "n_subsets": 17,      // total number of disjoint partitions
  "subset_index": 0     // which partition (0-indexed)
}
```

When both are present, `desi_data.py` replaces random subsampling with a
deterministic partition:

1. Apply all selection cuts (magnitude window, redshift, plane cuts) → N_after_cuts
2. Permute all N_after_cuts indices with `random_seed` (shared across subsets)
3. Split the permuted array into `n_subsets` contiguous chunks
4. Select chunk `subset_index`

This guarantees **zero overlap** between subsets and full coverage of all post-cut
galaxies. The `n_objects` field is ignored when partitioning.

### Config generation

17 configs live at `configs/abacus_2color_s00.json` through `s16.json`. They share
all selection parameters from `configs/abacus_2color.json` but differ in:
- `"run": "abacus_2color_s00"` … `"abacus_2color_s16"`
- `"subset_index": 0` … `16`
- `"n_subsets": 17`
- `"fits_file"`: local path `data/TF_AbacusSummit_…_appmag.fits`

To regenerate:

```python
import json
base = json.load(open('configs/abacus_2color.json'))
base['fits_file'] = 'data/TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11_appmag.fits'
base.pop('n_objects', None)
for i in range(17):
    cfg = dict(base)
    cfg['run'] = f'abacus_2color_s{i:02d}'
    cfg['subset_index'] = i
    cfg['n_subsets'] = 17
    with open(f'configs/abacus_2color_s{i:02d}.json', 'w') as f:
        json.dump(cfg, f, indent=2)
```

### Running one subset end-to-end

```bash
export CONFIG=configs/abacus_2color_s00.json

# Step 4: data prep (produces ~5500 objects per subset from 93k post-cut)
python desi_data.py --config $CONFIG

# Step 5d: MAP optimization
sbatch --export=CONFIG=$CONFIG slurm/step5d_map.sh

# Step 5e: skip — copy metric from the existing mock run
cp output/abacus_2color/metric.json output/abacus_2color_s00/metric.json

# Step 6+7+8: sampling + diagnostics + predictions
bash slurm/step6_submit.sh $CONFIG
```

### Running all 17 subsets

```bash
for i in $(seq -w 0 16); do
    CONFIG=configs/abacus_2color_s${i}.json
    python desi_data.py --config $CONFIG
    cp output/abacus_2color/metric.json output/abacus_2color_s${i}/metric.json
done

# Submit all (throttled):
for i in $(seq -w 0 16); do
    bash slurm/step6_submit.sh configs/abacus_2color_s${i}.json
done
```

### Predictions (plots only, no covariance)

For local runs where the covariance matrix would OOM:

```bash
python color_predict.py --run-dir output/abacus_2color_s00 --model 2color --no-cov --no-catalog
```

This produces diagnostic plots (`redshift_color.png`, `redshift_color_xonly.png`,
etc.) without computing the O(G²) covariance matrix. The `--xonly` flag is now
default (produces x-only plots alongside full-model plots).

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

With `n_subsets=17` and 93,796 galaxies passing cuts:
- Subsets 0–15: 5,517 galaxies each (93796 // 17)
- Subset 16: 5,524 galaxies (remainder)
