# Running the 2COLOR Pipeline on AbacusSummit Mocks (Batch)

This document covers running the 2COLOR Tully-Fisher pipeline over a **set of
AbacusSummit mock FITS files**, as a SLURM batch on NERSC. It is both a runbook
(for a person submitting the jobs) and an initial-condition brief (for a future
Claude session resuming this work). **Start here** for anything mock-related.

| You are running | Where | Doc |
|---|---|---|
| AbacusSummit mocks, as a batch | NERSC | **this file** — design decisions, config generation, submission |
| the mechanics of one run's steps | NERSC | [BATCH_NERSC.md](BATCH_NERSC.md) |
| diagnosing a failure | NERSC | [BATCH_CLAUDE.md](BATCH_CLAUDE.md) — step map, dependency graph, failure catalog |
| real DR2 data, whole catalog | locally | [DR2_SINGLE.md](DR2_SINGLE.md) |
| real DR2 data, split by morphology | locally | [DR2_TWOPOP.md](DR2_TWOPOP.md) |

This file covers what is *different* for the mock batch rather than duplicating
the other two. Two exceptions, because they are defined here and referenced from
elsewhere: **slice partitioning** (decisions #4/#4b) and **per-file dust
resolution** (#2c) are general pipeline mechanics that apply to DR2 too.

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

2b. **Initial conditions are also frozen and shared.** `configs/abacus_2color.json`
   sets `"fixed_init": "configs/fixed_init_2color.json"` — a set of
   physical-unit (`slope_orig`, `intercept_orig`, `sigma_int_x`, `w`, ...)
   parameter values from a hand-validated MAP fit. `desi_data.py` (step4)
   transforms these into each run's own standardized (`slope_std`,
   `intercept_std`) coordinates using that run's own freshly-computed
   `mean_x`/`sd_x` — a unit conversion, not a re-fit — and writes
   `output/<run>/init_MAP.json` directly. This is bound-safe by construction
   (`2color.stan`'s only data-dependent bounds, on `slope_std`/`intercept_std`,
   reduce to fixed physical-unit ranges `slope_orig ∈ [-9,-4]`,
   `intercept_orig ∈ [-24,-14]`, independent of `mean_x`/`sd_x`) and removes
   step5d's GPU MAP-optimize job from the chain entirely (see #5).

2c. **The dust uncertainty `d_err_r` is read per file from the FITS header, and
   is deliberately *not* frozen.** Unlike the selection cuts and the init, this
   one varies file to file, so each mock carries its own value on HDU 1.
   `color_predict.resolve_d_err_r()` resolves, first hit wins:

   | Source | Applies to |
   |---|---|
   | `cfg["dust_pickle"]` | DR2 (its FITS files carry no dust keywords) |
   | header `A_R_ERR` | the mocks produced for this batch |
   | header `DSTCFF_R_ERR` | earlier mocks, e.g. `..._zsnap0.20_zmax0.11.fits` (0.20456262) |
   | `_D_ERR_R` = 0.17680325 | built-in iron fallback — **wrong for mocks** |

   (`DSTCFF_R_ERR` is a `HIERARCH` card, being 12 characters; `A_R_ERR` is 7
   and so is an ordinary card. The lookup handles both without special casing.)

   It always logs which source won, and warns loudly on the fallback. Check
   `slurm/logs/step8_predict_*.out` for a line of the form
   `Loaded d_err_r = 0.20456262 mag from FITS header DSTCFF_R_ERR of <path>`
   — the keyword named will be whichever one that file actually carries. A
   `WARNING: no dust_pickle in config and no A_R_ERR/DSTCFF_R_ERR in the FITS
   header …` line means neither keyword was found and the covariance is wrong.
   Step 8 also records the value it used as a `d_err_r` attribute on
   `color_xonly_cov.h5`, which is what any downstream combine must read rather
   than re-deriving — re-deriving is how a combined product previously ended up
   with one dust value in its per-population blocks and another in its
   cross-population terms.

3. **There is no metric-building step.** Every chain starts from the
   **identity metric** and adapts a dense one during warmup
   (`metric=dense_e adapt save_metric=1`, no `metric_file=`), with
   `NUM_WARMUP=1000` as the entire adaptation budget. The scripts that built a
   metric — `slurm/step5e_metric.sh` and `make_metric.py` — have been
   **deleted**, and `--metric` has been removed from `batch_submit.sh` and
   `make_batch_configs.py`. This is a deliberate simplification, resting on
   three findings:

   - A local CPU A/B test found a pre-built metric ~2.7× slower *overall to
     obtain* than letting step6 adapt from identity, with no quality gain.
   - The builder was crude (100 post-warmup draws, `np.cov`, no
     regularization) over a parameter list that no longer matched the model.
   - A metric from a different data type was actively harmful: the DR1 metric
     on mocks caused repeated `cholesky_decompose` failures and
     ~50 s/iteration vs. ~1–2 s with a matched one.

   Warmup from identity is known to be sufficient here: the validated abacus
   slice completed warmup in ~4.4 h/chain on CPU without stalling at max
   treedepth. That is only true for the **rank-1** model (`S = w wᵀ`) — the
   earlier rank-2 parameterization did funnel, which is what Pathfinder
   seeding was introduced to fix.

   > The **local real-data** workflow still seeds a Pathfinder-built metric —
   > see [DR2_TWOPOP.md](DR2_TWOPOP.md) Step 5e and `make_pf_metric.py`. That
   > is retained deliberately and is separate from the batch; only the crude
   > short-chain builder was removed.

4. **Every mock file is split into `n_subsets` disjoint slices** (see "Subset
   Partition Mode" below) — this isn't just for the one oversized 170k-galaxy
   test file, it's the standard mode `make_batch_configs.py` uses for every
   file in the batch. The split is applied to the **valid pre-selection-cut**
   rows, so each slice stands in for a standalone FITS file (its own cuts,
   training sample, holdout, and full-sample-vs-MAIN contrast).
   **Run name** is the `c<NN>_ph<NN>_r<NN>` file token
   (regex `c\d+_ph\d+_r\d+`) plus a `_s<NN>` subset suffix, e.g.
   `TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits` → runs
   `c000_ph000_r001_s00` … `c000_ph000_r001_s04` (for `n_subsets=5`). Outputs
   go to `output/<run>/`, one independent run dir per (file, slice) pair.

4b. **Only slice `s00` is run.** `--n-subsets` and `--subsets` are independent:
   the first sets the slice *size*, the second says which slices actually run.
   The batch keeps `--n-subsets 5` and passes `--subsets 0`, so each file
   contributes exactly one run — **125 runs, not 625**. What this gives up is
   explicit: of each file's ~90,756 cut-passing galaxies, only slice 0's
   ~18,167 are fit and ~13,167 predicted; the other four slices are simply not
   run. The 5-way mechanism remains available (it is what makes the slice size
   meaningful, and `--subsets 0,1` etc. widens coverage later) but is not the
   operating mode.

   > **Never express "one slice" as `--n-subsets 1`.** That makes the single
   > slice the whole file (~170,781 valid rows for the reference mock), and
   > step 8's covariance dimension becomes the file's full MAIN count,
   > G ≈ 91,000 — a **67 GB** dense matrix that cannot finish inside step8's
   > 30-minute walltime. Keep `--n-subsets 5` and narrow with `--subsets`.

5. **Per-(file, slice) chain:** `step4 → step6 ×4 → step7 → step8`
   (no step5d — see #2b; no metric step — see #3). **Cost implication:** with
   `--subsets 0` the totals are `n_files` runs and `n_files × 4` step6 chains
   — for 125 files, **125 runs and 500 step6 chains**. (Running all five
   slices would be 625 runs / 2500 chains; see "Runtime / cost notes".)

**Target mock set:** `v0.5.7`
(`/global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7/`),
the `base`/`fullmocks` family (`TF_AbacusSummit_base_..._zsnap..._zmax....fits`,
`"source": "fullmocks"`, relies on a `MAIN` column) — this is what
everything below in this document covers. The batch driver takes a `--dir`,
so it processes whatever files are present when more arrive. (Other
populated sets exist: `DR2/.../v0.5.6/` 675 files, and older
`mocks/TF_mocks/fullmocks/v0.5.1–4/` 675 each.)

**A separate `spec`/`v0.5.8` family** also exists
(`v0.5.8/TF_AbacusSummit_spec_c###_ph###_r###.fits`, `"source": "DESI"`, no
`MAIN` column). It uses a **different** config generator,
`make_spec_batch_configs.py`, because unlike decision #2 above, its
selection cuts are only *partly* frozen: `haty_min`/`haty_max`/
`z_obs_min`/`z_obs_max`/`n_sigma_perp` are shared across the batch, but
`slope_plane`/`intercept_plane`/`intercept_plane2` are re-derived per file
from that file's own Maximum-Likelihood fit (Step 2), not copied from a
single base config — see `make_spec_batch_configs.py`'s docstring for the
exact construction. It also doesn't use `n_subsets` partitioning (file
sizes so far are comparable to the `base` family's single-subset scale).

---

## File / Command Map

| Artifact | Role |
|----------|------|
| `configs/abacus_2color.json` | Base config: frozen selection cuts + `fixed_init` + test fits file. |
| `configs/fixed_init_2color.json` | Frozen physical-unit init values (`slope_orig`, `intercept_orig`, `sigma_int_x`, `w`, ...) from a hand-validated MAP fit; transformed per-run into standardized coordinates by `desi_data.py`, skipping step5d. |
| `make_batch_configs.py` | Generate per-(file, slice) configs from a mock dir. `--n-subsets` sets the slice size (5), `--subsets` which slices run (`0`). |
| `make_spec_batch_configs.py` | Generate per-file configs for the `spec`/DESI-source family, re-deriving slope_plane/intercepts per file from its own MLE fit. |
| `slurm/batch_submit.sh` | Submit the full dependency chain per run (`--debug` for plumbing test). |
| `slurm/batch_status.sh` | Aggregate sentinel completion across all runs in a config dir. |
| `slurm/step6_node.sh` | All 4 MCMC chains, 1 node/4 GPUs (`CUDA_VISIBLE_DEVICES`); honors `DEBUG=1` (15 samples, no adaptation). `step6_chain.sh` still exists for resubmitting a single failed chain. |
| `batch/job_tracker.csv` | Appended log of submitted SLURM job IDs per run. |
| `slurm/step{4,6,7,8}_*.sh` | The underlying step scripts (see `BATCH_NERSC.md`). `step5d_map.sh` is no longer part of the standard chain — kept for manual/standalone use on any config without a `fixed_init`. |

---

## One-Time Setup

```bash
cd $SCRATCH/TFPV/ariel        # or wherever the repo lives on NERSC
git checkout batch && git pull
module load craype-accel-nvidia80 cudatoolkit nvidia PrgEnv-nvidia
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64

# GPU binary 2color_g must exist (already compiled). If not:
#   sbatch slurm/compile_2color_gpu.sh
```

No metric-seeding step — chains start from the identity metric and adapt
during warmup (see Mock Batch Overview decision #3).

---

## Step 0 — Validate on a single mock (recommended before any batch)

Use a **sliced** config, not the bare base config:

```bash
export CONFIG=configs/abacus_subsets/abacus_2color_s00.json
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
# verify output/abacus_2color_s00/input.json (N up to 5000, sane ranges),
# data.png, and init_MAP.json (finite, no NaN — written directly by step4
# since the config sets "fixed_init"; no step5d needed)
bash slurm/step6_submit.sh $CONFIG   # 4 chains + auto step7/8
```

> **Do not run Step 0 with `configs/abacus_2color.json` itself.** It carries no
> `n_subsets`/`subset_index`, so nothing is slice-scoped and step8's covariance
> dimension becomes the whole file's MAIN count — G ≈ 91,000 for the reference
> mock, i.e. a **67 GB** dense matrix that will not finish inside step8's 30-min
> debug walltime (and the same run's O(M×G) prediction temporaries are ~5.5 GB
> each). The base config exists to be cloned by `make_batch_configs.py` /
> `configs/abacus_subsets/`, which add the partition fields. Validate with a
> sliced config so Step 0 exercises the same code path as the batch.

After it completes, check `output/abacus_2color_s00/stansummary.txt` (R̂ < 1.01,
ESS > 100/chain) and `output/abacus_2color_s00/diagnose.txt` (divergences,
max-treedepth %).

---

## Debug Mode — fast end-to-end plumbing test

Before a real batch, confirm the *plumbing* (config generation, dependencies,
sentinels, output FITS) with short chains on the debug queue. Debug chains
skip adaptation and sample at a fixed known-good stepsize (0.08), so each
completes in a few minutes (step6 gets `-t 00:20:00` on the debug GPU queue).
Results are **not** science-grade. (Do not use `num_warmup<20` with
adaptation: Stan then disables adaptation, falls back to a tiny stepsize, and
a single iteration can exceed 10 min.)

```bash
python3 make_batch_configs.py \
    --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
    --base configs/abacus_2color.json \
    --outdir configs/batch_debug \
    --n-subsets 5 --subsets 0 --n-objects 5000 --run-suffix _dbg
bash slurm/batch_submit.sh configs/batch_debug --debug
watch bash slurm/batch_status.sh configs/batch_debug   # all sentinels in ~10-15 min
```

**`--run-suffix _dbg` is not optional here.** Run names derive from the file
token, not from `--outdir`, so without a suffix a debug batch writes into the
same `output/<run>/` dirs as the real batch and leaves `.step8_done` sentinels
behind — and `batch_submit.sh` skips any run whose `.step8_done` exists, so the
real run would be silently skipped in favour of throwaway debug output.

This generates one config/run per mock file (`--subsets 0`), so a debug run
against a handful of files stays small and fast on the debug queue.

Success = every `.step*_done` sentinel appears and
`output/<run>/color_xonly_catalog.fits` is written. Then drop `--debug` for
the real run.

---

## Full Batch Run

```bash
# 1. Generate one s00 config per fits file (slice size 1/5, only slice 0 run):
python3 make_batch_configs.py \
    --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
    --base configs/abacus_2color.json \
    --outdir configs/batch_v0.5.7 \
    --n-subsets 5 --subsets 0 --n-objects 5000

# 2. Submit (throttle so at most N runs' step6_node jobs are queued at once):
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

`input.json`, `init.json`, `init_MAP.json`, `2color_1.csv`…`2color_4.csv`,
`stansummary.txt`, `diagnose.txt`, `2color.png`, `color_xonly_catalog.fits`,
`color_xonly_cov.h5` (x-only is the default; pass `--full` to step8 for
`color_catalog.fits`/`color_cov.h5` too), `explore_residuals/`, and the
`.step*_done` sentinels.
No `metric.json` — there is no metric-building step (see decision #3). Each
chain's own adapted metric is saved as `2color_{1..4}_metric.json` for
inspection; nothing reads it back.

Step 8 runs `color_predict.py` first and `explore_residuals.py` second, so the
catalog and covariance are already written before any residual plot is
attempted — a plotting failure under `set -e` can no longer cost a run its
science output. The covariance carries a `d_err_r` HDF5 attribute recording the
dust value used (decision #2c).

### The batch as a whole

**125 independent per-file outputs — there is no combine step, and none is
intended.** Each mock file yields its own `color_xonly_catalog.fits` +
`color_xonly_cov.h5` for its slice 0, consumed separately (e.g. as independent
realizations for scatter across mocks).

This is deliberately unlike the DR2 two-population case, where
`combine_color_xonly.py` merges the spiral and irregular outputs into one
`DESI-DR2_TF_pv_cat_v5b.fits` + `_cov.h5` pair. Combining the mock batch is not
merely unimplemented but infeasible: 125 × ~18,167 galaxies is ~2.27M rows, and
a dense covariance over them would be ~2.27M² × 8 bytes ≈ **40 TB**. Treat the
per-file products as the batch's final output.

---

## Resubmitting failures

```bash
# A single chain (after step4 done):
sbatch --export=CONFIG=configs/batch_v0.5.7/c000_ph000_r001_s00.json,CHAIN_ID=2 slurm/step6_chain.sh
# Re-run one (file, subset) run's remaining steps: clear its sentinel(s) and re-submit:
rm output/c000_ph000_r001_s00/.step8_done
bash slurm/batch_submit.sh configs/batch_v0.5.7 8
# Per-run status:
bash slurm/check_status.sh configs/batch_v0.5.7/c000_ph000_r001_s00.json
```

---

## Runtime / cost notes

Totals below assume the `--subsets 0` mode: **125 runs** for 125 mock files.

- step4 (CPU debug, <5 min): cheap, one per run. Writes init_MAP.json
  directly (`fixed_init` is set), so no step5d GPU job is needed at all —
  eliminates 125 GPU MAP-optimize submissions and their queue-wait time (the
  GPU-hour magnitude was already small; the real win is fewer job-submission/
  queue cycles and removing the init-boundary failure mode step5d could hit).
- **step6 dominates GPU-hours:** 4 chains (run in parallel, 1 per GPU on a
  single `step6_node.sh` node) per run; timing depends on
  `NUM_WARMUP`/`MAX_DEPTH` (default 1000/10 — re-measure per dataset, don't
  assume older ~14h-per-chain figures quoted for the previous 250/8
  defaults). At one slice per file that is `n_files × 4` = **500 step6
  chains**. Running all five slices instead would be 2500 chains and 5× the
  GPU-hours — which is why the batch runs s00 only (decision #4b). Note the
  practical scale: the single validation run queued ~21.6 h before the
  scheduler even estimated a start, so queue wait, not GPU-hours, is the
  binding constraint.
- **step6 job-submission count is now `n_files × n_subsets`, not ×4.**
  `step6_node.sh` runs all 4 chains as backgrounded processes on the 4 GPUs of
  one already-allocated node (`sacct` confirms a `--gpus-per-task=1` job gets
  the whole 4-GPU node anyway — nothing here is fractionally shared), instead
  of `batch_submit.sh` submitting 4 separate `step6_chain.sh` jobs per run. This
  matters most on constrained QOS like debug (`MaxSubmitPU=5`): fanning out 5
  subsets × 4 old-style chain jobs could only fit ~1 run's GPU jobs in the
  queue at once and required manually threading each run's submission through
  as slots freed — with `step6_node.sh` each run needs only 1 debug-GPU slot,
  so far more runs fit in flight simultaneously. Use the
  `MAX_CONCURRENT` throttle (now counts `step6_node` jobs, i.e. runs, directly)
  and mind the NERSC regular-GPU QOS limits for the real (non-debug) batch.
- step7 (CPU debug, ~15 min) — one per run.
- step8 (CPU debug, ~5–10 min per slice-scoped run: `color_predict.py` then
  `explore_residuals.py`, dominated by the O(G²) covariance) — per (file,
  subset). Each run's `color_xonly_cov.h5` is ~1.17 GB, so budget ~150 GB of
  scratch for a 125-run batch; `--no-cov` skips it.
- **There is no metric-building step** (see decision #3), so nothing beyond
  step4/6/7/8 contributes to the budget. Warmup from identity is the whole
  adaptation cost, which is why `NUM_WARMUP` dominates step6's wall clock.

---

## Subset Partition Mode (mechanism, and a single-file manual walkthrough)

When a mock FITS file is too large for the prediction step to fit in memory (e.g.
170k galaxies → OOM on the O(M×G) posterior predictive computation), split it into
disjoint slices, each of which then behaves like its own standalone FITS file:
its own selection cuts, its own 5,000-galaxy training sample (matching DR2), its
own holdout for prediction, and its own full-sample-vs-MAIN
contrast. **This is not an opt-in special case
for oversized outliers** — decisions #4–#5 above establish it as the standard mode
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

1. Permute the **valid, pre-selection-cut** rows with `random_seed` (shared
   across subsets) and split into `n_subsets` contiguous chunks
2. Select chunk `subset_index` — this is the **slice**, recorded as
   `slice_sga_ids` (~34,156 valid rows for the reference file)
3. Apply the selection cuts (magnitude window, redshift, plane cuts) *within*
   the slice → the cut-passing galaxies, recorded as `subset_sga_ids`
   (~18,167); the rest of the slice is the cut-failing population
4. Subsample `n_objects` from those for training (5,000), recorded as
   `train_sga_ids`

**The partition is applied before the selection cuts, not after** — this is
what makes each subset behave like a standalone FITS file: it carries its own
cut-passing *and* cut-failing galaxies, so it has its own genuine
full-sample-vs-MAIN contrast. Partitioning the post-cut sample instead (the
earlier behavior) put 100% of every subset inside MAIN and left every
cut-failing galaxy outside all subsets, which made that contrast degenerate
and forced `explore_residuals.py`/`color_predict.py` to compute over the
entire catalog to find any — the O(M×G) blowup that OOMs at ~170k-galaxy
scale.

This guarantees **zero overlap** between slices and full coverage of the valid
sample. Predictions are made on the holdout within the slice (~13,167 galaxies
that pass the cuts but were not used for training).

`explore_residuals.py` and `color_predict.py` both restrict their
posterior-predictive computation to `slice_sga_ids` via `_slice_mask()`
(`color_predict.py`, the single source of truth — don't reimplement it
inline). `MU_TF`/`MU_ERR`/`LOGDIST` are therefore `NaN` outside the slice;
`MAIN`/`ANALYSIS` were already slice-scoped, and `combine_color_xonly.py`
reads only `MAIN=True` rows, so nothing downstream depends on the discarded
values.

### Config generation

5 configs live at `configs/abacus_subsets/abacus_2color_s00.json` through `s04.json`.
They share all selection parameters from `configs/abacus_2color.json` but differ in:
- `"run": "abacus_2color_s00"` … `"abacus_2color_s04"`
- `"subset_index": 0` … `4`
- `"n_subsets": 5`
- `"n_objects": 5000`
- `"fits_file"`: local path `data/TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits`

To regenerate:

```python
import json, os
base = json.load(open('configs/abacus_2color.json'))
base['fits_file'] = 'data/TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits'
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

`batch_submit.sh` handles the full chain for every config in a directory:

```bash
bash slurm/batch_submit.sh configs/abacus_subsets
```

This submits `step4 → step6×4 → step7 → step8` for all 5 subsets with
SLURM dependencies, throttled to 8 concurrent `step6_node` jobs (i.e. 8 runs,
each holding one whole 4-GPU node) by default.

### Running one subset end-to-end (manual, NERSC / SLURM)

```bash
export CONFIG=configs/abacus_subsets/abacus_2color_s00.json

sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
# After step4 done (writes init_MAP.json directly, since the config sets
# "fixed_init" -- no step5d needed):
bash slurm/step6_submit.sh $CONFIG
```

### Running locally without SLURM

`run_subsets.sh` is the no-scheduler counterpart to `slurm/batch_submit.sh`,
mirroring the same chain (step4 → step6 ×4 → step7 → step8, with step5d
skipped when the config sets `fixed_init`) via the CPU `./2color` binary, with
the 4 chains as background processes instead of one per GPU. Its sampler
settings track `slurm/step6_node.sh`'s non-debug defaults
(`num_warmup=1000`, `max_depth=10`, `delta=0.9`, no metric seeding), so local
results stay comparable to batch results; override via `NUM_WARMUP`,
`NUM_SAMPLES`, `MAX_DEPTH`, `DELTA`, or `NO_COV=1` in the environment.

It takes an optional subset list and defaults to s01–s04, skipping `s00`
because that one is normally run standalone first as the Step 0 validation:

```bash
zsh run_subsets.sh                 # s01 s02 s03 s04 (default)
zsh run_subsets.sh 00              # just s00
zsh run_subsets.sh 00 01 02 03 04  # all five
```

### Predictions (plots only, no covariance)

For local runs where the covariance matrix would OOM:

```bash
python color_predict.py --run-dir output/abacus_2color_s00 --model 2color --no-cov --no-catalog
```

This produces diagnostic plots (`redshift_color_xonly.png`, etc.) without
computing the O(G²) covariance matrix. x-only is the unconditional default
(pass `--full` for the full z/g-conditioned model's plots too). Predictions
are made on the ~13,167 holdout galaxies within the slice.

### Output structure

Each slice produces a full independent run:

```
output/abacus_2color_s00/
  ├── config.json, input.json, init.json
  ├── init_MAP.json              (written by step4 — fixed_init, no step5d)
  ├── data.png
  ├── 2color_{1..4}.csv          (MCMC chains)
  ├── 2color_{1..4}_metric.json  (each chain's own adapted metric)
  ├── stansummary.txt, diagnose.txt, 2color.png
  ├── redshift_color_xonly.png   (x-only residuals vs z, default)
  ├── color_xonly_catalog.fits   (if --no-catalog not set)
  ├── color_xonly_cov.h5         (if --no-cov not set; carries the d_err_r attr)
  ├── explore_residuals/         (step8, after color_predict.py)
  └── .step{4,6_chain1..4,7,8}_done   (sentinels)
```

In the `--subsets 0` mode only `_s00` exists per file; `_s01`…`_s04` are not
generated or run.

### Subset sizes

Slices are cut from the **valid pre-selection-cut** sample, so their size is
set by the valid-row count, not the post-cut count. For the reference file
(`TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits`, 170,781 valid
rows, `n_subsets=5`, `haty_min=-21.6`/`haty_max=-18.4`):

- Slices 0–3: 34,156 valid rows each (170781 // 5); slice 4 takes the remainder
- Of slice 0: 18,167 pass the selection cuts, 15,989 fail
  (the fail population is what gives each slice its own MAIN contrast)
- Training: 5,000 per slice, drawn from the cut-passing galaxies
- Holdout (for prediction): 13,167 for slice 0

These figures are what step4 actually reports for this file; the log line to
compare against is
`Slice 0/5: 34156 valid rows from 170781 …; of those 18167 pass the selection
cuts (15989 fail -> MAIN contrast population)`. The per-slice cut-passing count
varies a little across slices (18,092–18,256 for slices 1–4), since the split
is over valid rows and the cuts are applied within each slice.

This file is also the **reference/validation mock** used for Step 0 and for the
numbers quoted throughout this document. Note its dust keyword is the older
`DSTCFF_R_ERR` (0.20456262); production mocks are expected to carry `A_R_ERR`
instead (decision #2c).

The O(M×G) posterior-predictive work in step8 is scoped to the slice, so
`G` ≈ 34,156 rather than the full 170,781 — about 1.1 GB per dense
`(draws, galaxies)` temporary at 4,000 draws instead of ~5.5 GB.
