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
elsewhere: the **fixed-size subsample draw** (decision #4) and **per-file dust
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

   **The fallback is now unreachable for a generated batch.**
   `make_batch_configs.py` reads every file's header and refuses to emit *any*
   config if one lacks both keywords, naming the offenders. A missing dust value
   costs seconds at generation instead of a step-6 GPU allocation followed by a
   quietly wrong covariance. The runtime fallback survives only for paths that
   legitimately have no dust information.

2d. **The photsys calibration systematic comes from the `PHOTSYS_ERR` column.**
   DESI catalogs flag the northern footprint with a 1-character `PHOTSYS` and take
   the built-in `_D_A_SYS` = 0.02 floor. The mocks instead store `PHOTSYS` as a
   *numeric* offset (0.0 / −0.0234) plus a `PHOTSYS_ERR` column whose nonzero
   value is exactly 0.02, on the same 51,550 of 170,781 rows. Reading the column
   keeps the value with the data rather than hardcoded.

   This was silently broken: `np.where(photsys == 'N', 0.02, 0.0)` against a
   float64 column is elementwise-False without raising, so the term vanished from
   every mock covariance with no warning. It now raises if it cannot be resolved,
   and `make_batch_configs.py` checks for the column at generation time alongside
   the dust keyword. **Any mock covariance built before this must be
   regenerated** — it is missing a rank-1 block over ~30% of galaxies.

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
   run completed warmup in ~4.4 h/chain on CPU without stalling at max
   treedepth. That is only true for the **rank-1** model (`S = w wᵀ`) — the
   earlier rank-2 parameterization did funnel, which is what Pathfinder
   seeding was introduced to fix. Rank-1 removed the funnel, so the workaround
   went with it.

   > The **local real-data** workflow runs from identity as well, with the same
   > sampler arguments — see [DR2_TWOPOP.md](DR2_TWOPOP.md) Step 6. The two
   > paths sample identically by design. `make_pf_metric.py` survives for
   > manual experiments but is not a step in either.

4. **Each mock file contributes exactly one run, size-matched to the real DR2
   sample.** Step 4 restricts to the file's `MAIN` rows (see #4a), applies the
   frozen trapezoid cuts to them (90,119 cut-passing galaxies for the reference
   mock) and then draws **exactly `target_main_count = 17234`** of them at
   random. That number is
   the MAIN count of the real DR2 product, `DESI-DR2_TF_pv_cat_v5b.fits`, whose
   covariance `DESI-DR2_TF_pv_cat_v5b_cov.h5` is correspondingly
   `(17234, 17234)` — the two agree exactly. Matching it means mock and data
   samples carry comparable statistical weight, so their uncertainties are
   directly comparable.

4a. **Step 4 restricts mocks to `MAIN` rows**, because that is the population the
   frozen cuts were *derived* from: `selection_ellipse.py` filters to `MAIN` when
   `source: fullmocks`, so Phase A saw 154,976 of the reference mock's 170,781
   valid rows. Applying the resulting trapezoid to the whole file mixed in a
   population the ellipse never saw — `MAIN` is exactly `~DWARF`, a type-selected
   and therefore magnitude-correlated subset, and 637 of the 90,756 cut-passing
   galaxies were `DWARF` (0.70%, ~121 per draw). DR2 has no such asymmetry: its
   Phase A applies no `MAIN` filter, so derivation and application already agree
   there.

   The filter is gated on `source == "fullmocks"`, not on the column merely
   existing: the DR2 per-population FITS files carry their own pipeline-*written*
   `MAIN` column from an earlier `color_predict.py` run, and filtering on that
   would silently re-select a DR2 run against a stale selection. A mock with no
   `MAIN` column is a hard error.

   The draw is a **single step**, not an iteration or a partition, because the
   cuts are frozen: the post-cut population is fully determined before we
   choose, so the target is reachable exactly. A fraction-of-the-file scheme
   cannot hit it — the post-cut count falls out of the fraction rather than
   being chosen (1/5 of a file gives 18,092–18,256, 1/6 gives 15,039–15,211;
   17,234 would need 1/5.27). There is **no slice concept** anywhere in the
   pipeline any more.

   The drawn sample **is** MAIN: only these galaxies are analysed downstream.
   Within them, `n_objects = 5000` are the training set and the remaining
   **12,234** are the holdout — the same 5,000/12,234 split as DR2.

   **Run name** is the `c<NN>_ph<NN>_r<NN>` file token (regex
   `c\d+_ph\d+_r\d+`), with no suffix, e.g.
   `TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits` → run
   `c000_ph000_r001` → `output/c000_ph000_r001/`. **125 files → 125 runs.**

   > **`target_main_count` is what keeps step 8 tractable.** Without it, MAIN
   > becomes the file's whole cut-passing population, G ≈ 91,000 — a **67 GB**
   > dense covariance that cannot finish inside step8's 30-minute walltime. At
   > 17,234 the covariance is ~1.19 GB. `make_batch_configs.py` hard-errors if
   > the base config lacks the key.

   > **Consequence, stated plainly:** because the subsample is drawn from
   > cut-passing galaxies only, `explore_residuals.py`'s
   > full-sample-vs-MAIN contrast is degenerate for mock runs — both curves are
   > the same 17,234 galaxies. That follows from analysing MAIN only.

   **If a file cannot reach the target** (fewer than 17,234 cut-passing
   galaxies), step 4 exits non-zero naming the shortfall and
   `batch_submit.sh` never advances that file, rather than emitting a run
   whose sample size is silently unmatched.

5. **Per-file chain:** `step4 → step6 ×4 → step7 → step8`
   (no step5d — see #2b; no metric step — see #3). **Cost implication:** the
   totals are `n_files` runs and `n_files × 4` step6 chains — for 125 files,
   **125 runs and 500 step6 chains**.

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
exact construction. It also doesn't use `target_main_count` (file sizes so far
are comparable to the drawn-subsample scale).

---

## File / Command Map

| Artifact | Role |
|----------|------|
| `configs/abacus_2color.json` | Base config: frozen selection cuts + `fixed_init` + `target_main_count: 17234` + test fits file. |
| `configs/fixed_init_2color.json` | Frozen physical-unit init values (`slope_orig`, `intercept_orig`, `sigma_int_x`, `w`, ...) from a hand-validated MAP fit; transformed per-run into standardized coordinates by `desi_data.py`, skipping step5d. |
| `make_batch_configs.py` | Generate one config per file from a mock dir. Sample size comes from the base config's `target_main_count`; hard-errors if it is missing. |
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

The base config is now directly runnable — it carries `target_main_count`, so it
exercises the same code path as the batch:

```bash
export CONFIG=configs/abacus_2color.json
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
# verify output/abacus_2color/input.json:
#   len(subset_sga_ids) == 17234   (exactly)
#   len(train_sga_ids)  == 5000    (holdout 12234)
#   provenance present: target_main_count, random_seed, N_after_cuts
#   (the FITS path lives in config.json — input.json must stay string-free)
# and data.png plus init_MAP.json (finite, no NaN — written directly by step4
# since the config sets "fixed_init"; no step5d needed)
bash slurm/step6_submit.sh $CONFIG   # 4 chains + auto step7/8
```

> **Check `len(subset_sga_ids) == 17234` before submitting step 6.** If the key
> is absent, the config lost `target_main_count` and step 8 will try a
> G ≈ 91,000 (**67 GB**) dense covariance that cannot finish inside its
> 30-minute walltime — and the same run's O(M×G) prediction temporaries would be
> ~5.5 GB each.

After it completes, check `output/abacus_2color/stansummary.txt` (R̂ < 1.01,
ESS > 100/chain) and `output/abacus_2color/diagnose.txt` (divergences,
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
    --n-objects 5000 --run-suffix _dbg
bash slurm/batch_submit.sh configs/batch_debug --debug
watch bash slurm/batch_status.sh configs/batch_debug   # all sentinels in ~10-15 min
```

**`--run-suffix _dbg` is not optional here.** Run names derive from the file
token, not from `--outdir`, so without a suffix a debug batch writes into the
same `output/<run>/` dirs as the real batch and leaves `.step8_done` sentinels
behind — and `batch_submit.sh` skips any run whose `.step8_done` exists, so the
real run would be silently skipped in favour of throwaway debug output.

This generates one config/run per mock file, so a debug run
against a handful of files stays small and fast on the debug queue.

Success = every `.step*_done` sentinel appears and
`output/<run>/color_xonly_catalog.fits` is written. Then drop `--debug` for
the real run.

---

## Full Batch Run

```bash
# 1. Generate one config per fits file (sample size comes from the base config's
#    target_main_count; the generator hard-errors if it is missing):
python3 make_batch_configs.py \
    --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
    --base configs/abacus_2color.json \
    --outdir configs/batch_v0.5.7 \
    --n-objects 5000

# 2. Submit (throttle so at most N runs' step6_node jobs are queued at once):
bash slurm/batch_submit.sh configs/batch_v0.5.7 8

# 3. Monitor:
squeue -u $USER
bash slurm/batch_status.sh configs/batch_v0.5.7            # summary
bash slurm/batch_status.sh configs/batch_v0.5.7 --verbose  # per-run detail
```

`batch_submit.sh` skips any run whose `.step8_done` sentinel already exists, so it
is safe to re-run to pick up failed/incomplete runs.

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
`color_xonly_cov.h5` for its 17,234-galaxy draw, consumed separately (e.g. as independent
realizations for scatter across mocks).

This is deliberately unlike the DR2 two-population case, where
`combine_color_xonly.py` merges the spiral and irregular outputs into one
`DESI-DR2_TF_pv_cat_v5b.fits` + `_cov.h5` pair. Combining the mock batch is not
merely unimplemented but infeasible: 125 × 17,234 galaxies is ~2.15M rows, and
a dense covariance over them would be ~2.15M² × 8 bytes ≈ **37 TB**. Treat the
per-file products as the batch's final output.

---

## Resubmitting failures

```bash
# A single chain (after step4 done):
sbatch --export=CONFIG=configs/batch_v0.5.7/c000_ph000_r001.json,CHAIN_ID=2 slurm/step6_chain.sh
# Re-run one file's remaining steps: clear its sentinel(s) and re-submit:
rm output/c000_ph000_r001/.step8_done
bash slurm/batch_submit.sh configs/batch_v0.5.7 8
# Per-run status:
bash slurm/check_status.sh configs/batch_v0.5.7/c000_ph000_r001.json
```

---

## Runtime / cost notes

Totals below assume one run per file: **125 runs** for 125 mock files.

- step4 (CPU debug, <5 min): cheap, one per run. Writes init_MAP.json
  directly (`fixed_init` is set), so no step5d GPU job is needed at all —
  eliminates 125 GPU MAP-optimize submissions and their queue-wait time (the
  GPU-hour magnitude was already small; the real win is fewer job-submission/
  queue cycles and removing the init-boundary failure mode step5d could hit).
- **step6 dominates GPU-hours:** 4 chains (run in parallel, 1 per GPU on a
  single `step6_node.sh` node) per run; timing depends on
  `NUM_WARMUP`/`MAX_DEPTH` (default 1000/10 — re-measure per dataset, don't
  assume older ~14h-per-chain figures quoted for the previous 250/8
  defaults). At one run per file that is `n_files × 4` = **500 step6
  chains**. Note the
  practical scale: the single validation run queued ~21.6 h before the
  scheduler even estimated a start, so queue wait, not GPU-hours, is the
  binding constraint.
- **step6 job-submission count is now `n_files`, not `n_files × 4`.**
  `step6_node.sh` runs all 4 chains as backgrounded processes on the 4 GPUs of
  one already-allocated node (`sacct` confirms a `--gpus-per-task=1` job gets
  the whole 4-GPU node anyway — nothing here is fractionally shared), instead
  of `batch_submit.sh` submitting 4 separate `step6_chain.sh` jobs per run. This
  matters most on constrained QOS like debug (`MaxSubmitPU=5`): 4 old-style
  chain jobs per run could only fit ~1 run's GPU jobs in the queue at once and
  required manually threading each run's submission through as slots freed —
  with `step6_node.sh` each run needs only 1 debug-GPU slot,
  so far more runs fit in flight simultaneously. Use the
  `MAX_CONCURRENT` throttle (now counts `step6_node` jobs, i.e. runs, directly)
  and mind the NERSC regular-GPU QOS limits for the real (non-debug) batch.
- step7 (CPU debug, ~15 min) — one per run.
- step8 (CPU debug, ~5–10 min per run: `color_predict.py` then
  `explore_residuals.py`, dominated by the O(G²) covariance) — one per file.
  Each run's `color_xonly_cov.h5` is ~1.19 GB, so budget ~150 GB of
  scratch for a 125-run batch; `--no-cov` skips it.
- **There is no metric-building step** (see decision #3), so nothing beyond
  step4/6/7/8 contributes to the budget. Warmup from identity is the whole
  adaptation cost, which is why `NUM_WARMUP` dominates step6's wall clock.

---

## The Subsample Draw (mechanism, and a single-file manual walkthrough)

Every mock run analyses a **fixed-size random draw** from the file's
cut-passing galaxies rather than the whole file. This section documents the
mechanism and walks through one file by hand — useful for understanding what's
happening, debugging a single run, or validating locally before a real
multi-file batch.

Two things motivate it, and they point the same way:

1. **Size-matching to the data.** The draw is `target_main_count = 17234`, the
   MAIN count of `DESI-DR2_TF_pv_cat_v5b.fits` (whose covariance
   `DESI-DR2_TF_pv_cat_v5b_cov.h5` is `(17234, 17234)` — the two agree
   exactly). Mock and real samples then carry comparable statistical weight.
2. **Memory.** The O(M×G) posterior predictive and the O(G²) dense covariance
   in step 8 cannot be run over a whole file: G ≈ 91,000 cut-passing galaxies
   is a **67 GB** covariance and ~5.5 GB per `(draws, galaxies)` temporary. At
   G = 17,234 the covariance is ~1.19 GB and each temporary ~0.55 GB.

### How it works

`desi_data.py` takes one config field (plus the training-size knob):

```json
{
  "source": "fullmocks",       // restricts step 4 to MAIN rows
  "target_main_count": 17234,  // galaxies drawn from the cut-passing sample
  "n_objects": 5000,           // training sample size within the draw
  "random_seed": 42            // seeds both draws
}
```

When `target_main_count` is present, `desi_data.py`:

1. Restricts to `MAIN` rows when `source: fullmocks` (154,976 of 170,781 for the
   reference mock), then applies the selection cuts (magnitude window, redshift,
   plane cuts) → 90,119 cut-passing galaxies. See decision #4a for why the MAIN
   restriction matters and why it is gated on `source`.
2. Draws exactly `target_main_count` of them with
   `default_rng(random_seed).choice(..., replace=False)`, sorted — recorded as
   `subset_sga_ids`. **This drawn sample is MAIN**; only these galaxies are
   analysed downstream.
3. Draws `n_objects` of *those* for training with the derived seed
   `random_seed + 1` — recorded as `train_sga_ids`. The remaining 12,234 are
   the holdout used for prediction.

It is **one step, not an iteration or a partition.** The trapezoid cuts are
frozen for the mock analysis (decision #2), so the post-cut population is fully
determined before we choose — the target is reachable exactly. A
fraction-of-the-file scheme cannot hit it: the post-cut count would fall out of
the fraction rather than being chosen.
The earlier slice mechanism has been **removed entirely** — there is no
`n_subsets`, `subset_index`, or `slice_sga_ids` anywhere in the pipeline.

If a file has **fewer** cut-passing galaxies than the target, step 4 exits
non-zero naming the shortfall, so `batch_submit.sh` skips that file rather than
producing a run whose size is silently unmatched.

`explore_residuals.py` and `color_predict.py` both restrict their
posterior-predictive computation to `subset_sga_ids` via `_subset_mask()`
(`color_predict.py`, the single source of truth — don't reimplement it inline).
`MU_TF`/`MU_ERR`/`LOGDIST` are therefore `NaN` outside the draw;
`MAIN`/`ANALYSIS` are already draw-scoped, and `combine_color_xonly.py` reads
only `MAIN=True` rows, so nothing downstream depends on the discarded values.
Both gates key on `subset_sga_ids` being present, so DR2 runs — which never set
`target_main_count` — are untouched and continue to use their full sample.

> **The contrast you give up.** Because the draw is taken from cut-passing
> galaxies only, `explore_residuals.py`'s full-sample-vs-MAIN comparison is
> degenerate for mock runs: both curves are the same 17,234 galaxies. That is a
> direct consequence of analysing MAIN only.

### Reproducibility

Three layers, in order of authority:

1. **The explicit ID list is authoritative.** `subset_sga_ids` and
   `train_sga_ids` are written into `output/<run>/input.json`, so any run's
   sample can be reconstructed by reading them — no RNG replay needed.
2. **Provenance for regeneration.** `input.json` also records
   `target_main_count`, `random_seed` and `N_after_cuts`; the FITS path comes
   from `config.json`, which also records `target_main_count`. Between them that
   is everything needed to recompute the draw from scratch.

   > Numeric only, deliberately. CmdStan rejects a data file containing any
   > string value ("Variable: …, error: string values not allowed"), so an
   > earlier revision that recorded `fits_file` in `input.json` broke step 6 for
   > every `target_main_count` run. Step 4 now refuses to write a data file
   > containing a string, naming the offending key. Record text in
   > `config.json`.
3. **A drift check.** NumPy does not promise `Generator` stream stability
   across versions, so verify rather than assume:

   ```bash
   python3 desi_data.py --verify_subset output/c000_ph000_r001
   ```

   It regenerates the draw through `process_desi_tf_data` itself (not a
   reimplementation) and reports `MATCH` or `MISMATCH` per ID list, exiting 1
   on drift. Measured for the environments in play — local numpy 2.4.3 and
   NERSC 2.4.6 produce identical `subset_sga_ids` and `train_sga_ids`.

### Running one file end-to-end (manual, NERSC / SLURM)

```bash
export CONFIG=configs/batch_v0.5.7/c000_ph000_r001.json

sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
# After step4 done (writes init_MAP.json directly, since the config sets
# "fixed_init" -- no step5d needed):
bash slurm/step6_submit.sh $CONFIG
```

### Running locally without SLURM

**`run_batch_local.sh` is the no-scheduler counterpart to
`slurm/batch_submit.sh`.** It takes a config directory or a single config and
runs the identical chain — step4 → step6 (N chains in parallel) → step7 → step8,
with step5d skipped when the config sets `fixed_init` — via the CPU `./2color`
binary, with the chains as background processes instead of one per GPU.

```bash
bash run_batch_local.sh configs/abacus_2color.json          # one mock
bash run_batch_local.sh configs/batch_v0.5.7                # a whole directory
bash run_batch_local.sh --fits-dir /path/to/mock_fits       # generate configs first
bash run_batch_local.sh configs/abacus_2color.json --debug  # ~25 s plumbing test
```

| Flag | Meaning |
|---|---|
| `--fits-dir DIR` | run `make_batch_configs.py` on DIR first, then run the configs |
| `--base` / `--outdir` | base config and output dir for `--fits-dir` |
| `--chains N` | chains per run, in parallel (default 4) |
| `--jobs N` | runs processed concurrently (default 1) |
| `--from-step S` | clear sentinels from S onward and re-run (`4 5d 6 7 8`) |
| `--warmup` / `--samples` / `--max-depth` / `--delta` | sampler overrides |
| `--debug` | `step6_node.sh`'s DEBUG branch + `--no-cov`; **not science-grade** |
| `--force` | re-run even where `.step8_done` exists |

**Its sampler arguments are identical to `slurm/step6_node.sh`'s**, token for
token, in both the production and debug branches — `num_warmup=1000`,
`num_samples=1000`, `adapt delta=0.9 save_metric=1`, `algorithm=hmc engine=nuts
max_depth=10`, `metric=dense_e`, identity start. That parity is the whole point:
change one, change the other, or mock-derived uncertainties stop calibrating the
real measurement.

Sentinels use the same names as `slurm/check_status.sh`, so both status scripts
work on local runs unmodified:

```bash
bash slurm/check_status.sh configs/abacus_2color.json
bash slurm/batch_status.sh configs/batch_local
```

Per-step logs land in `output/<run>/local_step*.log`, and one row per run in
`batch/local_tracker.csv` (the analogue of the batch's `batch/job_tracker.csv`).
Unlike the batch, it also rebuilds `./2color` when `2color.stan` is newer —
mtime-based, so an edited model is actually picked up.

**Cost.** ~6.4 h/chain on CPU at 17,234 galaxies (~4.4 h of it warmup), 4 chains
in parallel, so roughly 6–7 h per mock plus ~5–10 min for step 8. `--jobs 2`
doubles throughput on 8 cores, but each concurrent step 8 holds a ~1.19 GB
covariance plus ~0.55 GB of temporaries.

The local validation config is `configs/abacus_2color.json` itself: it points at
`data/TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits` and carries
both `target_main_count: 17234` and `fixed_init`, so a local run exercises the
same frozen-init path the batch uses. (Two predecessors were deleted:
`configs/abacus_zsnap020_zmax011_2color_test.json`, which omitted `fixed_init`
and so validated the step5d MAP path instead of the batch's, and
`run_subsets.sh` with `configs/abacus_subsets/`, which still carried the removed
`n_subsets`/`subset_index` keys and no `target_main_count`.)

### Predictions (plots only, no covariance)

For local runs where the covariance matrix would OOM:

```bash
python color_predict.py --run-dir output/c000_ph000_r001 --model 2color --no-cov --no-catalog
```

This produces diagnostic plots (`redshift_color_xonly.png`, etc.) without
computing the O(G²) covariance matrix. x-only is the unconditional default
(pass `--full` for the full z/g-conditioned model's plots too). Predictions are
made on the 12,234 holdout galaxies within the draw.

### Output structure

Each file produces one full independent run:

```
output/c000_ph000_r001/
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

### Sample sizes

For the reference file
(`TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits`, 170,781 valid
rows, `haty_min=-21.6`/`haty_max=-18.4`, `z_obs` ∈ [0.01, 0.065]):

| Stage | Count |
|-------|-------|
| valid rows | 170,781 |
| of those, `MAIN` (see #4a) | 154,976 |
| pass the selection cuts (`N_after_cuts`) | 90,119 |
| **drawn subsample = MAIN** | **17,234** |
| training (`n_objects`) | 5,000 |
| holdout / ANALYSIS | 12,234 |

These are what step4 actually reports; the log lines to compare against are
`MAIN filter (source=fullmocks): 154976 of 170781 valid rows are MAIN (15805
non-MAIN dropped)`, `Subsample: 17234 of 90119 cut-passing galaxies
(target_main_count=17234, random_seed=42)` and `Training: 5000, holdout: 12234`. The 17,234/5,000/12,234
split matches `DESI-DR2_TF_pv_cat_v5b.fits` exactly (spiral 13,569 + irregular
3,665 = 17,234; ANALYSIS 8,569 + 3,665 = 12,234).

`N_after_cuts` varies from file to file, but 17,234 does not — that is the point
of the draw.

This file is also the **reference/validation mock** used for Step 0 and for the
numbers quoted throughout this document. Note its dust keyword is the older
`DSTCFF_R_ERR` (0.20456262); production mocks are expected to carry `A_R_ERR`
instead (decision #2c).
