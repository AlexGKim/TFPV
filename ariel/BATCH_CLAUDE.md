# Batch Pipeline Context for Claude

This file provides structured context for a Claude session on NERSC to understand
the 2COLOR batch pipeline, debug failures, and assist with re-runs — without
needing to re-read all source files. It is the **troubleshooting layer**: step
map, dependency graph, and a symptom → diagnosis → fix catalogue.

| You are running | Where | Doc |
|---|---|---|
| AbacusSummit mocks, as a batch | NERSC | [BATCH_MOCKS.md](BATCH_MOCKS.md) — design decisions, config generation, submission. **Start there.** |
| the mechanics of one run's steps | NERSC | [BATCH_NERSC.md](BATCH_NERSC.md) |
| diagnosing a failure | NERSC | **this file** |
| real DR2 data, whole catalog | locally | [DR2_SINGLE.md](DR2_SINGLE.md) |
| real DR2 data, split by morphology | locally | [DR2_TWOPOP.md](DR2_TWOPOP.md) |

---

## Pipeline Overview

The pipeline fits a two-color Tully-Fisher Relation (TFR) model to DESI galaxy
data using Stan (CmdStan). Of its numbered steps, only **4, 6, 7, 8** run on
NERSC in the current chain — step 5d is skipped whenever a config sets
`fixed_init`, and there is no step 5e (the metric-building scripts were
removed).

**Model:** `2color.stan` — quadrivariate (x̂, ŷ, ẑ, ĝ) TFR with two independent
latent color factors (r–z and g–r), and a **rank-1** intrinsic covariance
`S = w wᵀ` over (y,z,g). **13 sampling parameters**: `slope_std`,
`intercept_std`, `sigma_int_x`, `w[3]`, `delta_c`, `mu_c`, `delta_g`, `mu_g`,
and three `alpha_kcorr_*`. (`make_pf_metric.py` accordingly builds a 13×13
dense metric.)

**Key constraint:** The model has condition number ~2.7M. `step6_node.sh`/
`step6_chain.sh` handle this by running from the identity metric with
in-warmup dense-metric adaptation (`metric=dense_e adapt save_metric=1`, no
pre-built `metric_file=`) and `NUM_WARMUP=1000`/`MAX_DEPTH=10` (both
env-overridable). There is **no pre-built-metric step in the batch** — the
scripts that built one (`step5e_metric.sh`, `make_metric.py`) have been
deleted. A local A/B test found that approach ~2.7x slower overall to obtain
than letting step6 adapt from scratch, with no quality gain, and its builder
was a crude 100-draw `np.cov` over a stale parameter list. Warmup from
identity suffices for the rank-1 model: the validated abacus run finished
warmup in ~4.4 h/chain on CPU without stalling at max treedepth. The local
real-data workflow (`run_dr2_onepop.sh`) runs from identity with the same
sampler arguments, so both paths sample identically — DR2_TWOPOP.md Step 6.

---

## Step Map

| Step | Script | Inputs | Outputs | Time | Queue |
|------|--------|--------|---------|------|-------|
| 4 | `slurm/step4_data.sh` | config JSON, FITS file | `input.json`, `init.json`, and (if config sets `fixed_init`) `init_MAP.json` directly | <5 min | debug |
| 5d *(skipped if config sets `fixed_init`)* | `slurm/step5d_map.sh` | `input.json`, `init.json` | `optimize.csv`, `init_MAP.json` | ~5 min | debug GPU |
| 6 | `slurm/step6_node.sh` (1 node, 4 GPUs, 4 chains, identity metric + adapt) | `input.json`, `init_MAP.json` | `2color_{1..4}.csv`, `2color_metric_{1..4}.json` (each chain's own adapted metric, `save_metric=1`) | dataset-dependent (`NUM_WARMUP=1000`/`MAX_DEPTH=10` defaults — re-measure, don't assume a fixed figure) | regular GPU |
| 7 | `slurm/step7_diagnose.sh` | `2color_?.csv` | `stansummary.txt`, `diagnose.txt`, `2color.png` | ~15 min | debug CPU |
| 8 | `slurm/step8_predict.sh` (`color_predict.py`, then `explore_residuals.py`) | config, `input.json`, `2color_?.csv` | `color_xonly_catalog.fits`, `color_xonly_cov.h5` (x-only default; `--full` adds `color_catalog.fits`/`color_cov.h5`), `explore_residuals/` | ~5–10 min per run (mostly the O(G²) covariance; ~1.19 GB `.h5` each) | debug CPU |

**Sentinel files:** Each step writes `output/$RUN/.step<N>_done` on success.
Check completion with: `bash slurm/check_status.sh $CONFIG`

---

## File Dependency Graph

```
FITS catalog
    └── step4 (desi_data.py)
            ├── input.json
            ├── init.json
            └── init_MAP.json  (written directly IF config sets "fixed_init" --
                 |               transforms frozen physical-unit values into
                 |               this run's own standardized coordinates)
                 |
                 |   (config WITHOUT "fixed_init" instead goes:
                 |    init.json -> step5d (2color_g optimize) -> optimize.csv
                 |    -> init_MAP.json)
                 |
                 └── step6 (1 node, 4 GPUs, 4 chains via CUDA_VISIBLE_DEVICES,
                     identity metric + in-warmup adapt -- no metric file)
                             ├── 2color_1.csv
                             ├── 2color_2.csv
                             ├── 2color_3.csv
                             └── 2color_4.csv
                                     └── step7 (stansummary, corner.py)
                                             └── step8 (color_predict.py,
                                                 then explore_residuals.py)
```

There is no metric side branch: the batch has no metric-building step at all
(see BATCH_MOCKS.md decision #3). Each chain saves its own adapted metric as
`2color_{1..4}_metric.json` via `save_metric=1`, for inspection only —
nothing reads it back.

---

## Config File

Referred to as `$CONFIG` throughout. Mock-batch configs are generated into
`configs/batch_*/` by `make_batch_configs.py`; the reference validation config
is `configs/abacus_2color.json`, which also runs locally via
`run_batch_local.sh`.

Key fields:
- `"run"`: output directory name → `output/<run>/`
- `"fits_file"`: input FITS path (relative to ariel/, or absolute for CFS mocks)
- `"exe"`: Stan binary name (base; GPU variant is `2color_g`)
- `"haty_min"`, `"haty_max"`: magnitude selection window
- `"z_obs_min"`, `"z_obs_max"`: redshift window
- `"slope_plane"`, `"intercept_plane"`, `"intercept_plane2"`: oblique cut geometry
- `"fixed_init"`: path to frozen physical-unit init values. **When present,
  step 4 writes `init_MAP.json` itself and step 5d is not run.**
- `"source"`: `"fullmocks"` or `"DESI"`. **Load-bearing for mocks:** step 4
  restricts to `MAIN` rows when it is `"fullmocks"`, because the frozen cuts were
  derived from MAIN rows only. Gated on `source`, not on the column existing —
  the DR2 per-population files carry their own pipeline-written `MAIN`. See
  `BATCH_MOCKS.md` decision #4a.
- `"target_main_count"`: mock batch only. Step 4 applies the frozen cuts to the
  MAIN rows, then draws exactly this many cut-passing galaxies as the analysed
  sample (MAIN in the output). `17234`, matching `DESI-DR2_TF_pv_cat_v5b.fits`.
  There is no slice partitioning — no `n_subsets`/`subset_index`/`slice_sga_ids`
  anywhere.
- `"n_objects"`: training sample size within the drawn subsample (5000)
- `"random_seed"`: 42 — fixes both the subsample draw and the training draw
- `"model"`: now actually honoured by `color_predict.py` (it previously had a hard
  `"color"` default that shadowed this key). Precedence: explicit `--model` flag >
  this key > `"color"`.
- `"dust_pickle"`: DR2 only. Mocks instead carry `d_err_r` in the FITS header
  (`A_R_ERR` / `DSTCFF_R_ERR`), read per file. `make_batch_configs.py` refuses to
  emit configs for a mock file carrying neither keyword, so the iron-value
  fallback is unreachable for a generated batch.

All pipeline scripts accept `--config $CONFIG` or `--run $RUN`.

---

## Common Failure Modes

### Compilation failure: missing libOpenCL

**Symptom:** `make` exits with `cannot find -lOpenCL`

**Fix:**
```bash
export LIBRARY_PATH=$LIBRARY_PATH:${CUDATOOLKIT_HOME}/lib64
```
This must be set before `make`. It is already in `compile_2color_gpu.sh` and all
GPU SLURM scripts, but must also be set if compiling manually on the login node.

---

### Step 5d / step4: `init_MAP.json` looks wrong

**For a config WITHOUT `fixed_init`** (still uses step5d):

**Symptom:** `init_MAP.json` exists but has NaN values or all zeros.

**Diagnosis:** Check `output/$RUN/optimize.csv` — if it has only one data row or
all `lp__` values are identical, the optimizer did not converge.

**Fix:** Re-run step5d. If it consistently fails, check `input.json` for data
issues (e.g. N_total = 0, extreme y_min/y_max). Run `python desi_data.py` locally
and inspect `output/$RUN/data.png`.

**For a config WITH `fixed_init`** (step4 writes `init_MAP.json` directly, no
step5d in the chain): this failure mode should be structurally impossible —
`2color.stan`'s only data-dependent bounds (`slope_std`/`intercept_std`)
reduce to fixed physical-unit ranges (`slope_orig ∈ [-9,-4]`,
`intercept_orig ∈ [-24,-14]`) independent of any run's own `mean_x`/`sd_x`,
so a valid `fixed_init` file is bound-safe for every run by construction. If
`init_MAP.json` still looks wrong, check instead: (1) `configs/fixed_init_2color.json`
was read correctly (see the `fixed_init: ... -> slope_std=... intercept_std=...`
line step4 prints to its log), (2) `mean_x`/`sd_x` in `output/$RUN/init_MAP.json`
came out sane for this run's actual data (not NaN/zero from an empty selection),
and (3) `configs/fixed_init_2color.json` itself hasn't been edited to contain a
value outside `2color.stan`'s bounds for some *other* parameter that isn't
data-dependent but is still constrained (currently none are, per `2color.stan`'s
`parameters` block — re-verify if the model changes).

---

### Step 6: MCMC chain produces no samples

**Symptom:** CSV file exists but has only comment lines (no data rows).

**Diagnosis:** Check SLURM log (`slurm/logs/step6_node_*.out`). Common cause:
- Stan segfault: `input.json` has wrong array lengths (check `N_total`)
- Time limit exceeded: extend `#SBATCH -t`
- `init_MAP.json` has out-of-range values: re-run step5d (config without
  `fixed_init`) or see "Step 5d / step4: `init_MAP.json` looks wrong" above
  (config with `fixed_init` — this should not normally happen)

---

### Step 6: High divergences or max treedepth

**Symptom:** `diagnose.txt` reports many divergences or `% transitions hitting
max treedepth` > 20%. `stepsize__` in the CSV stays tiny (~0.002) instead of
adapting to something reasonable (~0.05-0.1), or each iteration takes ~50s.

**Diagnosis:** `step6_node.sh`/`step6_chain.sh` run from the identity metric
with in-warmup adaptation (no pre-built `metric.json` — that path is
optional/unused, see Pipeline Overview). A tiny stepsize / heavy treedepth
usually means either `NUM_WARMUP` is too short for the adaptation to
converge, or the posterior geometry for this dataset is unusually
ill-conditioned even for depth 10.

**Fix:** First try increasing `NUM_WARMUP` (default 1000):
```bash
sbatch --export=CONFIG=$CONFIG,NUM_WARMUP=2000 slurm/step6_node.sh
```
If treedepth saturation (not divergences) dominates, `MAX_DEPTH` (default 10)
is already at Stan's own ceiling — the model's near-singular free-covariance
posterior is genuinely expensive there; consider whether the dataset's
selection cuts or training-sample size are unusually degenerate for this
run. Pre-building a metric is not an option here (those scripts were
removed) and would not be expected to help — see
Pipeline Overview) and an empirical A/B test found it slower overall with no
quality gain.

---

### Step 7: `stansummary` R̂ > 1.01

**Symptom:** One or more parameters have `R-hat > 1.01` in `stansummary.txt`.

**Diagnosis:** Chains have not converged. Check if all 4 chain CSVs are present
and have 1000 samples each:
```bash
grep -c "^[^#]" output/$RUN/2color_?.csv
```
Expected output: `1001` for each file — 1000 draws plus the CSV header row.

**Fix:** If chains are truncated (time limit), re-run step6 for missing chains
(`sbatch --export=CONFIG=$CONFIG slurm/step6_node.sh` for all 4, or
`slurm/step6_chain.sh` with `CHAIN_ID=N` for just one). If all 4 chains
completed but R̂ is still high, increase `num_warmup` (edit
`slurm/step6_node.sh`'s default, or pass e.g. `NUM_WARMUP=2000` via
`--export`). Note the default is already **1000**, so `NUM_WARMUP=500` would
be a decrease — that suggestion dates from the old 250 default.

---

### Step 8: Memory error in `color_predict.py`

**Symptom:** Python OOM error (or SIGKILL / exit 137) during the posterior
predictive or covariance computation.

**Diagnosis:** Check the galaxy count the run is working over. Both the
O(draws × galaxies) prediction and the O(G²) covariance must be **scoped to the
run's drawn subsample**. If a mock config lacks `target_main_count`, nothing is
scoped: G becomes the whole file's cut-passing count (≈91,000 for the reference
mock → a ~67 GB dense matrix). `color_predict._subset_mask()` does the scoping,
keyed on `subset_sga_ids` in `input.json` — if that key is absent from a *mock*
run's `input.json`, step 4 ran without `target_main_count` and must be re-run.
(For DR2 runs the key is legitimately absent: they use their full sample, which
is small enough.) Confirm with:
```bash
python3 -c "import json;d=json.load(open('output/$RUN/input.json'));print(len(d['subset_sga_ids']))"
```
It should print `17234`.

**Fix:** Confirm the config carries `target_main_count`, re-run step 4, then
step 8. To skip the covariance entirely:
```bash
python color_predict.py --config $CONFIG --model 2color --no-cov
```

---

### Step 8: covariance built with the wrong dust value

**Symptom:** No error — the run completes and the covariance looks fine, but its
dust off-diagonal term is wrong. This is the silent failure
`color_predict.resolve_d_err_r()` exists to prevent.

**Diagnosis:** Step 8 logs which source supplied `d_err_r`:
```bash
grep -E "Loaded d_err_r|WARNING: no dust_pickle" slurm/logs/step8_predict_*.out
```
- `Loaded d_err_r = … mag from FITS header A_R_ERR of …` — correct for mocks.
- `Loaded d_err_r = … mag from data/loa_internalDust_…pickle` — correct for DR2.
- `WARNING: no dust_pickle in config and no A_R_ERR/DSTCFF_R_ERR in the FITS
  header …` — **fell through to the built-in iron default 0.17680325, which is
  wrong for both mocks and DR2.** Getting it wrong by the DR2 margin
  (0.1768 vs 0.2173) scales the dust variance by ~1.5.

For a **generated** batch this should now be impossible: `make_batch_configs.py`
checks every file's header and refuses to emit any config if one carries neither
keyword. Reaching the fallback means the config was hand-written or predates that
gate.

**Fix:** For a mock, confirm the FITS header actually carries `A_R_ERR` or
`DSTCFF_R_ERR` on HDU 1. For DR2, confirm the config sets `dust_pickle` *and*
that it reached `color_predict.py` — it reads `output/$RUN/config.json`, so a
key added to the pipeline config after step 4 last ran needs either a step-4
re-run or the `--config` overlay (which logs `config overlay: dust_pickle = …`).
Then re-run step 8.

Step 8 records the value it used as a `d_err_r` attribute on
`color_xonly_cov.h5`. `combine_color_xonly.py` reads that attribute rather than
re-deriving it — re-deriving is how a combined product once ended up with one
dust value in its per-population blocks and another in its cross-population
terms. Inspect it with:
```bash
python3 -c "import h5py,sys; f=h5py.File(sys.argv[1]); print(dict(f.attrs))" output/$RUN/color_xonly_cov.h5
```

---

### Config generation: `make_batch_configs.py` exits without writing anything

**Symptom:** `ERROR: N of M files cannot produce a correct step-8 covariance. No
configs written.`, followed by one line per offending file.

**Diagnosis:** Deliberate, and cheap — it costs seconds here instead of a step-6
GPU allocation followed by a quietly wrong (or failed) step 8. Three reasons:
- `no dust error keyword (need one of A_R_ERR, DSTCFF_R_ERR)` — step 8 would fall
  back to the iron 0.17680 (see the dust failure mode above).
- `no PHOTSYS_ERR column and PHOTSYS is numeric (TFORM='D')` — step 8's
  `_systematic_offdiag_terms` would raise, since a float `PHOTSYS` cannot match
  `'N'`. See `BATCH_MOCKS.md` decision #2d.
- `has no "target_main_count"` (on the *base* config) — every run would analyse
  the whole file's cut-passing population, G ~ 91,000, a ~67 GB covariance.

The check is all-or-nothing on purpose: a partial batch is harder to notice than
no batch.

**Fix:** Fix the input files, or narrow `--pattern` to exclude them. Inspect a
header with:
```bash
python3 -c "from astropy.io import fits;h=fits.open('FILE')[1];print({k:h.header[k] for k in h.header if 'ERR' in k});print([c.name for c in h.columns if 'PHOTSYS' in c.name])"
```

---

### Step 8: `TypeError: PHOTSYS has non-string dtype … and no PHOTSYS_ERR`

**Symptom:** Step 8 raises from `_systematic_offdiag_terms`.

**Diagnosis:** Working as intended — this replaced a silent failure. The mock
catalogs store `PHOTSYS` as a numeric offset, and `np.where(photsys == 'N', …)`
against a float array is elementwise-False *without raising*, so the 0.02
photsys calibration systematic used to vanish from every mock covariance with no
warning. The value lives in the `PHOTSYS_ERR` column; if a catalog has neither a
usable `PHOTSYS_ERR` nor a string `PHOTSYS`, the term is unresolvable and the
covariance would be wrong.

**Fix:** Confirm the catalog has `PHOTSYS_ERR`. Every mock this pipeline targets
does, which is why `make_batch_configs.py` now checks for it up front. Any mock
covariance built before this change is missing a rank-1 block over ~30% of
galaxies and must be regenerated.

---

### Step 7: `corner.py` fails with `ModuleNotFoundError: No module named 'chainconsumer'`

**Symptom:** `step7_diagnose.sh` (which runs `corner.py` right after
`stansummary`/`diagnose`) dies with this traceback, and `set -e` aborts the rest
of the script — no `2color.png`, and (if this happens inside
`step6_submit.sh`'s auto-chained step7→step8) step8 never runs either, even
though the MCMC chains themselves finished cleanly.

**Diagnosis:** `chainconsumer` isn't installed in whatever environment invoked
`corner.py`. Confirm with `python3 -c "import chainconsumer"` inside the same
conda env `step7_diagnose.sh` activates
(`/global/cfs/projectdirs/desi/users/akim/conda/envs/TFPV`).

**Fix:** `pip install chainconsumer` into that env, then re-run step 7 (and step
8 if it got skipped):
```bash
sbatch --export=CONFIG=$CONFIG slurm/step7_diagnose.sh
sbatch --export=CONFIG=$CONFIG slurm/step8_predict.sh
```

---

### Step 4: silently stale `input.json` after editing a config

**Symptom:** Chains sample fine and everything looks normal, but predictions
don't match the selection/holdout you expect — e.g.
`n_objects`/`target_main_count` were changed in a config but the run's actual
training rows didn't change.

**Diagnosis:** `desi_data.py` now warns when it's about to overwrite an
`output/$RUN/input.json` whose partition metadata (`target_main_count`,
`n_objects`, `random_seed`, `train_fraction`) differs from the config currently
being applied —
check `slurm/logs/step4_data_*.out` for a `WARNING: overwriting ... whose
partition metadata differs` line. If step 4 was never re-run after a config
edit, `input.json`/`init_MAP.json`/the MCMC chains are all still fit to the old
config and must be regenerated (step4 → step6, or step4 → step5d → step6 for
a config without `fixed_init`) before trusting step 8.

**Fix:** Always re-run step4 after editing a config, even if you think only an
unrelated field changed:
```bash
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
```

**Note — this warning cannot catch a sampling-*code* change.** It compares
config metadata only. The mock sampling scheme was replaced: slice partitioning
gave way to a single fixed-size draw of `target_main_count` cut-passing galaxies
(see `BATCH_MOCKS.md` "The Subsample Draw"). A run dir predating that change is
recognizable by an `n_subsets`/`subset_index`/`slice_sga_ids` key in its
`input.json` — the current code writes none of those, and `apply_config`
silently *ignores* those keys in an old config, so a stale config produces an
unscoped run with no error. Any such run must be regenerated from step4.

To check that a mock run's recorded draw still regenerates identically (numpy
does not promise RNG stream stability across versions):
```bash
python3 desi_data.py --verify_subset output/$RUN
```

---

## Key Parameters to Check After Step 7

From `stansummary.txt` or corner plot `2color.png`:

| Parameter | Meaning |
|-----------|---------|
| `slope` | TFR slope (physical units; `slope_std` is its standardized form) |
| `sigma_int_x` | Intrinsic scatter in x |
| `w.1/.2/.3` | Rank-1 intrinsic-scatter loading over (y,z,g); `S = w wᵀ` |
| `w_norm` | \|w\| — magnitude of the single scatter axis (~0.35 mag) |
| `scatter_angle_deg` | Angle between the scatter direction and the achromatic axis |
| `delta_c`, `delta_g` | Color–velocity slopes (mean structure) |
| `mu_c`, `mu_g` | Mean colors at `x = x_bar` |
| `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` | Band k-correction slopes |

Two earlier parameterizations are gone and their names will not appear:
`gamma`/`gamma_g`/`tau_c`/`tau_g` (the gamma-tau form), and
`Sc_scale.1/.2`/`Sc_Lcorr.2.1`/`n_null.1/.2/.3` (the rank-2 free-null
covariance, replaced by the rank-1 `S = w wᵀ` in `2color.stan`). If you see
those in a `stansummary.txt`, the chains predate the current model. No
"expected" reference values are recorded here since they're dataset-dependent —
compare against a previous successful run of the *same* dataset/config.

Good convergence: R̂ < 1.01 for all parameters, ESS > 100 per chain.

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `make_map_init.py --run $RUN` | Parse `optimize.csv` → `init_MAP.json` |
| `slurm/check_status.sh $CONFIG` | Show which steps are done/missing |
| `slurm/step6_submit.sh $CONFIG` | Submit step6_node.sh (4 chains, 1 node) + auto-chain steps 7 and 8 |

---

## Re-run Reference

```bash
# Any single step:
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
sbatch --export=CONFIG=$CONFIG slurm/step5d_map.sh       # only if config has no fixed_init
sbatch --export=CONFIG=$CONFIG slurm/step6_node.sh       # all 4 chains, 1 node/4 GPUs
sbatch --export=CONFIG=$CONFIG,CHAIN_ID=1 slurm/step6_chain.sh  # just chain 1
sbatch --export=CONFIG=$CONFIG slurm/step7_diagnose.sh
sbatch --export=CONFIG=$CONFIG slurm/step8_predict.sh

# All 4 chains + auto-chain steps 7 and 8:
bash slurm/step6_submit.sh $CONFIG

# Clear a sentinel to force re-run:
rm output/$RUN/.step6_chain2_done
```
