# Batch Pipeline Context for Claude

This file provides structured context for a Claude session on NERSC to understand
the 2COLOR batch pipeline, debug failures, and assist with re-runs — without
needing to re-read all source files.

---

## Pipeline Overview

The pipeline fits a two-color Tully-Fisher Relation (TFR) model to DESI galaxy
data using Stan (CmdStan). It has 8 steps; only steps 4–8 run on NERSC.

**Model:** `2color.stan` — quadrivariate (x̂, ŷ, ẑ, ĝ) TFR with two independent
latent color factors (r–z and g–r). 17 sampling parameters.

**Key constraint:** The model has condition number ~2.7M. `step6_node.sh`/
`step6_chain.sh` handle this by running from the identity metric with
in-warmup dense-metric adaptation (`metric=dense_e adapt save_metric=1`, no
pre-built `metric_file=`) and `NUM_WARMUP=1000`/`MAX_DEPTH=10` (both
env-overridable). A separate pre-built-metric step (5e, below) exists but is
**not** read by step6 and is optional/historical — an empirical local A/B
test found it ~2.7x slower overall to obtain than just letting step6 adapt
from scratch, with no quality gain (its own metric-building method is a
crude 100-draw short-chain `np.cov`, prone to the same fragility).

---

## Step Map

| Step | Script | Inputs | Outputs | Time | Queue |
|------|--------|--------|---------|------|-------|
| 4 | `slurm/step4_data.sh` | config JSON, FITS file | `input.json`, `init.json`, and (if config sets `fixed_init`) `init_MAP.json` directly | <5 min | debug |
| 5d *(skipped if config sets `fixed_init`)* | `slurm/step5d_map.sh` | `input.json`, `init.json` | `optimize.csv`, `init_MAP.json` | ~5 min | debug GPU |
| 5e *(optional, unused by step6)* | `slurm/step5e_metric.sh` | `input.json`, `init_MAP.json` | `2color_metric_build.csv`, `metric.json` | ~7 h | regular GPU |
| 6 | `slurm/step6_node.sh` (1 node, 4 GPUs, 4 chains, identity metric + adapt) | `input.json`, `init_MAP.json` | `2color_{1..4}.csv`, `2color_metric_{1..4}.json` (each chain's own adapted metric, `save_metric=1`) | dataset-dependent (`NUM_WARMUP=1000`/`MAX_DEPTH=10` defaults — re-measure, don't assume a fixed figure) | regular GPU |
| 7 | `slurm/step7_diagnose.sh` | `2color_?.csv` | `stansummary.txt`, `diagnose.txt`, `2color.png` | ~15 min | debug CPU |
| 8 | `slurm/step8_predict.sh` (`color_predict.py`, then `explore_residuals.py`) | config, `input.json`, `2color_?.csv` | `color_xonly_catalog.fits`, `color_xonly_cov.h5` (x-only default; `--full` adds `color_catalog.fits`/`color_cov.h5`), `explore_residuals/` | ~5–10 min per slice-scoped run (mostly the O(G²) covariance; ~1.17 GB `.h5` each) | debug CPU |

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
                     identity metric + in-warmup adapt -- step5e not required)
                             ├── 2color_1.csv
                             ├── 2color_2.csv
                             ├── 2color_3.csv
                             └── 2color_4.csv
                                     └── step7 (stansummary, corner.py)
                                             └── step8 (color_predict.py,
                                                 then explore_residuals.py)
```

Step 5e (`2color_g sample`, 1 chain → `metric.json`) exists as an optional
side branch off `init_MAP.json`, not on the critical path — step6 never
reads its output. Historically, when metric-seeding *was* used, a metric was
reusable only **within the same data type** (the DR1 metric must not be used
for AbacusSummit mocks; see failure mode below) — that caveat is now moot
for the standard workflow since step6 doesn't consume `metric.json` at all.

---

## Config File

Location: `configs/dr1_v6_2color.json`

Key fields:
- `"run"`: output directory name → `output/DR1_v6_2color/`
- `"fits_file"`: input FITS path (relative to ariel/)
- `"exe"`: Stan binary name (base; GPU variant is `2color_g`)
- `"haty_min"`, `"haty_max"`: magnitude selection window
- `"z_obs_min"`, `"z_obs_max"`: redshift window


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
run. Building a metric via the optional `step5e_metric.sh` is not expected
to help — its own metric-building method has the same fragility (see
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
Expected output: `1000` for each file (plus 1 for the header).

**Fix:** If chains are truncated (time limit), re-run step6 for missing chains
(`sbatch --export=CONFIG=$CONFIG slurm/step6_node.sh` for all 4, or
`slurm/step6_chain.sh` with `CHAIN_ID=N` for just one). If all 4 chains
completed but R̂ is still high, increase `num_warmup` (edit
`slurm/step6_node.sh`'s default, or pass `NUM_WARMUP=500` via `--export`).

---

### Step 8: Memory error in `color_predict.py`

**Symptom:** Python OOM error during covariance matrix computation.

**Fix:** Run without covariance first:
```bash
sbatch --export=CONFIG=$CONFIG slurm/step8_predict.sh
# or manually:
python color_predict.py --config $CONFIG --model 2color --no-cov
```

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
don't match the selection/holdout you expect — e.g. `n_objects`/`n_subsets`
were changed in a config but the run's actual training rows didn't change.

**Diagnosis:** `desi_data.py` now warns when it's about to overwrite an
`output/$RUN/input.json` whose partition metadata (`n_subsets`, `subset_index`,
`n_objects`, `random_seed`) differs from the config currently being applied —
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

**Note — this warning cannot catch a partition-*code* change.** It compares
config metadata only. The partition point moved from post-selection-cut to
pre-selection-cut (see `BATCH_MOCKS.md` "Subset Partition Mode"), so any
`input.json` written before that change has identical metadata
(`n_subsets`/`subset_index`/`n_objects`/`random_seed`) but *different actual
membership* — and no warning will fire. Any run dir predating the change must
be regenerated from step4, and its `input.json` is recognizable by the absence
of a `slice_sga_ids` key.

---

## Key Parameters to Check After Step 7

From `stansummary.txt` or corner plot `2color.png`:

| Parameter | Meaning |
|-----------|---------|
| `slope` | TFR slope |
| `Sc_scale.1`, `Sc_scale.2` | Rank-2 chromatic intrinsic scatter scales |
| `Sc_Lcorr.2.1` | The single free entry of the 2×2 chromatic correlation Cholesky |
| `n_null.1/.2/.3` | Free null direction of the rank-2 intrinsic covariance |
| `delta_c`, `delta_g` | Color–velocity slopes (mean structure) |
| `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` | Band k-correction slopes |

(`gamma`/`gamma_g`/`tau_c`/`tau_g` from the old gamma-tau parameterization no
longer exist in the current `2color.stan` — see `2color.stan`'s header
comment for the free-null rank-2 covariance model.) No "expected" reference
values are recorded here since they're dataset-dependent — compare against a
previous successful run of the *same* dataset/config, not a fixed table.

Good convergence: R̂ < 1.01 for all parameters, ESS > 100 per chain.

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `make_map_init.py --run $RUN` | Parse `optimize.csv` → `init_MAP.json` |
| `make_metric.py --run $RUN` | Compute covariance from short CSV → `metric.json` |
| `slurm/check_status.sh $CONFIG` | Show which steps are done/missing |
| `slurm/step6_submit.sh $CONFIG` | Submit step6_node.sh (4 chains, 1 node) + auto-chain steps 7 and 8 |

---

## Re-run Reference

```bash
# Any single step:
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step4_data.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5d_map.sh       # only if config has no fixed_init
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5e_metric.sh    # optional, unused by step6
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step6_node.sh       # all 4 chains, 1 node/4 GPUs
sbatch --export=CONFIG=configs/dr1_v6_2color.json,CHAIN_ID=1 slurm/step6_chain.sh  # just chain 1
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step7_diagnose.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step8_predict.sh

# All 4 chains + auto-chain steps 7 and 8:
bash slurm/step6_submit.sh configs/dr1_v6_2color.json

# Clear a sentinel to force re-run:
rm output/DR1_v6_2color/.step6_chain2_done
```
