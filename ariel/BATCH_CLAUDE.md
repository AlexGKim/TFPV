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

**Key constraint:** The model has condition number ~2.7M, so a pre-computed mass
matrix (`metric.json`) is essential. Without it, stepsize ~0.002 and every step
hits max treedepth (10). With it, stepsize ~0.08 and treedepth 4–6.

---

## Step Map

| Step | Script | Inputs | Outputs | Time | Queue |
|------|--------|--------|---------|------|-------|
| 4 | `slurm/step4_data.sh` | config JSON, FITS file | `input.json`, `init.json` | <5 min | debug |
| 5d | `slurm/step5d_map.sh` | `input.json`, `init.json` | `optimize.csv`, `init_MAP.json` | ~5 min | debug GPU |
| 5e | `slurm/step5e_metric.sh` | `input.json`, `init_MAP.json` | `2color_metric_build.csv`, `metric.json` | ~7 h | regular GPU |
| 6 | `slurm/step6_chain.sh` (×4) | `input.json`, `init_MAP.json`, `metric.json` | `2color_{1..4}.csv`, `2color_metric_{1..4}.json` | ~14 h each | regular GPU |
| 7 | `slurm/step7_diagnose.sh` | `2color_?.csv` | `stansummary.txt`, `diagnose.txt`, `2color.png` | ~15 min | debug CPU |
| 8 | `slurm/step8_predict.sh` | config, `input.json`, `2color_?.csv` | `color_catalog.fits`, `color_cov.fits` | 1–4 h | regular CPU |

**Sentinel files:** Each step writes `output/$RUN/.step<N>_done` on success.
Check completion with: `bash slurm/check_status.sh $CONFIG`

---

## File Dependency Graph

```
FITS catalog
    └── step4 (desi_data.py)
            ├── input.json
            └── init.json
                    └── step5d (2color_g optimize)
                            ├── optimize.csv
                            └── init_MAP.json
                                    └── step5e (2color_g sample, 1 chain)
                                            ├── 2color_metric_build.csv
                                            └── metric.json ──────────────┐
                                                                           │
                                            ┌──────────────────────────────┘
                                            └── step6 ×4 (2color_g sample, 1 chain each)
                                                    ├── 2color_1.csv
                                                    ├── 2color_2.csv
                                                    ├── 2color_3.csv
                                                    └── 2color_4.csv
                                                            └── step7 (stansummary, corner.py)
                                                                    └── step8 (color_predict.py)
```

`metric.json` is reusable **within the same data type** — copy it to a new run
directory to skip step5e. It is **not** transferable across data types (e.g.
the DR1 metric must not be used for AbacusSummit mocks; see failure mode below).

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

### Step 5d: `init_MAP.json` looks wrong

**Symptom:** `init_MAP.json` exists but has NaN values or all zeros.

**Diagnosis:** Check `output/$RUN/optimize.csv` — if it has only one data row or
all `lp__` values are identical, the optimizer did not converge.

**Fix:** Re-run step5d. If it consistently fails, check `input.json` for data
issues (e.g. N_total = 0, extreme y_min/y_max). Run `python desi_data.py` locally
and inspect `output/$RUN/data.png`.

---

### Step 5e / Step 6: MCMC chain produces no samples

**Symptom:** CSV file exists but has only comment lines (no data rows).

**Diagnosis:** Check SLURM log (`slurm/logs/step5e_*.out`). Common cause:
- Stan segfault: `input.json` has wrong array lengths (check `N_total`)
- Time limit exceeded: extend `#SBATCH -t`
- `init_MAP.json` has out-of-range values: re-run step5d

---

### Step 6: Metric incompatible with data type (cholesky failures + timeout)

**Symptom:** Repeated `cholesky_decompose: Matrix m is not positive definite`
warnings in the step6 SLURM log. `stepsize__` in the CSV is ~0.002 (vs. ~0.08
with a good metric). Each iteration takes ~50s. The chain does not reach 1000
samples within the 18h time limit.

**Diagnosis:** The `metric.json` in `output/$RUN/` was built from a different
data type (e.g. DR1 metric copied to a mock run). The posterior geometry differs
enough that the mass matrix is invalid for the new data.

**Fix:** Build a metric from this data type via step5e:
```bash
sbatch --export=CONFIG=$CONFIG slurm/step5e_metric.sh   # ~7h
```
Then resubmit the step6 chains. For subsequent files of the same data type, copy
the new metric instead of re-running step5e.

---

### Step 6: High divergences or max treedepth

**Symptom:** `diagnose.txt` reports many divergences or `% transitions hitting
max treedepth` > 20%.

**Diagnosis:** Check whether `metric.json` was used. If `stepsize__` in the CSV
is near 0.002, the metric was not applied or is incompatible (see failure mode
above).

**Fix:** Confirm `metric.json` exists in `output/$RUN/` and was built from the
same data type as this run. If missing, run step5e. Do not copy the DR1 metric
to a mock run (or vice versa).

---

### Step 7: `stansummary` R̂ > 1.01

**Symptom:** One or more parameters have `R-hat > 1.01` in `stansummary.txt`.

**Diagnosis:** Chains have not converged. Check if all 4 chain CSVs are present
and have 1000 samples each:
```bash
grep -c "^[^#]" output/$RUN/2color_?.csv
```
Expected output: `1000` for each file (plus 1 for the header).

**Fix:** If chains are truncated (time limit), re-run step6 for missing chains.
If all 4 chains completed but R̂ is still high, increase `num_warmup` in
`step6_chain.sh` (edit `num_warmup=250` → `num_warmup=500`).

---

### Step 8: Memory error in `color_predict.py`

**Symptom:** Python OOM error during covariance matrix computation.

**Fix:** Run without covariance first:
```bash
sbatch --export=CONFIG=$CONFIG slurm/step8_predict.sh
# or manually:
python color_predict.py --config $CONFIG --model 2color --xonly --no-cov
```

---

### Step 7: `corner.py` fails with `ModuleNotFoundError: No module named 'chainconsumer'`

**Symptom:** `step7_diagnose.sh` (which runs `corner.py` and `explore_residuals.py`
right after `stansummary`/`diagnose`) dies with this traceback, and `set -e`
aborts the rest of the script — no `2color.png`, and (if this happens inside
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
config and must be regenerated (step4 → step5d → step6) before trusting step 8.

**Fix:** Always re-run step4 after editing a config, even if you think only an
unrelated field changed:
```bash
sbatch --export=CONFIG=$CONFIG slurm/step4_data.sh
```

---

## Key Parameters to Check After Step 7

From `stansummary.txt` or corner plot `2color.png`:

| Parameter | Expected (DR1_v6) | Meaning |
|-----------|-------------------|---------|
| `slope` | ~−8 | TFR slope |
| `gamma` | −0.70 ± 0.20 | r–z luminosity–color slope |
| `gamma_g` | −1.1 ± 0.05 | g–r luminosity–color slope |
| `alpha_kcorr_r` | ~−5.7 | r-band k-correction slope |
| `alpha_kcorr_z` | ~−5.3 | z-band k-correction slope |
| `alpha_kcorr_g` | ~−6.3 | g-band k-correction slope |

Good convergence: R̂ < 1.01 for all parameters, ESS > 100 per chain.

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `make_map_init.py --run $RUN` | Parse `optimize.csv` → `init_MAP.json` |
| `make_metric.py --run $RUN` | Compute covariance from short CSV → `metric.json` |
| `slurm/check_status.sh $CONFIG` | Show which steps are done/missing |
| `slurm/step6_submit.sh $CONFIG` | Submit 4 chains + auto-chain steps 7 and 8 |

---

## Re-run Reference

```bash
# Any single step:
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step4_data.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5d_map.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step5e_metric.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json,CHAIN_ID=1 slurm/step6_chain.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step7_diagnose.sh
sbatch --export=CONFIG=configs/dr1_v6_2color.json slurm/step8_predict.sh

# All 4 chains + auto-chain steps 7 and 8:
bash slurm/step6_submit.sh configs/dr1_v6_2color.json

# Clear a sentinel to force re-run:
rm output/DR1_v6_2color/.step6_chain2_done
```
