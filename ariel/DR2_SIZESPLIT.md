# DR2 Run, Two Populations by Size: D26_kpc ≷ 30 kpc (Steps 1–8)

**Which DR2 doc do I want?** All three use the 2color model; they differ in how
the catalog is partitioned:

| Doc | Fits | Input |
|---|---|---|
| **DR2_SIZESPLIT.md** (this file) | disk galaxies split by physical size, fit independently | `DR2_TF_disks_v5_{large,small}.fits` — 15,953 large + 15,415 small |
| [DR2_TWOPOP.md](DR2_TWOPOP.md) | disk galaxies split by morphology, fit independently | `DR2_TF_{spirals,irrs}_v5_*.fits` — 24,813 `Spiral` + 6,555 `Irregular` |
| [DR2_SINGLE.md](DR2_SINGLE.md) | the whole catalog as one population | `SGA-2020_loa_Vrot_VI_v0.fits`, 35,666 galaxies |

This is a **parallel analysis chain**, not a replacement: it adds
`make_size_subsets.py`, `run_size_onepop.sh` and `combine_size_xonly.py`
alongside their morphology counterparts and changes none of them. Steps 1–8 are
the *same commands with the same arguments* as DR2_TWOPOP.md — only the input
files and the run names differ — so the two chains sample the same posterior
with the same algorithm and their results are directly comparable. **Read
[DR2_TWOPOP.md](DR2_TWOPOP.md) for the rationale** behind the sampler settings
(identity metric, rank-1 `S = w wᵀ`, `adapt delta=0.9`, `num_warmup=1000`,
`max_depth=10`) and the `dust_pickle` requirement; it is not repeated here.

---

## Size-population setup

| Population | Predicate | Rows (DR2_TF v5 disks) |
|---|---|---|
| large | `D26_kpc > 30` and not VI-rejected | 15,953 |
| small | `D26_kpc <= 30` and not VI-rejected | 15,415 |

`D26_kpc` is the SGA r-band 26 mag/arcsec² isophotal diameter **in kpc**. The
cut is on that column, not the angular `D26` (arcmin, max ≈ 22 in this catalog —
a threshold of 30 there would select nothing). 30 kpc is essentially the sample
median (30.30 kpc), so the split is close to even by construction.

The parent is the **union of the two DR2_TF v5 morphology files**, so this is a
repartition of exactly the galaxies DR2_TWOPOP.md fits: `data/DR2_TF_spirals_v5.fits`
(25,388) + `data/DR2_TF_irrs_v5.fits` (7,100) = 32,488 rows, of which 31,368
survive "not VI-rejected" (`JOHN_VI.mask == True`, the same predicate the
morphology chain uses) and are split above.

**Interpretation caveat.** Unlike morphology, `D26_kpc` is distance-derived, so
the two populations have different redshift distributions and therefore
different effective selection functions. See `plot_d25_d26_vs_z.py` for the
size-vs-redshift trend this refers to. Keep it in mind when comparing the two
populations' fitted parameters.

The size cut is applied **once, upstream**, by pre-filtering into one FITS file
per population, so the rest of the pipeline is an ordinary single-FITS run and
never needs to know about the split:

```bash
python make_size_subsets.py --input data/DR2_TF_spirals_v5.fits data/DR2_TF_irrs_v5.fits
# writes data/DR2_TF_disks_v5.fits        (the stacked parent)
#        data/DR2_TF_disks_v5_large.fits  (D26_kpc > 30)
#        data/DR2_TF_disks_v5_small.fits  (D26_kpc <= 30)
```

The stacked parent is written out because the run naming below derives from a
parent filename that has to exist on disk. `--force` overwrites existing files,
`--outdir` writes elsewhere, `--threshold`/`--column` change the cut, and
`--parent-stem` renames the parent. The script asserts that the inputs share a
schema, that no `SGA_ID` appears twice, that no kept galaxy has a non-finite
`D26_kpc`, and that the two subsets are disjoint *and* partition the kept
parent — a size split cannot silently lose galaxies.

---

## Setup (per population, env vars)

Run everything below **once per population**:

```bash
export PARENT_FITS=data/DR2_TF_disks_v5.fits
export POPULATION=large                                     # or small
export FITS=${PARENT_FITS%.fits}_${POPULATION}.fits
export RUN=$(basename ${PARENT_FITS%.fits})_2color_${POPULATION}
export CONFIG=configs/$(echo $RUN | tr '[:upper:]' '[:lower:]').json
```

giving `RUN=DR2_TF_disks_v5_2color_large` and
`CONFIG=configs/dr2_tf_disks_v5_2color_large.json`.
[run_size_onepop.sh](run_size_onepop.sh) computes these the same way and
automates Steps 1–8 (pausing only for Step 3's interactive fiducial choice):

```bash
./run_size_onepop.sh --population large --fits data/DR2_TF_disks_v5.fits
./run_size_onepop.sh --population small --fits data/DR2_TF_disks_v5.fits
```

It also runs the split above if the subset file is missing, so the explicit
`make_size_subsets.py` call is optional. Its options match run_dr2_onepop.sh
(`--from-step`, `--chains`, `--warmup`, `--delta`, `--max-depth`, `--run-suffix`),
plus `--size-input PATH` (repeatable) to stack different catalogs into the parent.

---

## Steps 1–8

Identical to [DR2_TWOPOP.md](DR2_TWOPOP.md) Steps 1–8 with the `$FITS`, `$RUN`
and `$CONFIG` above. In brief:

```bash
# Step 1: estimate the core distribution
python selection_ellipse.py --file $FITS --run $RUN --source DESI \
    --z_obs_min 0.01 --z_obs_max 0.065 --haty_min -23 --haty_max -18
open output/$RUN/selection_ellipse.png

# Step 2: MLE fit and pull-profile diagnostic (always the tophat binary)
python select_v2.py --run $RUN --fits_file $FITS --exe ./tophat \
    --z_obs_min 0.01 --z_obs_max 0.065
open output/$RUN/select_v2_pull.png

# Step 3: set fiducial selection criteria — the one interactive step
python set_fiducial.py --run $RUN
open output/$RUN/select_v2_fiducial_pull.png

# Step 3b: export run config (exe=2color, source=DESI, model=2color, n_sigma=3.0)
python export_config.py --run $RUN --out $CONFIG
python -c "
import json
cfg = json.load(open('$CONFIG'))
cfg['fits_file'] = '$FITS'
json.dump(cfg, open('$CONFIG', 'w'), indent=2)
print('fits_file set to', cfg['fits_file'])
"

# Step 4: prepare data
python desi_data.py --config $CONFIG

# Step 5: compile the Stan model (once)
( cd ../../cmdstan && make ../TFPV/ariel/2color )

# Step 5d: MAP estimate
./2color optimize data file=output/$RUN/input.json \
    init=output/$RUN/init.json output file=output/$RUN/optimize.csv
python3 make_map_init.py --run $RUN

# Step 6: MCMC — 4 parallel chains from the identity metric
PIDS=()
for CHAIN_ID in 1 2 3 4; do
    ./2color sample num_warmup=1000 num_samples=1000 \
        adapt delta=0.9 save_metric=1 \
        algorithm=hmc engine=nuts max_depth=10 \
        metric=dense_e id=$CHAIN_ID \
        data file=output/$RUN/input.json \
        init=output/$RUN/init_MAP.json \
        output file=output/$RUN/2color_${CHAIN_ID}.csv &
    PIDS+=($!)
done
for PID in "${PIDS[@]}"; do wait "$PID" || echo "ERROR: a chain failed"; done

# Step 7: diagnose and visualize
../../cmdstan/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt
python corner.py --run $RUN --model 2color

# Step 8: predict absolute magnitudes (x-only; add --full for the full model)
python color_predict.py --config $CONFIG --model 2color
python explore_residuals.py --config $CONFIG --kind 2color
```

Two things about Step 3 and Step 3b that matter here:

- **Choose `haty_min`/`haty_max` from this population's own
  `select_v2_pull.png`.** The large and small samples have different luminosity
  distributions, so DR2_TWOPOP.md's spiral/irregular reference values
  (`-22.0` / `-17.8` / `-17.5`) do not carry over.
- **Add `dust_pickle` to `$CONFIG` by hand** after Step 3b — it is not
  prompted for, and without it step 8 silently falls back to the iron default
  `d_err_r = 0.17680325` instead of the correct loa value `0.21734862`:

  ```json
  "dust_pickle": "data/loa_internalDust_nokcorr_mcmc.pickle"
  ```

  Verify it took effect: `grep "Loaded d_err_r" output/$RUN/step8.log`.
  See DR2_TWOPOP.md Step 3b for the full resolution order.

---

## Step 9: Combine the two populations

Once both populations have finished Step 8, merge their x-only catalogs and
covariances into one product. The two Stan runs are independent, so the
cross-population *statistical* covariance is zero — but the PHOTSYS calibration
floor and the internal-dust slope uncertainty are survey-wide systematics that
do correlate across populations, and this fills those cross-terms in:

```bash
python combine_size_xonly.py \
    --run-a DR2_TF_disks_v5_2color_large  --label-a large \
    --run-b DR2_TF_disks_v5_2color_small  --label-b small \
    --out-run DR2_TF_disks_v5_2color_combined \
    --catalog-name DESI-DR2_TF_pv_cat_v5_size.fits
```

Writes `output/DR2_TF_disks_v5_2color_combined/DESI-DR2_TF_pv_cat_v5_size.fits`
(with a `POPULATION` column) and `..._cov.h5` (datasets `cov`, `analysis`,
`population`, `sga_id`; attributes `d_err_r`, `population_labels`, `row_order`).
It aborts if the two runs resolved different `d_err_r` values — the cross term
is only valid when both share one dust-slope uncertainty.

`combine_size_xonly.py` is [combine_color_xonly.py](combine_color_xonly.py) with
the population labels lifted into arguments; it imports that module's loading and
systematics helpers rather than reimplementing them. Verified to reproduce
`output/DR2_TF_v5_2color_combined/` bit-for-bit when given the spiral/irregular
runs and labels.

---

## File Reference

| File | Purpose |
|------|---------|
| `make_size_subsets.py` | Stacks the input catalogs and splits them on `D26_kpc` |
| [run_size_onepop.sh](run_size_onepop.sh) | Automates Steps 1–8, one population per invocation |
| `combine_size_xonly.py` | Step 9: label-parameterized combine of the two populations |
| [DR2_TWOPOP.md](DR2_TWOPOP.md) | The morphology-split chain: rationale for every sampler setting used above |
| `2color.stan` | Stan model: quadrivariate TFR with two independent color factors |
| `plot_d25_d26_vs_z.py` | The D26-vs-redshift trend behind the interpretation caveat |
| `data/loa_internalDust_nokcorr_mcmc.pickle` | Internal-dust MCMC for DR2; `"dust_pickle"` in `$CONFIG` points here |
