# DR2 Run, One Population: all disk galaxies, unsplit (Steps 1–8)

**Which DR2 doc do I want?** All of these use the 2color model; they differ in
how the catalog is partitioned:

| Doc | Fits | Input |
|---|---|---|
| **DR2_DISKS_SINGLE.md** (this file) | every disk galaxy as one population | `DR2_TF_disks_v5_all.fits` — 31,368 galaxies |
| [DR2_TWOPOP.md](DR2_TWOPOP.md) | the same disks split by morphology, fit independently | `DR2_TF_{spirals,irrs}_v5_*.fits` — 24,813 `Spiral` + 6,555 `Irregular` |
| [DR2_SINGLE.md](DR2_SINGLE.md) | the whole loa catalog as one population | `SGA-2020_loa_Vrot_VI_v0.fits` — 35,666 galaxies, early types included |

This is a **parallel analysis chain**: it adds `make_disk_union.py` and
`run_disks_single.sh` and changes nothing that already exists. Steps 1–8 are
the *same commands with the same arguments* as DR2_TWOPOP.md — only the input
file and the run name differ — so this fit and the two-population fit sample
the same posterior with the same algorithm and can be compared directly.
**Read [DR2_TWOPOP.md](DR2_TWOPOP.md) for the rationale** behind the sampler
settings (identity metric, rank-1 `S = w wᵀ`, `adapt delta=0.9`,
`num_warmup=1000`, `max_depth=10`) and the `dust_pickle` requirement; it is not
repeated here.

Note the distinction from [DR2_SINGLE.md](DR2_SINGLE.md), which also fits one
population: that one fits the full loa catalog, including the ~4,300 early
types and non-galaxy morphologies (`E`, `S0`, `PSF`, `REX`, …). This one fits
exactly the galaxies the two-population run fits, just without splitting them —
so a difference between the two is about the split, not about the sample.

---

## Building the input

```bash
python make_disk_union.py
# writes data/DR2_TF_disks_v5_all.fits  (31,368 rows)
```

Stacks `data/DR2_TF_spirals_v5.fits` (25,388) and `data/DR2_TF_irrs_v5.fits`
(7,100) into 32,488 rows and drops the 1,120 that John's visual inspection
rejected, leaving **31,368**. `--input` takes different catalogs, `--out`
renames the output, `--force` overwrites.

**Why the VI cut lives here.** It is the only cut this script applies, and it
has to be applied upstream because nothing downstream does it: `desi_data.py`
and `color_predict.py` re-derive validity and phase-space cuts, but `JOHN_VI`
is not among them, so a VI-rejected galaxy would otherwise be fit. Dropping
those rows is also what keeps this run comparable to the morphology and size
chains, which drop the same 1,120. Pass `--keep-vi-rejected` to fit all 32,488
rows instead — that is a different sample, and the comparison no longer
isolates the effect of the split.

The stacked file's provenance is in its header: `PARENT` (the inputs),
`POPULATN`, `NPARENT` (32,488), `NROWS` (31,368), `GITSHA`, and a HISTORY line
naming the predicate. The script also asserts the inputs share a schema and
that no `SGA_ID` appears twice — a duplicated galaxy would enter the likelihood
twice.

---

## Setup (env vars)

```bash
export FITS=data/DR2_TF_disks_v5_all.fits
export RUN=DR2_TF_disks_v5_all_2color
export CONFIG=configs/dr2_tf_disks_v5_all_2color.json
```

### Provenance: `DR2_TF_vrot_all_v5.fits` is already here (checked 2026-11-18)

`/global/cfs/cdirs/desi/science/td/pv/tfgalaxies/Y3/DR2_TF_vrot_all_v5.fits`
is **the same catalogue as `data/DR2_TF_disks_v5.fits`**, not a new input. It
does not supersede anything. Verified on Perlmutter with md5 fingerprints of
the SGA_ID-sorted `SGA_ID`, `LOGVROT`, `R_MAG_SB26` and `Z_DESI` columns — all
four identical, 32,488 rows on both sides.

| | rows | state |
|---|---|---|
| `DR2_TF_vrot_all_v5.fits` (CFS) = `data/DR2_TF_disks_v5.fits` | 32,488 | disk-only, **before** VI filtering |
| `data/DR2_TF_disks_v5_all.fits` | 31,368 | the same, **after** dropping 1,120 VI rejects |

What the CFS file contains: `MORPHTYPE` is Spiral 25,388 + Irregular 7,100 —
exactly the two halves `make_disk_union.py` stacks — with **zero** early types
(E/S0/PSF/REX), so no morphology selection is needed. `JOHN_VI` has 1,120
`reject` rows, so it is **not** VI-filtered. Its columns are a strict subset of
the local union's; the 7 extra columns there (`MAIN`, `GOOD_MORPH`,
`MU_ZCMB_ERR`, `R_ABSMAG_SB26_TF*`) are pipeline outputs written by a previous
run, not new data. No negative magnitude errors.

Three consequences:

- **Nothing needs downloading.** Phase A runs locally on files already in
  `data/`.
- **Step 0 is still required, for the VI cut alone.** Do not feed
  `DR2_TF_disks_v5.fits` to the pipeline: nothing downstream applies
  `JOHN_VI` — `desi_data.py` and `color_predict.py` re-derive validity and
  phase-space cuts but not visual inspection — so you would silently fit
  1,120 VI-rejected galaxies and lose comparability with v5b/v10b. The
  stacking half of Step 0 is already done; its output,
  `data/DR2_TF_disks_v5_all.fits`, already exists. **Use that as `--fits`.**
- **The `MAIN` column in the local union is stale** — 28,856 true, which is
  v5a's MAIN count, so it was written by the v5a (Hyper) run and says nothing
  about v11a. It is harmless only because the config keeps `source: "DESI"`:
  `desi_data.py:232-239` gates the MAIN restriction on the source value
  precisely so a DR2 run is never re-selected against a stale pipeline-written
  MAIN. Do not change `source` to `"fullmocks"`, and do not filter on MAIN.
  (This also settles the open question of how v5a was built: from the combined
  disk set *before* VI filtering, which is why its row count is 32,488.)

Counts in `data/DR2_TF_disks_v5_all.fits` that set expectations for the v11a
run — Spiral 24,813 / Irregular 6,555; z>0.01 30,897; z>0.03 25,332;
0.03 < z <= 0.065 **14,682**. Of those, 13,472 also carry the stale MAIN flag,
which is the right order of magnitude for v11a's own MAIN count and hence for
the mock batch's `target_main_count` (section 7a) — but read the real number
off the completed catalogue rather than using this one.

[run_disks_single.sh](run_disks_single.sh) derives `RUN`/`CONFIG` from the FITS
basename the same way and automates Steps 1–8, pausing only for Step 3's
interactive fiducial choice:

```bash
./run_disks_single.sh --fits data/DR2_TF_disks_v5_all.fits
```

### The v11a variant: z > 0.03 and a redshift-dependent slope

`run_disks_single.sh` also takes `--model` and `--z_obs_min` / `--z_obs_max`:

```bash
# v11a: combined sample, z > 0.03 training cut, redshift-dependent TF slope
./run_disks_single.sh --fits <combined>.fits --run-suffix DR2_TF_all_v11a \
    --model 2color_v10 --z_obs_min 0.03

# v11a_fixedslope: the control -- same sample and cuts, fixed slope
./run_disks_single.sh --fits <combined>.fits --run-suffix DR2_TF_all_v11a_fixedslope \
    --model 2color --z_obs_min 0.03
```

`--z_obs_min` is not cosmetic. `selection_ellipse.py` fits the GMM *under* the
cut, and `set_fiducial.py:102` derives `slope_plane` / `intercept_plane` /
`intercept_plane2` from the resulting `selection_ellipse.json` via
`_cuts_at_nsigma`, so the z cut propagates into the cut geometry rather than
only into a config field. Take `haty_min` / `haty_max` at Step 3 from **this**
sample's own `select_v2_pull.png`; for reference the combined sample at
z > 0.01 refit to -21.7 / -18.0, with all five cut parameters moving off the
spiral values.

Because the cuts are refit, a v11a run differs from
`DR2_TF_disks_v5_all_2color` in **two** respects (the z cut and the cuts
themselves), so a difference between them cannot be attributed to the z cut
alone. `v5_3` deliberately froze its cuts when it raised `z_obs_min` for that
reason; a frozen-cut variant is worth adding later as a robustness check.

The two runs are a controlled pair: `2color_v10.stan` reduces to `2color.stan`
at `slope_std_1 = 0`, so they are likelihood-ratio comparable, and the pair is
what separates a measured `slope_dz` from the known selection artifact
(`c34ac82` records v10 returning `slope_dz` at 7.6 sigma on a mock whose true
drift is zero by construction).

It builds the stacked catalog first if it is missing, so the explicit
`make_disk_union.py` call is optional. Its options match run_dr2_onepop.sh
(`--from-step`, `--chains`, `--warmup`, `--delta`, `--max-depth`,
`--run-suffix`) minus `--population`. Steps 1–3b run in the foreground; Steps
4–8 are re-exec'd detached and take hours, logged to
`output/$RUN/run_dr2_phase2.log`. Each step is sentinel-guarded
(`output/$RUN/.step{N}_done`), and Step 6 tracks per-chain sentinels, so a
partial failure only reruns the missing chains.

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

# Step 8: predict absolute magnitudes (x-only is the default)
# --model is omitted on purpose: color_predict.py's precedence is explicit
# flag > config "model" > "color", so $CONFIG's own "model" key decides.
python color_predict.py --config $CONFIG
python explore_residuals.py --config $CONFIG --kind 2color   # --kind 2color_v10 for a v10 run
```

Two things about Step 3 and Step 3b that matter here:

- **Choose `haty_min`/`haty_max` from this run's own `select_v2_pull.png`.**
  The unsplit sample spans both populations' luminosity ranges, so
  DR2_TWOPOP.md's spiral/irregular reference values (`-22.0` / `-17.8` /
  `-17.5`) do not carry over.
- **Add `dust_pickle` to `$CONFIG` by hand** after Step 3b — it is not prompted
  for, and without it step 8 silently falls back to the iron default
  `d_err_r = 0.17680325` instead of the correct loa value `0.21734862`:

  ```json
  "dust_pickle": "data/loa_internalDust_nokcorr_mcmc.pickle"
  ```

  Verify: `grep "Loaded d_err_r" output/$RUN/step8.log`. See DR2_TWOPOP.md
  Step 3b for the full resolution order.

**There is no Step 9.** `combine_color_xonly.py` exists to stitch two
independently-fit populations back together with their shared systematics; a
single fit has nothing to combine, and step 8 already writes the final
`color_xonly_catalog.fits` and `color_xonly_cov.h5` for the whole sample.

**`--full` is unavailable for `2color_v10`.** `color_predict.py`'s `_reject_v10`
(lines 753 and 1099) blocks the full-colour predictors; only the x-only path
supports v10. Do not pass `--full` on a v10 run -- and note that x-only is
the default and the delivered product either way, so nothing is lost.

**For a v10 run, replace `2color` with `2color_v10` throughout the manual
commands above** -- the binary (`./2color_v10`, and `./2color_v10_g` on GPU),
the CSV names, the `stansummary`/`diagnose` globs, `corner.py --model`, and
`explore_residuals.py --kind`. `run_disks_single.sh --model 2color_v10` does
all of that for you, and so does the NERSC chain with
`--export=...,MODEL=2color_v10` (see [BATCH_NERSC.md](BATCH_NERSC.md)).

---

## File Reference

| File | Purpose |
|------|---------|
| `make_disk_union.py` | Stacks the two v5 disk catalogs, drops VI-rejects |
| [run_disks_single.sh](run_disks_single.sh) | Automates Steps 1–8 for the unsplit sample |
| [DR2_TWOPOP.md](DR2_TWOPOP.md) | The morphology-split chain: rationale for every sampler setting used above |
| [DR2_SINGLE.md](DR2_SINGLE.md) | The other single-population run — full loa catalog, early types included |
| `2color.stan` | Stan model: quadrivariate TFR with two independent color factors |
| `data/loa_internalDust_nokcorr_mcmc.pickle` | Internal-dust MCMC for DR2; `"dust_pickle"` in `$CONFIG` points here |
