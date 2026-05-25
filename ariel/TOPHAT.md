# Tophat + K-correction Run: SGA-2020

This document records the full command sequence for fitting and predicting using
the baseline tophat TFR model with r-band k-correction (α_k,r) on the
`SGA-2020_iron_Vrot_VI_corr_v5.fits` dataset.

The tophat + k-correction model extends the baseline tophat fit by adding a
linear k-correction in log(1+z), absorbing the redshift-dependent magnitude
offset in the r-band. See `doc/model1.tex` for the formal model.

## Full Workflow

| Phase | Steps | How |
|-------|-------|-----|
| A — Setup | 1–3b | Complete [DR1.md](DR1.md) Steps 1–3b (selection ellipse, pull profile, fiducial, export config) |
| B — Tophat+kcorr | 4–8 | Run Steps 4–8 below (data prep, compilation, MAP, sampling, prediction) |

---

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr_v5.fits   # input FITS catalog
export RUN=DR1_v6_tophat                              # output directory name: output/$RUN/
export CONFIG=configs/dr1_v6.json                     # pipeline config
```

---

## Step 4: Prepare data

Convert the FITS file to Stan JSON format. `desi_data.py` includes `z_obs` in
the output (needed by the k-correction in the Stan model).

```bash
# via config
python desi_data.py --config $CONFIG

# via flags
python desi_data.py --input $FITS --run $RUN \
    --haty_min -21.5 --haty_max -19.0 \
    --slope_plane -6.386925076468424 \
    --intercept_plane -20.74814050932727 \
    --intercept_plane2 -18.31309635087515 \
    --z_obs_min 0.03 --z_obs_max 0.08
```

Inspect the scatter plot:

```bash
open output/$RUN/data.png
```

---

## Step 5: Compile Stan model

Run from inside the `../../cmdstan/` directory:

```bash
cd ../../cmdstan
make ../TFPV/ariel/tophat
cd ../TFPV/ariel
```

---

## Step 5d: Find MAP estimate (init_MAP.json)

The MAP provides a warm start near the posterior mode, reducing warmup time.

```bash
./tophat optimize \
    data file=output/$RUN/input.json \
    init=output/$RUN/init.json \
    output file=output/$RUN/optimize.csv

# Convert optimizer output to MCMC init file
python - <<'EOF'
import pandas as pd, json, os

RUN = os.environ['RUN']
df = pd.read_csv(f'output/{RUN}/optimize.csv', comment='#')
row = df.iloc[0]
old = json.load(open(f'output/{RUN}/init.json'))
new = {}
for k in old.keys():
    if k == 'intercept_std':
        cols = sorted([c for c in df.columns if c.startswith('intercept_std.')],
                      key=lambda s: int(s.split('.')[1]))
        new[k] = [float(row[c]) for c in cols]
    elif k in df.columns:
        new[k] = float(row[k])
    else:
        new[k] = old[k]
with open(f'output/{RUN}/init_MAP.json', 'w') as f:
    json.dump(new, f, indent=2)
print(f'MAP init written to output/{RUN}/init_MAP.json')
EOF
```

---

## Step 6: Run MCMC sampling

```bash
./tophat sample num_warmup=250 num_samples=1000 num_chains=4 \
    adapt save_metric=1 \
    algorithm=hmc metric=dense_e \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/tophat.csv
```

---

## Step 7: Diagnose and visualize

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/$RUN/tophat_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/tophat_?.csv > output/$RUN/diagnose.txt

# Corner plot
python corner.py --run $RUN --model tophat
```

Inspect:

```bash
cat output/$RUN/stansummary.txt
open output/$RUN/tophat.png
```

Key parameters to check:
- `slope` — TFR slope (expected ~ -8)
- `intercept` — TFR zero-point (expected ~ -20)
- `sigma_int_x` — intrinsic x-scatter
- `sigma_int_y` — intrinsic y-scatter
- `alpha_kcorr_r` — r-band k-correction slope (expected ~ -5 to -6)

---

## Step 8: Predict absolute magnitudes

```bash
python predict.py --run $RUN --model tophat
```

The script reads:
- `output/$RUN/config.json` — phase-space selection and FITS path
- `output/$RUN/input.json` — bounds (y_min, y_max) and z_obs (for mean_log1pz)
- `output/$RUN/tophat_?.csv` — posterior MCMC draws

Diagnostic plots produced:

| File | Description |
|------|-------------|
| `output/$RUN/tophat_grid.png` | Mean residual on (x̂, ŷ) grid |
| `output/$RUN/redshift_tophat.png` | Residual vs. redshift scatter |
| `output/$RUN/variance_redshift_tophat.png` | Prediction variance vs. redshift |

---

## Step 8b: Write DESI catalog

```bash
python predict.py --run $RUN --model tophat --write-catalog
```

Produces `output/$RUN/tophat_catalog.fits` with columns:
- `MU_TF` — TF distance modulus
- `MU_ERR` — distance modulus uncertainty
- `LOGDIST` — log-distance ratio
- `LOGDIST_ERR` — log-distance ratio error
- `MAIN` — flag for galaxies passing all selection cuts

---

## Step 8c: Write covariance matrix

```bash
python predict.py --run $RUN --model tophat --write-cov
```

Produces `output/$RUN/tophat_cov.png` — visualization of the posterior
predictive covariance matrix for the MAIN sample.

---

## File Reference

| File | Purpose |
|------|---------|
| `tophat.stan` | Stan model: baseline TFR + r-band k-correction |
| `predict.py` | Posterior predictive computation and diagnostics |
| `desi_data.py` | Data preparation (FITS → Stan JSON + init) |
| `corner.py` | Corner plot from MCMC output |
| [TOPHAT.md](TOPHAT.md) | This file — tophat + k-correction workflow |
