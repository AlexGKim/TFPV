# Two-Color Run: SGA-2020

This document records the full command sequence for fitting and predicting using
the two-color TFR model on the `SGA-2020_iron_Vrot_VI_corr_v6.fits` dataset.

The two-color model extends the single-color (r–z) model by adding g-band as a
**second independent latent color factor** (g–r). The joint distribution becomes
quadrivariate (x̂, ŷ, ẑ, ĝ) with a 4×4 covariance matrix. The two color factors
ε_c (r–z) and ε_g (g–r) are uncorrelated. See `2color.stan` for the implementation.

## Full Workflow

| Phase | Steps | How |
|-------|-------|-----|
| A — Setup | 1–3b | Complete [DR1.md](DR1.md) Steps 1–3b (selection ellipse, pull profile, fiducial, export config) |
| B — 2color | 4–8 | Run Steps 4–8 below (data prep, compilation, MAP, metric, sampling, prediction) |

---

## Setup

```bash
export FITS=data/SGA-2020_iron_Vrot_VI_corr_v6.fits   # input FITS catalog
export RUN=DR1_v6_2color                              # output directory name: output/$RUN/
export CONFIG=configs/dr1_v6_2color.json              # pipeline config
```

---

## Step 4: Prepare data

Convert the FITS file to Stan JSON format. Both z-band and g-band magnitudes are
loaded automatically when the FITS catalog contains the relevant columns.

```bash
# via config
python desi_data.py --config $CONFIG
```

`desi_data.py` writes `g`, `sigma_g`, and `c_bar_g_obs` to `input.json` in addition
to the z-band fields required by `color.stan`.

Inspect the scatter plot:

```bash
open output/$RUN/data.png
```

---

## Step 5: Compile Stan model

Run from inside the `../../cmdstan/` directory:

```bash
cd ../../cmdstan
make ../TFPV/ariel/2color
cd ../TFPV/ariel
```

---

## Step 5d: Find MAP estimate (init_MAP.json)

The MAP provides a warm start near the posterior mode.

```bash
./2color optimize \
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

## Step 5e: Build initial metric from short run

The 2color model has highly varying parameter scales (condition number ~2.7M),
so the default identity mass matrix leads to very small stepsizes and deep
leapfrog trees. Running a short 1-chain warmup and extracting the sample
covariance as an initial metric dramatically reduces warmup cost.

```bash
# Short 1-chain run to adapt metric (100 warmup + 100 samples, ~7 hours)
./2color sample num_warmup=100 num_samples=100 num_chains=1 \
    algorithm=hmc metric=dense_e \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color_metric_init.csv

# Extract sample covariance as inverse mass matrix
python - <<'EOF'
import pandas as pd, numpy as np, json, os

RUN = os.environ['RUN']
df = pd.read_csv(f'output/{RUN}/2color_metric_init.csv', comment='#')
sampling_params = [
    'slope_std', 'intercept_std.1', 'sigma_int_x', 'sigma_int_y',
    'log_sigma_int_z', 'gamma_tau_c', 'delta_c', 'mu_c', 'log_tau_c',
    'gamma_tau_g', 'delta_g', 'mu_g', 'log_tau_g', 'log_sigma_int_g',
    'alpha_kcorr_r', 'alpha_kcorr_z', 'alpha_kcorr_g'
]
X = df[sampling_params].values
cov = np.cov(X.T)
metric = {"inv_metric": cov.tolist()}
with open(f'output/{RUN}/metric.json', 'w') as f:
    json.dump(metric, f)
print(f'Metric written to output/{RUN}/metric.json')
print(f'Condition number: {np.linalg.cond(cov):.1f}')
EOF
```

---

## Step 6: Run MCMC sampling

Provide the pre-computed metric via `metric_file` to skip poorly-scaled warmup.
The 250-step warmup will refine the metric further.

```bash
./2color sample num_warmup=250 num_samples=1000 num_chains=4 \
    adapt save_metric=1 \
    algorithm=hmc metric=dense_e \
    metric_file=output/$RUN/metric.json \
    data file=output/$RUN/input.json \
    init=output/$RUN/init_MAP.json \
    output file=output/$RUN/2color.csv
```

This produces `2color_1.csv` … `2color_4.csv` in `output/$RUN/`.

---

## Step 7: Diagnose and visualize

```bash
# Convergence diagnostics
../../cmdstan/bin/stansummary output/$RUN/2color_?.csv > output/$RUN/stansummary.txt
../../cmdstan/bin/diagnose    output/$RUN/2color_?.csv > output/$RUN/diagnose.txt

# Corner plot
python corner.py --run $RUN --model 2color
```

Inspect:

```bash
cat output/$RUN/stansummary.txt
open output/$RUN/2color.png
```

Key parameters to check:
- `slope` — TFR slope
- `gamma` — r–z luminosity–color slope
- `gamma_g` — g–r luminosity–color slope
- `delta_c`, `delta_g` — color–velocity slopes
- `tau_c`, `tau_g` — intrinsic color scatter
- `alpha_kcorr_r`, `alpha_kcorr_z`, `alpha_kcorr_g` — band k-corrections

---

## Step 8: Predict absolute magnitudes

```bash
python color_predict.py --config $CONFIG --model 2color --xonly
```

The script reads:
- `output/$RUN/config.json` — phase-space selection and FITS path
- `output/$RUN/input.json` — bounds, mean_x, z-band and g-band data
- `output/$RUN/2color_?.csv` — posterior MCMC draws

Diagnostic plots produced:

| File | Description |
|------|-------------|
| `output/$RUN/2color_grid.png` | Mean residual on (x̂, ŷ) grid |
| `output/$RUN/redshift_2color.png` | Residual vs. redshift scatter |
| `output/$RUN/variance_redshift_2color.png` | Prediction variance vs. redshift |
| `output/$RUN/gr_color_2color.png` | Residual vs. g–r color |

---

## Step 8b: Write DESI catalog

```bash
python color_predict.py --config $CONFIG --model 2color
```

Produces `output/$RUN/2color_catalog.fits`.

---

## Step 8c: Write covariance matrix

```bash
python color_predict.py --config $CONFIG --model 2color --no-catalog
```

Produces `output/$RUN/2color_cov.fits`.

---

## Performance note

The 4×4 Cholesky per galaxy makes each gradient evaluation ~4× more expensive
than the 3×3 color model. With N=4728 galaxies, expect:

| Phase | Approx. time |
|-------|-------------|
| MAP optimization | ~20 min |
| Short metric-init run (1 chain, 200 iter) | ~7 hours |
| Full MCMC (4 chains, 1250 iter, with metric) | TBD — expect significant speedup vs. identity metric |

The identity mass matrix produces treedepth ≈ 10 (maxed out) and stepsize ≈ 0.002.
The pre-computed metric should bring treedepth to ≈ 4–6 and enable a practical run.

---

## File Reference

| File | Purpose |
|------|---------|
| `2color.stan` | Stan model: quadrivariate TFR with two independent color factors |
| `color_predict.py --model 2color` | Posterior predictive computation (shared with color model) |
| `desi_data.py` | Data preparation — writes g-band fields when FITS has G_MAG_SB26 |
| `corner.py` | Corner plot including γ_g, δ_g, μ_g, τ_g, α_{k,g} |
| `configs/dr1_v6_2color.json` | Pipeline config for this run |
| [COLOR.md](COLOR.md) | Single-color (r–z) workflow |
| [TOPHAT.md](TOPHAT.md) | Baseline tophat + k-correction workflow |
| [2COLOR.md](2COLOR.md) | This file |
