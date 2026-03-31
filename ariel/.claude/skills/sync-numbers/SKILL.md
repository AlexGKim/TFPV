# sync-numbers skill

Update all hardcoded numeric values in `paper/main.tex` from the current
`output/DR1/` result files.  Run this after any DR1 re-fit.

## Steps

### 1. Extract full-precision MCMC posteriors (stansummary.txt is truncated)

```bash
python3 -c "
import glob, pandas as pd, numpy as np, json, sys
files = sorted(glob.glob('output/DR1/tophat_?.csv'))
if not files:
    sys.exit('No tophat_?.csv files found in output/DR1/')
dfs = [pd.read_csv(f, comment='#') for f in files]
df = pd.concat(dfs, ignore_index=True)
pairs = [('slope','slope'), ('intercept','intercept.1'),
         ('sigma_int_x','sigma_int_x'), ('sigma_int_y','sigma_int_y')]
r = {k: {'mean': float(df[v].mean()), 'std': float(df[v].std())}
     for k,v in pairs}
print(json.dumps(r, indent=2))
"
```

Format each as `mean ± std` with **3 decimal places**.

### 2. Extract training sample N

```bash
python3 -c "
from astropy.io import fits; import numpy as np
with fits.open('output/DR1/tophat_catalog.fits') as h:
    print(int(np.sum(h[1].data['MAIN'])))
"
```

Format with LaTeX thousands separator: e.g. `5{,}928`.

### 3. Read JSON source files

- `output/DR1/selection_ellipse.json`
- `output/DR1/select_v2_mle.json`
- `output/DR1/select_v2_fiducial.json`

### 4. Compute formatted values

| Paper quantity | Source | Format |
|----------------|--------|--------|
| GMM weight $w$ | `selection_ellipse.json` → `weight` | 3 dp |
| GMM mean $(\bar x, \bar y)$ | `selection_ellipse.json` → `mean[0]`, `mean[1]` | 3 dp each |
| GMM semi-axes | `selection_ellipse.json` → `semi_axes[0]`, `semi_axes[1]` | 3 dp each |
| GMM angle | `selection_ellipse.json` → `angle_deg` | 1 dp |
| MLE slope | `select_v2_mle.json` → `slope` | 4 dp |
| MLE intercept | `select_v2_mle.json` → `intercept.1` | 4 dp |
| MLE $\sigma_{\rm int,x}$ | `select_v2_mle.json` → `sigma_int_x` | sci notation e.g. `8.083 \times 10^{-5}` |
| MLE $\sigma_{\rm int,y}$ | `select_v2_mle.json` → `sigma_int_y` | 4 dp |
| $\hat y_{\rm min}$ | `select_v2_fiducial.json` → `haty_min` | 2 dp |
| $\hat y_{\rm max}$ | `select_v2_fiducial.json` → `haty_max` | 2 dp |
| Oblique slope $\bar s$ | `select_v2_fiducial.json` → `slope_plane` | 3 dp |
| Lower intercept $\bar c_1$ | `select_v2_fiducial.json` → `intercept_plane` | 3 dp |
| Upper intercept $\bar c_2$ | `select_v2_fiducial.json` → `intercept_plane2` | 3 dp |
| $n_{\sigma,\perp}$ | `select_v2_fiducial.json` → `n_sigma_perp` | 1 dp |
| $z_{\rm min}$, $z_{\rm max}$ | `select_v2_fiducial.json` → `z_obs_min`, `z_obs_max` | 2 dp |
| MCMC $\alpha \pm \sigma$ | step 1 → `slope` | 3 dp mean, 3 dp std |
| MCMC $\beta \pm \sigma$ | step 1 → `intercept` | 3 dp mean, 3 dp std |
| MCMC $\sigma_{\rm int,x} \pm \sigma$ | step 1 → `sigma_int_x` | 3 dp mean, 3 dp std |
| MCMC $\sigma_{\rm int,y} \pm \sigma$ | step 1 → `sigma_int_y` | 3 dp mean, 3 dp std |
| Training $N$ | step 2 | thousands-separated |

### 5. Locate and update main.tex

Use `mcp__latex-server__get_latex_structure` on `paper/main.tex`, then
`mcp__latex-server__read_latex_file` to read the relevant sections.

Passages to update (search for the old value strings):

**Abstract** — MCMC α, β, σ_int_x, σ_int_y with errors; training N

**Section "Sample Selection" / GMM paragraph** — w, mean, semi-axes, angle

**MLE paragraph** — MLE slope, intercept, σ_int_x (sci notation), σ_int_y

**Table 1 (fiducial parameters)** — all fiducial fields; training N row

**Table 2 (MCMC results)** — α, β, σ_int_x, σ_int_y means and stds

**Summary section** — w, training N, MCMC α, β, σ_int_x, σ_int_y

Use `mcp__latex-server__edit_latex_file` for each replacement.

### 6. Validate

Run `mcp__latex-server__validate_latex` on `paper/main.tex`.

### 7. Report

List every value that changed as `quantity: OLD → NEW`.
If a value is unchanged, note it was verified correct.
