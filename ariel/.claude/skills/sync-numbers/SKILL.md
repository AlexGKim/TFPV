# sync-numbers skill

Update all hardcoded numeric values in `paper/main.tex` from the result files in
whichever directory `\ResultDir` points to.  Run this after any re-fit.

## Steps

### 0. Resolve the result directory

Read `\ResultDir` from `paper/main.tex` and derive the project-root-relative path:

```python
import re, pathlib
tex = pathlib.Path('paper/main.tex').read_text()
m = re.search(r'\\newcommand\{\\ResultDir\}\{([^}]+)\}', tex)
if not m:
    raise SystemExit('\\ResultDir not found in paper/main.tex')
# LaTeX path is relative to paper/; strip leading ../ to get project-root path
result_dir = m.group(1).lstrip('../').rstrip('/')   # e.g. "output/DR1"
print('Using result directory:', result_dir)
```

Use `result_dir` in place of `output/DR1` in every subsequent step.

### 1. Extract full-precision MCMC posteriors and quantiles

```bash
python3 -c "
import re, pathlib, glob, pandas as pd, numpy as np, json, sys
tex = pathlib.Path('paper/main.tex').read_text()
m = re.search(r'\\\\newcommand\{\\\\ResultDir\}\{([^}]+)\}', tex)
result_dir = m.group(1).lstrip('../').rstrip('/')
files = sorted(glob.glob(f'{result_dir}/tophat_?.csv'))
if not files:
    sys.exit(f'No tophat_?.csv files found in {result_dir}/')
dfs = [pd.read_csv(f, comment='#') for f in files]
df = pd.concat(dfs, ignore_index=True)
pairs = [('slope','slope'), ('intercept','intercept.1'),
         ('sigma_int_x','sigma_int_x'), ('sigma_int_y','sigma_int_y')]
r = {}
for k,v in pairs:
    col = df[v]
    r[k] = {'mean': float(col.mean()), 'std': float(col.std()),
             'p5': float(col.quantile(0.05)), 'p50': float(col.quantile(0.50)),
             'p95': float(col.quantile(0.95))}
print(json.dumps(r, indent=2))
"
```

Format mean ± std with **3 decimal places**; p5/p50/p95 with **3 decimal places**.

### 2. Extract ESS_bulk and R_hat from stansummary.txt

Parse `{result_dir}/stansummary.txt` for the rows: `slope`, `intercept[1]`, `sigma_int_x`, `sigma_int_y`.
Extract ESS_bulk (column 8) and R_hat (column 10) for each. Values are integers (ESS) and 1 decimal place (R_hat).

### 3. Extract N counts

```bash
python3 -c "
import re, pathlib, json
tex = pathlib.Path('paper/main.tex').read_text()
m = re.search(r'\\\\newcommand\{\\\\ResultDir\}\{([^}]+)\}', tex)
result_dir = m.group(1).lstrip('../').rstrip('/')
with open(f'{result_dir}/config.json') as f:
    cfg = json.load(f)
print('n_training:', cfg['n_training'])
print('n_total_fits:', cfg['n_total_fits'])
"
```

Format both with LaTeX thousands separator: e.g. `7{,}525`.

### 4. Read JSON source files

- `{result_dir}/selection_ellipse.json`
- `{result_dir}/select_v2_mle.json`
- `{result_dir}/select_v2_fiducial.json`

### 5. Compute formatted values

| Command | Source | Format |
|---------|--------|--------|
| `\Ntrain` | `config.json` → `n_training` | thousands-separated |
| `\Ntotal` | `config.json` → `n_total_fits` | thousands-separated |
| `\GmmW` | `selection_ellipse.json` → `weight` | 3 dp |
| `\GmmMeanX` | `selection_ellipse.json` → `mean[0]` | 3 dp |
| `\GmmMeanY` | `selection_ellipse.json` → `mean[1]` | 3 dp |
| `\GmmSemiA` | `selection_ellipse.json` → `semi_axes[0]` | 3 dp |
| `\GmmSemiB` | `selection_ellipse.json` → `semi_axes[1]` | 3 dp |
| `\GmmAngle` | `selection_ellipse.json` → `angle_deg` | 1 dp |
| `\MleSlope` | `select_v2_mle.json` → `slope` | 4 dp |
| `\MleIntercept` | `select_v2_mle.json` → `intercept.1` | 4 dp |
| `\MleSigIntX` | `select_v2_mle.json` → `sigma_int_x` | sci notation e.g. `8.083 \times 10^{-5}` |
| `\MleSigIntY` | `select_v2_mle.json` → `sigma_int_y` | 4 dp |
| `\FidHatyMin` | `select_v2_fiducial.json` → `haty_min` | 2 dp |
| `\FidHatyMax` | `select_v2_fiducial.json` → `haty_max` | 2 dp |
| `\FidSlopePlane` | `select_v2_fiducial.json` → `slope_plane` | 3 dp |
| `\FidCOne` | `select_v2_fiducial.json` → `intercept_plane` | 3 dp |
| `\FidCTwo` | `select_v2_fiducial.json` → `intercept_plane2` | 3 dp |
| `\FidNSigPerp` | `select_v2_fiducial.json` → `n_sigma_perp` | 1 dp |
| `\FidZMin` | `select_v2_fiducial.json` → `z_obs_min` | 2 dp |
| `\FidZMax` | `select_v2_fiducial.json` → `z_obs_max` | 2 dp |
| `\SlopeMean` | step 1 → `slope.mean` | 3 dp |
| `\SlopeStd` | step 1 → `slope.std` | 3 dp |
| `\SlopeQlo` | step 1 → `slope.p5` | 3 dp |
| `\SlopeQmed` | step 1 → `slope.p50` | 3 dp |
| `\SlopeQhi` | step 1 → `slope.p95` | 3 dp |
| `\SlopeEss` | step 2 → slope ESS_bulk | integer |
| `\SlopeRhat` | step 2 → slope R_hat | 1 dp |
| `\IntMean` | step 1 → `intercept.mean` | 3 dp |
| `\IntStd` | step 1 → `intercept.std` | 3 dp |
| `\IntQlo` | step 1 → `intercept.p5` | 3 dp |
| `\IntQmed` | step 1 → `intercept.p50` | 3 dp |
| `\IntQhi` | step 1 → `intercept.p95` | 3 dp |
| `\IntEss` | step 2 → intercept ESS_bulk | integer |
| `\IntRhat` | step 2 → intercept R_hat | 1 dp |
| `\SigXMean` | step 1 → `sigma_int_x.mean` | 3 dp |
| `\SigXStd` | step 1 → `sigma_int_x.std` | 3 dp |
| `\SigXQlo` | step 1 → `sigma_int_x.p5` | 3 dp |
| `\SigXQmed` | step 1 → `sigma_int_x.p50` | 3 dp |
| `\SigXQhi` | step 1 → `sigma_int_x.p95` | 3 dp |
| `\SigXEss` | step 2 → sigma_int_x ESS_bulk | integer |
| `\SigXRhat` | step 2 → sigma_int_x R_hat | 1 dp |
| `\SigYMean` | step 1 → `sigma_int_y.mean` | 3 dp |
| `\SigYStd` | step 1 → `sigma_int_y.std` | 3 dp |
| `\SigYQlo` | step 1 → `sigma_int_y.p5` | 3 dp |
| `\SigYQmed` | step 1 → `sigma_int_y.p50` | 3 dp |
| `\SigYQhi` | step 1 → `sigma_int_y.p95` | 3 dp |
| `\SigYEss` | step 2 → sigma_int_y ESS_bulk | integer |
| `\SigYRhat` | step 2 → sigma_int_y R_hat | 1 dp |

### 6. Update the \newcommand block in paper/main.tex

The `\newcommand` block is at the top of the file, just before `\begin{document}`, between the comment lines:
```
%% Synced values — updated automatically by /sync-numbers
```
and
```
\newcommand{\SigYRhat}{...}
```

For each command whose value has changed, use `mcp__latex-server__edit_latex_file` with `replace` to update **only that line**. For example:
```
search:  \newcommand{\Ntrain}{6{,}140}
replace: \newcommand{\Ntrain}{7{,}525}
```

Do NOT touch any other part of the document — all value propagation is handled automatically by LaTeX.

### 7. Validate

Run `mcp__latex-server__validate_latex` on `paper/main.tex`.

### 8. Report

List every `\newcommand` value that changed as `\CommandName: OLD → NEW`.
If a value is unchanged, note it was verified correct.
