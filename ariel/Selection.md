# Sample Selection

## Introduction

The objective of this stage is to use sample selection to obtain a normal galaxy
sample with high purity and completeness.  Normal galaxies occupy a compact region
of the absolute-magnitude / rotation-velocity phase space and obey a Tully-Fisher
relation with a specific slope.  A small fraction of other galaxies do not follow
the Tully-Fisher relation and for the most part occupy a different region of the
phase space.  An analysis of the full sample would yield a different slope with a
poorer quality of fit.

## Step 1: Estimating the core distribution

Algorithm: `selection_ellipse.py`

The script fits a 2-component Gaussian Mixture Model (GMM) to the observed
(x, y) = (log₁₀(V/100 km/s), R-band absolute magnitude) phase space, identifies
the tight TFR core component, and derives selection-cut parameters directly from
its 1σ confidence ellipse.  The fit corrects for both per-galaxy measurement noise
and survey truncation.

### 1. Data loading

Galaxies are read from a FITS file (AbacusSummit mock or real survey).  Only
objects with `MAIN=True` are retained.  A loose pre-filter
(`--haty_min`, `--haty_max`, defaults −23 and −18) removes unphysical extremes
before fitting.  Each galaxy contributes four quantities:

| Symbol | FITS column | Definition |
|--------|-------------|-----------|
| x | `LOGVROT − 2` | log₁₀(V / 100 km/s) |
| y | `R_ABSMAG_SB26` | R-band absolute magnitude |
| σ_x | `LOGVROT_ERR` | uncertainty in x |
| σ_y | `R_ABSMAG_SB26_ERR` | uncertainty in y |

### 2. Truncation detection

The survey is assumed to be truncated at:

- **x_hi = x.max()** — a hard upper limit on observed rotation velocity, and
- **y_lo = y.min()** — the bright (most-negative magnitude) completeness limit.

These bounds define the observable region R = {x ≤ x_hi, y ≥ y_lo}.

### 3. Truncated noisy GMM fit

A 2-component full-covariance GMM is fit by maximising the **truncated noisy
log-likelihood**:

$$\mathcal{L} = \sum_i \log \left[ \sum_{k=1}^{2} \frac{w_k}{Z_k}
  \mathcal{N}\!\left(\mathbf{x}_i;\, \boldsymbol{\mu}_k,\,
  \boldsymbol{\Sigma}_k + \mathrm{diag}(\sigma_{x,i}^2,\, \sigma_{y,i}^2)\right)
  \right]$$

**Per-point noise convolution.** Each galaxy's likelihood contribution is
evaluated under the effective covariance
Σ_k + diag(σ²_{x,i}, σ²_{y,i}), which deconvolves measurement noise from the
intrinsic component scatter.  This is evaluated in closed form via the vectorised
2×2 Gaussian formula (no scipy call per point).

**Truncation normalisation.** Each component is divided by

$$Z_k = P(\mathbf{X} \in R \mid \boldsymbol{\mu}_k,\, \bar{\boldsymbol{\Sigma}}_k)
       = P(X_1 \le x_\mathrm{hi}) - P(X_1 \le x_\mathrm{hi},\, X_2 \le y_\mathrm{lo})$$

where $\bar{\boldsymbol{\Sigma}}_k = \boldsymbol{\Sigma}_k +
\mathrm{diag}(\overline{\sigma_x^2}, \overline{\sigma_y^2})$ uses the
**sample-mean noise variances** (fixed scalars) rather than per-point values.
This keeps Z_k a scalar per component, making the optimisation tractable while
capturing the dominant correction.

**Optimisation.** The 11 free parameters
(logit weight, 2D means, lower-triangular Cholesky factors of both covariances)
are initialised from a standard sklearn `GaussianMixture` fit and then refined
with `scipy.optimize.minimize` (Powell method, `ftol=1e-9`).

### 4. Core component identification

The component with the smaller covariance determinant is identified as the TFR
core (tighter cluster).

### 5. Derived selection cuts

From the core component mean **μ** and covariance **Σ**, the following
selection-boundary parameters are derived geometrically from the 1σ ellipse.

**Eigendecomposition** of Σ yields eigenvalues λ₁ ≤ λ₂ with corresponding
unit eigenvectors **e**_minor and **e**_major.  Define:

- σ_minor = √λ₁, σ_major = √λ₂
- θ = arctan2(e_major[1], e_major[0]) — angle of major axis from x-axis
- slope = tan θ — TFR slope implied by ellipse orientation

**Horizontal magnitude cuts** (extreme y-values of the 1σ ellipse):

$$\hat{y}_\mathrm{min},\, \hat{y}_\mathrm{max}
  = \mu_y \;\mp\; \sqrt{\sigma_\mathrm{major}^2 \sin^2\theta
                       + \sigma_\mathrm{minor}^2 \cos^2\theta}$$

**Oblique plane cuts** (lines with slope = tan θ through the ± endpoints of the
semi-minor axis):

$$\mathbf{p}_{1,2} = \boldsymbol{\mu} \pm \sigma_\mathrm{minor}\,\mathbf{e}_\mathrm{minor}$$
$$b_{1,2} = p_{1,2,y} - \mathrm{slope} \times p_{1,2,x}$$

These four lines — two horizontal and two oblique — form a parallelogram that
tightly bounds the 1σ core ellipse and serve as starting-point selection cuts for
the Stan models.

### 6. Output

**`output/<run>/selection_ellipse.json`** — core Gaussian parameters and derived cuts:

```json
{
  "weight":            <mixture weight of core component>,
  "mean":              [mu_x, mu_y],
  "covariance":        [[Sxx, Sxy], [Sxy, Syy]],
  "semi_axes":         [sigma_minor, sigma_major],
  "angle_deg":         <major-axis angle in degrees>,
  "x_trunc":           <truncation upper bound in x>,
  "y_trunc":           <truncation lower bound in y>,
  "haty_min":          <bright-end horizontal cut>,
  "haty_max":          <dim-end horizontal cut>,
  "slope_plane":       <oblique line slope = tan(angle)>,
  "intercept_plane":   <intercept of plane-1 oblique line>,
  "intercept_plane2":  <intercept of plane-2 oblique line>
}
```

**`output/<run>/selection_ellipse.png`** — scatter plot of all valid galaxies
coloured by P(core component), with 1σ/2σ/3σ ellipses and the four derived
selection cuts overlaid.

### Usage

```bash
python selection_ellipse.py --file $FITS --run $RUN
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--file FILE` | required | Path to FITS file |
| `--run RUN` | required | Output subdirectory under `output/` |
| `--haty_min` | −23.0 | Loose lower magnitude pre-filter |
| `--haty_max` | −18.0 | Loose upper magnitude pre-filter |
| `--n_init` | 20 | GMM random restarts for initialisation |

## Step 2: Selection plateau search

Algorithm: `ellipse_sweep.py`

Starting from the GMM core parameters in `selection_ellipse.json`, this step
evaluates MLE TFR slopes on a 3-D grid over (n_σ_perp, n_σ_ŷmin, n_σ_ŷmax)
and automatically selects the fiducial cut values.

### 1. Data loading

Reads the same FITS file used in Step 1 (fullmocks or DESI) and applies
redshift cuts.  The raw sample may be randomly subsampled to
`--n_sweep_objects` objects (default 10 000) to keep each Stan call fast.
Reads `output/<run>/selection_ellipse.json` for the GMM mean **μ** and
covariance **Σ** of the core component.

### 2. Grid parametrisation

Three independent scale parameters control the selection boundary:

| Parameter | Controls |
|-----------|----------|
| n_σ_perp | width of the oblique plane cut (perpendicular to TFR) |
| n_σ_ŷmin | faint-end magnitude cut (ŷ_min = μ_y − n_σ_ŷmin · y_extent) |
| n_σ_ŷmax | bright-end magnitude cut (ŷ_max = μ_y + n_σ_ŷmax · y_extent) |

where y_extent = √(σ_major² sin²θ + σ_minor² cos²θ) is the y-projection of
the 1σ ellipse.  The oblique-cut slope (slope_plane = tan θ) is fixed
throughout.

### 3. Stan MAP optimisation

At each grid point the galaxy sample is filtered by the current cuts, then the
compiled `tophat` executable is called in `optimize` (MAP) mode via
`subprocess`.  The MLE slope s is read from the `slope` generated quantity in
the Stan CSV output.

### 4. Fiducial selection

The fiducial point is the highest-N grid cell satisfying
|MLE slope − GMM slope| ≤ slope_tol (default 0.5).  Ties are broken by
selecting the point with the most galaxies.  The chosen (n_σ_perp, n_σ_ŷmin,
n_σ_ŷmax) is converted to actual cut values (haty_min, haty_max,
intercept_plane, intercept_plane2) and written to
`output/<run>/mag_split_fiducial.json`.

### 5. Output

| File | Description |
|------|-------------|
| `output/<run>/mag_split_grid.json` | full 3-D grid results |
| `output/<run>/mag_split_grid.png` | (n_σ_perp × 2) heatmap of MLE slope |
| `output/<run>/fiducial_slope_hist.png` | histogram of MLE slopes at fiducial n_σ_perp |
| `output/<run>/mag_split_fiducial.json` | chosen cut values |

### Usage

```bash
python ellipse_sweep.py --source DESI --fits_file $FITS --run $RUN

# Replot from saved JSON without rerunning Stan:
python ellipse_sweep.py --source DESI --fits_file $FITS --run $RUN --mag_split_plot
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `fullmocks` | Data source: `fullmocks` or `DESI` |
| `--fits_file FILE` | auto | Path to FITS file; auto-detected from `--dir` if omitted |
| `--dir DIR` | `data/` | Directory searched for FITS files |
| `--run RUN` | required | Run name; reads `output/<run>/selection_ellipse.json` |
| `--exe EXE` | `tophat` | Path to compiled Stan tophat executable |
| `--z_obs_min` | 0.03 | Minimum redshift cut |
| `--z_obs_max` | 0.10 | Maximum redshift cut |
| `--n_sweep_objects` | 10000 | Subsample size (0 = use all) |
| `--slope_tol` | 0.5 | Max \|MLE slope − GMM slope\| for fiducial selection |
| `--n_sigma_perp_min/max/n` | 5/5/1 | n_σ_perp grid range and count |
| `--n_sigma_mag_lo_min/max/n` | 0.2/1.6/8 | n_σ_ŷmin grid range and count |
| `--n_sigma_mag_hi_min/max/n` | 2.5/4.5/9 | n_σ_ŷmax grid range and count |
| `--mag_split_plot` | off | Replot from saved `mag_split_grid.json`; no Stan calls |

## Step 3: Selection criteria

> **Note:** `selection_criteria.py` provides an alternative plateau-detection
> algorithm that reads a 1-D per-parameter sweep from `ellipse_sweep.json`.
> This file is not produced by the current `ellipse_sweep.py`; the script is
> not part of the active DESI or fullmocks workflow.  The fiducial cuts written
> by Step 2 (`mag_split_fiducial.json`) are consumed directly by the data
> preparation scripts in Step 4.
