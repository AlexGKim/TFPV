# Sample Selection

## Introduction

The objective of this stage is to use sample selection to obtain a normal galaxy
sample with high purity and completeness.  Normal galaxies occupy a compact region
of the absolute-magnitude / rotation-velocity phase space and obey a Tully-Fisher
relation with a specific slope.  A small fraction of other galaxies do not follow
the Tully-Fisher relation and for the most part occupy a different region of the
phase space.  An analysis of the full sample would yield a different slope with a
poorer quality of fit.

## Algorithm: `selection_ellipse.py`

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
python selection_ellipse.py \
    --file data/TF_extended_AbacusSummit_base_c000_ph000_r001_z0.11.fits \
    --run c000_ph000_r001
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--file FILE` | required | Path to FITS file |
| `--run RUN` | required | Output subdirectory under `output/` |
| `--haty_min` | −23.0 | Loose lower magnitude pre-filter |
| `--haty_max` | −18.0 | Loose upper magnitude pre-filter |
| `--n_init` | 20 | GMM random restarts for initialisation |
