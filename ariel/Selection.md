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

The four selection-cut parameters derived in Step 1 from the 1σ GMM ellipse
(ŷ_min, ŷ_max, c₁, c₂) are starting-point estimates.  Tightening or loosening
each cut changes which galaxies enter the Stan model and therefore shifts the
maximum-likelihood TFR slope s.  The goal of this step is to identify the
range of each parameter for which s is insensitive to small perturbations —
the **stability plateau** — and to verify that the 1σ ellipse boundary falls
within that plateau.

### 1. Data loading

Reads the same FITS file used in Step 1 (fullmocks or DESI) and applies
redshift cuts z_obs_min < z ≤ z_obs_max.  The raw sample may be randomly
subsampled to `--n_sweep_objects` objects (default 10 000) to keep each Stan
call fast.  Reads `output/<run>/selection_ellipse.json` for the GMM mean **μ**
and covariance **Σ** of the core component.

### 2. Ellipse scale parametrisation

Each cut is expressed as a function of a single scale parameter n_σ ≥ 0.
Eigendecomposing **Σ** gives σ_minor, σ_major, and major-axis angle θ.  The
four cuts at scale n_σ are:

| Cut | At scale n_σ |
|-----|-------------|
| ŷ_min | μ_y − n_σ · y_extent |
| ŷ_max | μ_y + n_σ · y_extent |
| c₁ (intercept_plane) | min intercept of lines through ±n_σ·σ_minor endpoints |
| c₂ (intercept_plane2) | max intercept of same lines |

where y_extent = √(σ_major² sin²θ + σ_minor² cos²θ) is the y-projection of
the ellipse.  The oblique-cut slope (slope_plane = tan θ) is fixed at the 1σ
value for all n_σ.  At n_σ = 1 the cuts exactly bound the 1σ ellipse.

### 3. Stan MAP optimisation

At each grid point the galaxy sample is filtered by the current cuts, then:

1. The sample mean and standard deviation of x are computed; x is
   standardised to x_std = (x − mean_x) / sd_x.
2. An OLS slope on the standardised sample initialises `slope_std`, clamped
   to the model's parameter bounds (−9·sd_x, −4·sd_x).  Clamping is necessary
   because a loose selection window can push OLS outside those bounds, causing
   Stan to fail at initialisation.
3. Data and init JSON files are written to a temporary directory and the
   compiled `tophat` executable is called in `optimize` (MAP) mode via
   `subprocess`.
4. The MLE slope s is read from the `slope` generated quantity in the Stan CSV
   output.  (`slope = slope_std / sd_x` converts from the standardised
   parameterisation automatically.)

### 4. Sweep

For each of the four cut parameters p ∈ {ŷ_min, ŷ_max, c₁, c₂}:

- Fix the other three at their 1σ values.
- Evaluate n_σ on a log-spaced grid from n_σ_min to n_σ_max (default 0.7–1.7,
  21 points).
- Run Stan MAP at each grid point (Section 3) and record (n_σ, cut value,
  slope, N_selected).
- Compute the numerical derivative ∂s/∂(n_σ) via `numpy.gradient` using the
  actual (non-uniform) n_σ coordinates as the second argument, giving correct
  central differences on the log-spaced grid.

### 5. Interpretation

**Stability plateau:** A region where ∂s/∂(n_σ) ≈ 0 indicates that the fitted
TFR slope is insensitive to the exact placement of that cut.  The 1σ ellipse
boundary (n_σ = 1, dashed vertical line) should ideally fall within this flat
region.

**Bound-hitting at tight cuts:** For n_σ ≲ 0.7 the MLE slope often saturates
at the Stan model's upper bound (−4).  This is not a numerical failure but a
physical effect: a very narrow selection window makes the selection-correction
term dominate the likelihood, the signal-to-correction ratio collapses, and
the optimiser finds no well-identified interior maximum.  Points at the
constraint boundary should be excluded from plateau identification.

**Monotone vs non-monotone profiles:** A monotone slope profile (∂s/∂(n_σ)
constant sign) with a flat plateau in the middle indicates the cut is
well-behaved.  A non-monotone profile with sign changes signals that multiple
regimes compete (e.g., at loose cuts the selection region extends into a
non-TFR population, reversing the bias direction).

### 6. Output

**`output/<run>/ellipse_sweep.json`** — sweep data for downstream use:

```json
{
  "<param>": {
    "n_sigma":          [float, ...],
    "cut_values":       [float, ...],
    "slopes":           [float or null, ...],
    "d_slope_d_nsigma": [float or null, ...]
  },
  ...
}
```

One entry per cut parameter (`haty_min`, `haty_max`, `intercept_plane`,
`intercept_plane2`).  Grid points where Stan returned no valid result are
stored as `null`.

**`output/<run>/ellipse_sweep.png`** — 2-row × 4-column figure:

- **Top row:** MLE slope s vs n_σ on a log x-axis for each cut parameter.
  A secondary top axis shows the corresponding cut value.  Dashed vertical
  line marks n_σ = 1.
- **Bottom row:** ∂s/∂(n_σ) vs n_σ (log x-axis).  Dashed horizontal line at
  zero; dashed vertical line at n_σ = 1.

A summary table is also printed to stdout reporting, at the grid point closest
to n_σ = 1, the cut value, MLE slope, derivative, and number of selected
galaxies for each parameter.

### Usage

```bash
python ellipse_sweep.py --source fullmocks --fits_file $FITS --run $RUN
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `fullmocks` | Data source: `fullmocks` or `DESI` |
| `--fits_file FILE` | auto | Path to FITS file; auto-detected from `--dir` if omitted |
| `--dir DIR` | `data/` | Directory searched for FITS files |
| `--run RUN` | required | Run name; reads `output/<run>/selection_ellipse.json` |
| `--exe EXE` | `tophat` | Path to compiled Stan tophat executable |
| `--n_sigma_min` | 0.7 | Lower end of n_σ grid |
| `--n_sigma_max` | 1.7 | Upper end of n_σ grid |
| `--n_sigma_n` | 21 | Number of log-spaced grid points |
| `--z_obs_min` | 0.03 | Minimum redshift cut |
| `--z_obs_max` | 0.10 | Maximum redshift cut |
| `--n_sweep_objects` | 10000 | Subsample size (0 = use all) |

## Step 3: Selection criteria

Algorithm: `selection_criteria.py`

The sweep profiles from Step 2 show that the MLE slope is stable (∂s/∂(n_σ) ≈ 0)
over a central plateau and diverges at large n_σ where non-TFR galaxies enter the
sample.  This step automatically identifies that plateau for each cut parameter,
chooses a value near its large-n_σ edge, and records the result as the final
selection cuts used in subsequent analysis.

### 1. Valid points

For each sweep parameter the following points are excluded before any plateau
analysis:

- slope is NaN (Stan failed or sample too small),
- derivative is NaN,
- slope is within `BOUND_TOL = 0.05` of the Stan hard bounds (−9 or −4) —
  these are bound-hitting points where the optimiser did not find a free interior
  maximum.

### 2. Two-level plateau detection (`choose_nsigma`)

The algorithm operates in two levels.

**Level 1 — broad plateau.**  A global threshold

$$\tau_1 = d_\mathrm{threshold} \times \max_i |\partial s / \partial n_\sigma|$$

is computed over all valid points.  Points with |∂s/∂(n_σ)| < τ₁ form the broad
plateau.  If fewer than two valid points exist the algorithm falls back to n_σ = 1.
If no broad plateau exists the closest valid point to n_σ = 1 is returned with
status `no_plateau`.

**Level 2 — group selection.**  The broad plateau is split into maximal contiguous
runs (adjacent grid indices).  The run containing the globally flattest valid point
(minimum |∂s/∂(n_σ)|) is selected as the candidate group.

**Boundary exception.**  When the global minimum |∂s/∂(n_σ)| falls on the
leftmost valid point the derivative profile is monotonically drifting — there is
no interior stable region.  In this case Levels 2–3 are skipped and the right edge
of the broad plateau group is used directly (Level 1 fallback).

**Level 3 — local "close to zero" criterion.**  Within the selected group a point
is considered close to zero if

$$|\partial s / \partial n_\sigma| < d_\mathrm{threshold} \times \mathrm{IQR}(\partial s / \partial n_\sigma)$$

where IQR is computed over the signed derivatives of the selected group.  This
criterion is relative to the width of the local derivative distribution rather
than its maximum, so it correctly identifies points that are genuinely near zero
even when the group contains some moderate outliers.  The close-to-zero points are
split into contiguous sub-runs and the **rightmost** sub-run is chosen — this gives
the loosest cut that is still demonstrably stable.

**Edge and chosen n_σ.**  Let the rightmost sub-run have last two members
n_σ_{prev} and n_σ_{edge}.  The chosen value is

$$n_{\sigma,\mathrm{chosen}} = n_{\sigma,\mathrm{prev}} + \mathrm{frac} \times (n_{\sigma,\mathrm{edge}} - n_{\sigma,\mathrm{prev}})$$

with `frac = 0.8` by default.  If the sub-run contains only one point,
n_σ_chosen = n_σ_edge.  The cut value is obtained by linear interpolation of
the sweep grid at n_σ_chosen.

If n_σ_edge equals the rightmost valid point the plateau may extend beyond the
sweep range; status `plateau_at_edge` is reported and a wider sweep is
recommended.

### 3. slope_plane

The oblique-cut slope is fixed at the value from `selection_ellipse.json`
throughout and is copied unchanged into the output.

### 4. Output

**`output/<run>/selection_criteria.json`** — final cut values:

```json
{
  "haty_min":         <float>,
  "haty_max":         <float>,
  "slope_plane":      <float>,
  "intercept_plane":  <float>,
  "intercept_plane2": <float>,
  "n_sigma_chosen": {
    "haty_min":         <float>,
    "haty_max":         <float>,
    "intercept_plane":  <float>,
    "intercept_plane2": <float>
  }
}
```

**`output/<run>/selection_criteria.png`** — 2-row × 4-column figure (same layout
as `ellipse_sweep.png`) with an added orange dashed vertical line at n_σ_chosen
in both the slope and derivative rows for each parameter.  Warning text appears in
the subplot title for non-`ok` status flags.

**`output/<run>/selection_criteria_data.png`** — galaxy scatter plot (identical
layout to `selection_ellipse.png`) with the selected cut lines overlaid in
deepskyblue (horizontal magnitude cuts) and limegreen (oblique plane cuts).
Galaxy colour encodes exp(−½χ²) computed from the saved core-component covariance,
serving as a proxy for P(core component) without refitting the GMM.  Produced only
when `--source` is provided.

### Usage

```bash
python selection_criteria.py --run $RUN

# Also produce the galaxy scatter overplot:
python selection_criteria.py --run $RUN --source fullmocks --fits_file $FITS
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run RUN` | required | Run name; reads/writes `output/<run>/` |
| `--d_threshold` | 0.3 | Plateau threshold as fraction of max\|∂s/∂n_σ\| (Level 1) and IQR (Level 3) |
| `--frac` | 0.8 | Interpolation fraction between preceding sub-run point and edge |
| `--source` | — | Data source for scatter overplot: `fullmocks` or `DESI` (omit to skip) |
| `--fits_file FILE` | auto | Path to FITS file; auto-detected from `--dir` if omitted |
| `--dir DIR` | `data/` | Directory searched for FITS files |
