# Sample Selection v2

## Introduction

The objective of this stage is to use sample selection to obtain a normal galaxy
sample with high purity and completeness.  Normal galaxies occupy a compact region
of the absolute-magnitude / rotation-velocity phase space and obey a Tully-Fisher
relation (TFR) with a specific slope.  A small fraction of other galaxies do not follow
the Tully-Fisher relation and for the most part occupy a different region of the
phase space.  An analysis of the full sample would yield a different slope with a
poorer quality of fit.

The procedure is to make a rough estimate of the TFR, apply that TFR to all
galaxies and identify the selection limits where the TFR breaks down, e.g. as expected due interloping
dwarf galaxies.

---

## Step 1: Estimating the core distribution
This step identifies a subsample from which to make the rough estimate of the TFR.
Starting with the full sample, z > 0.03 is required to be insensitive to the effect of peculiar velocities,
and a loose magnitude pre-filter −23 < y < −18 removes unphysical extremes.
The remaining subsample is fit to a Gaussian Mixture Model (GMM) to identify the core distribution
of normal galaxies.

The GMM fit yields the core mean **μ** and covariance **Σ**, which are saved for use
in Step 2.  There the 3σ ellipse sets the maximum and minimum $y$ values, the slope
of the parallel band, and the $y$-intercepts of the band edges, which intersect the
co-vertices of the ellipse.

### Algorithm: `selection_ellipse.py`

The script fits a 2-component GMM to the
observed (x, y) = (log₁₀(V / 100 km/s), R-band absolute magnitude) phase space,
identifies the TFR core component, and writes the core mean **μ** and covariance
**Σ** together with the derived 1σ selection cut parameters to
`output/<run>/selection_ellipse.json`.

### 1. Data loading

Galaxies are read from a FITS file (AbacusSummit mock or real survey). A loose pre-filter
(`--haty_min`, `--haty_max`, defaults −23 and −18) removes unphysical extremes
before fitting.  Each galaxy contributes four quantities:

| Symbol | FITS column | Definition |
|--------|-------------|-----------|
| x | `V_0p4R26` | log₁₀(V / 100 km/s), computed as log₁₀(`V_0p4R26` / 100) |
| y | `R_ABSMAG_SB26` | R-band absolute magnitude |
| σ_x | `V_0p4R26_ERR` | uncertainty in x, propagated as `V_0p4R26_ERR` / (`V_0p4R26` ln 10) |
| σ_y | `R_ABSMAG_SB26_ERR` | uncertainty in y |

### 2. Truncation detection

The survey is assumed to be truncated at:

- **x_hi = x.max()** — a hard upper limit on observed rotation velocity, and
- **y_lo = y.min()** — the bright (most-negative magnitude) completeness limit.

These bounds define the observable region R = {x ≤ x_hi, y ≥ y_lo}.

### Usage

```bash
python selection_ellipse.py --config $CONFIG
```

### Output

| File | Description |
|------|-------------|
| `output/<run>/selection_ellipse.json` | Core GMM parameters and derived 1σ cut values |
| `output/<run>/selection_ellipse.png` | Phase-space scatter with GMM ellipses and 1σ cuts |

---

## Step 2: MLE fit on the 3σ-ellipse selection

Algorithm: `select_v2.py` (MLE stage)

The 3σ ellipse derived from the GMM mean **μ** and covariance **Σ** is used
directly as the selection region.  Specifically, `_cuts_at_nsigma(μ, Σ, 3.0)`
produces the oblique cut boundaries and magnitude window corresponding to 3σ
along the minor axis and 3σ in the y-direction.  All galaxies falling inside this
parallelogram and satisfying z > 0.03 are passed to the Stan `tophat` model in `optimize` mode to obtain
MAP (MLE) estimates of the TFR parameters.

### Selection geometry

| Parameter | Meaning |
|-----------|---------|
| `haty_min`, `haty_max` | Bright- and dim-end magnitude limits at 3σ |
| `slope_plane` | Oblique cut slope = tan(major-axis angle of GMM) |
| `intercept_plane`, `intercept_plane2` | Intercepts of the two oblique boundaries at 3σ |

### Stan MAP optimisation

The data dictionary passed to Stan follows the standard `tophat` format.  The
call returns:

| Parameter | Meaning |
|-----------|---------|
| `slope` | TFR slope |
| `intercept.1` | TFR intercept |
| `sigma_int_x` | Intrinsic scatter in x (log velocity) |
| `sigma_int_y` | Intrinsic scatter in y (absolute magnitude) |

### Output

| File | Description |
|------|-------------|
| `output/<run>/select_v2_mle.json` | MLE TFR parameters from the 3σ selection |

---

## Step 3: Difference and pull plots over all objects

Algorithm: `select_v2.py` (diagnostic stage)

Using the MLE parameters from Step 2, the posterior predictive mean absolute
magnitude is computed for **every object in the full catalog** (not only those
inside the 3σ selection).  This gives a model prediction for each galaxy
regardless of whether it would survive any particular selection cut, making the
resulting plots useful for choosing the final magnitude window.

### Per-galaxy residual

$$\Delta M_i = \langle M_{\mathrm{abs},*} \rangle_i - M_{\mathrm{abs},i}^{\mathrm{obs}}$$

$$\sigma_{\Delta,i} = \sqrt{\sigma_{\mathrm{pred},i}^2 + \sigma_{y,i}^2}$$

where $\langle M_{\mathrm{abs},*} \rangle_i$ and $\sigma_{\mathrm{pred},i}$ come
from `ystar_pp_mean_sd_tophat_vectorized` (predict.py) evaluated at the MLE point.
$\Delta M_i$ is equivalent to the apparent-magnitude residual $m_{\mathrm{pred}} -
m_{\mathrm{obs}}$ because the distance modulus cancels.

### Binned pull profile

Galaxies are placed into **equal-occupancy bins** in M_abs (quantile edges,
default 20 bins) over the full catalog range.  In each bin:

$$\langle \Delta \rangle_w = \frac{\sum_i w_i \Delta_i}{\sum_i w_i}, \qquad
  \sigma_{\langle\Delta\rangle} = \frac{1}{\sqrt{\sum_i w_i}}, \qquad
  w_i = \sigma_{\Delta,i}^{-2}$$

$$\mathrm{pull} = \frac{\langle \Delta \rangle_w}{\sigma_{\langle\Delta\rangle}}$$

The pull profile is plotted as a bar chart against M_abs bin centre together with
a panel showing the weighted-mean residual ± uncertainty.  Flat pull near zero
across a magnitude range indicates that the model is consistent with the data
there; departures at the bright or faint ends indicate where the selection window
should be truncated.

### Output

| File | Description |
|------|-------------|
| `output/<run>/select_v2_pull.png` | Pull profile and weighted-mean residual vs M_abs for all objects |

---

## Step 4: Set fiducial selection parameters

Algorithm: `select_v2.py --set_fiducial`

Based on the pull profile from Step 3, the user chooses the magnitude window
(`haty_min`, `haty_max`).  The oblique cut boundaries are taken directly from
the 3σ GMM ellipse (same as the diagnostic step).  No Stan call is made.

### Usage

```bash
python select_v2.py --config $CONFIG \
    --set_fiducial --haty_min -22 --haty_max -19.5
```

### Output

| File | Description |
|------|-------------|
| `output/<run>/select_v2_fiducial.json` | Fiducial selection cuts for downstream scripts |

The fiducial JSON contains:

```json
{
  "haty_min":         <user-chosen bright-end limit>,
  "haty_max":         <user-chosen dim-end limit>,
  "slope_plane":      <GMM oblique slope>,
  "intercept_plane":  <lower oblique intercept at 3σ>,
  "intercept_plane2": <upper oblique intercept at 3σ>
}
```

`desi_data.py` reads all selection parameters from the config file passed via `--config`.

---

## CLI reference: `select_v2.py`

```bash
python select_v2.py --config $CONFIG [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config FILE` | — | Path to JSON config (e.g. `configs/dr1_v3.json`) |
| `--run RUN` | from config | Run name; reads `output/<run>/selection_ellipse.json` |
| `--fits_file FILE` | from config | Path to DESI FITS file |
| `--exe EXE` | `tophat` | Path to compiled tophat Stan binary |
| `--set_fiducial` | off | Write fiducial JSON from `--haty_min`/`--haty_max`; skip Stan |
| `--haty_min` | from config | Fiducial bright-end magnitude limit |
| `--haty_max` | from config | Fiducial dim-end magnitude limit |
| `--n_bins` | 20 | Number of equal-occupancy M_abs bins for pull plot |
| `--z_obs_min` | 0.03 | Minimum redshift for Stan MLE sample |

---

## Summary of Generated Files

| File | Script | Description |
|------|--------|-------------|
| `output/<run>/selection_ellipse.json` | `selection_ellipse.py` | Core GMM parameters and derived 1σ cut values |
| `output/<run>/selection_ellipse.png` | `selection_ellipse.py` | Phase-space scatter coloured by P(core component) with GMM ellipses |
| `output/<run>/select_v2_mle.json` | `select_v2.py` | MLE TFR parameters from 3σ selection: slope, intercept, sigma_int_x, sigma_int_y |
| `output/<run>/select_v2_pull.png` | `select_v2.py` | Pull profile and weighted-mean residual vs M_abs for all catalog objects |
| `output/<run>/select_v2_fiducial.json` | `select_v2.py --set_fiducial` | Fiducial selection cuts chosen in Step 4: haty_min, haty_max, slope_plane, intercept_plane, intercept_plane2 |
