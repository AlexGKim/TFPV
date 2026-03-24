# Sample Selection v2

## Introduction

The objective of this stage is to use sample selection to obtain a normal galaxy
sample with high purity and completeness.  Normal galaxies occupy a compact region
of the absolute-magnitude / rotation-velocity phase space and obey a Tully-Fisher
relation with a specific slope.  A small fraction of other galaxies do not follow
the Tully-Fisher relation and for the most part occupy a different region of the
phase space.  An analysis of the full sample would yield a different slope with a
poorer quality of fit.

This v2 procedure replaces the costly 3-D grid search of the original pipeline
(Steps 2–3 of `Selection.md`) with a direct approach: the 3σ ellipse from the
GMM fit defines the initial selection, the resulting TFR parameters are evaluated
over the full catalog to produce a diagnostic pull profile, and the user then
sets the final selection window by hand.

---

## Step 1: Estimating the core distribution

Algorithm: `selection_ellipse.py`

Identical to Step 1 of `Selection.md`.  The script fits a 2-component GMM to the
observed (x, y) = (log₁₀(V / 100 km/s), R-band absolute magnitude) phase space,
identifies the TFR core component, and writes the core mean **μ** and covariance
**Σ** together with the derived 1σ selection cut parameters to
`output/<run>/selection_ellipse.json`.

See `Selection.md § Step 1` for the full description of the GMM fit and derived
geometry.

### Usage

```bash
python selection_ellipse.py --file $FITS --run $RUN --source DESI --z_obs_min 0.03
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
parallelogram are passed to the Stan `tophat` model in `optimize` mode to obtain
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

## Step 3: Set fiducial selection parameters

Algorithm: `set_fiducial.py`

Based on the pull profile from Step 2, the user chooses the perpendicular cut
width (in sigma units) and the magnitude window (`haty_min`, `haty_max`).  The
script prompts interactively, computes the oblique intercepts from the GMM
ellipse, and writes `select_v2_fiducial.json` for use by `desi_data.py`.

### Usage

```bash
python set_fiducial.py --run $RUN
```

The script prints the 1σ reference values as a guide, then prompts:

```
Enter n_sigma_perp (perpendicular cut width in sigma units): 3
Enter haty_min (bright-end magnitude limit, e.g. -22): -22
Enter haty_max (dim-end   magnitude limit, e.g. -19.5): -19.5
Enter z_obs_min (minimum redshift) [default 0.03]:
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
  "intercept_plane":  <lower oblique intercept at n_sigma_perp>,
  "intercept_plane2": <upper oblique intercept at n_sigma_perp>,
  "n_sigma_perp":     <perpendicular cut width in sigma>,
  "z_obs_min":        <minimum redshift>
}
```

`desi_data.py` automatically loads `select_v2_fiducial.json` (falling back to
`mag_split_fiducial.json` from the old pipeline if the v2 file is absent).

---

## CLI reference: `select_v2.py`

```bash
python select_v2.py --run $RUN --fits_file $FITS --exe ./tophat [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run RUN` | required | Run name; reads `output/<run>/selection_ellipse.json` |
| `--fits_file FILE` | required | Path to DESI FITS file |
| `--exe EXE` | `tophat` | Path to compiled tophat Stan binary |
| `--haty_min` | 3σ ellipse value | Override bright-end magnitude limit (Step 4) |
| `--haty_max` | 3σ ellipse value | Override dim-end magnitude limit (Step 4) |
| `--n_bins` | 20 | Number of equal-occupancy M_abs bins for pull plot |
| `--z_obs_min` | none | Optional minimum redshift cut |
| `--z_obs_max` | none | Optional maximum redshift cut |

---

## Summary of Generated Files

| File | Script | Description |
|------|--------|-------------|
| `output/<run>/selection_ellipse.json` | `selection_ellipse.py` | Core GMM parameters and derived 1σ cut values |
| `output/<run>/selection_ellipse.png` | `selection_ellipse.py` | Phase-space scatter coloured by P(core component) with GMM ellipses |
| `output/<run>/select_v2_mle.json` | `select_v2.py` | MLE TFR parameters from 3σ selection: slope, intercept, sigma_int_x, sigma_int_y |
| `output/<run>/select_v2_pull.png` | `select_v2.py` | Pull profile and weighted-mean residual vs M_abs for all catalog objects |
| `output/<run>/select_v2_fiducial.json` | `set_fiducial.py` | Fiducial selection cuts chosen in Step 3: haty_min, haty_max, slope_plane, intercept_plane, intercept_plane2, n_sigma_perp, z_obs_min |
