"""
selection_ellipse.py — Fit a 2-component GMM to (x, y) phase space and draw
confidence ellipses to estimate the TFR core selection boundary.

The fit accounts for:
  - Per-galaxy measurement noise: effective covariance per point is
    Sigma_k + diag(sigma_x_i^2, sigma_y_i^2).
  - Rectangular truncation at x <= x_hi = x.max() and y >= y_lo = y.min()
    (bright/more-negative end), normalising each component by
    P(X1 <= x_hi, X2 >= y_lo) evaluated at the mean noise level.

Usage:
    python selection_ellipse.py \
        --file data/TF_extended_AbacusSummit_base_c000_ph000_r001_z0.11.fits \
        --run c000_ph000_r001
"""

import argparse
import json
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvnorm
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(fits_file, haty_min, haty_max, source="fullmocks", z_obs_min=None, z_obs_max=None):
    print(f"Reading FITS file: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # type: ignore[union-attr]
        total_rows = len(data)
        names = set(data.dtype.names or ())

        if source == "fullmocks":
            main_mask = np.asarray(data["MAIN"], dtype=bool)
            data_main = data[main_mask]
            print(f"  Total rows: {total_rows}  |  MAIN=True: {np.sum(main_mask)}")

            logvrot     = np.asarray(data_main["LOGVROT"],           dtype=float)
            logvrot_err = np.asarray(data_main["LOGVROT_ERR"],       dtype=float)
            absmag      = np.asarray(data_main["R_ABSMAG_SB26"],     dtype=float)
            absmag_err  = np.asarray(data_main["R_ABSMAG_SB26_ERR"], dtype=float)

            x_raw   = logvrot - 2.0
            sigma_x = logvrot_err
            y_raw   = absmag
            sigma_y = absmag_err

            # Redshift for fullmocks
            zobs = None
            for col in ("ZOBS", "Z_OBS", "zobs"):
                if col in names:
                    zobs = np.asarray(data_main[col], dtype=float)
                    break

        else:  # DESI
            print(f"  Total rows: {total_rows}  (no MAIN filter for DESI source)")

            V     = np.asarray(data["V_0p4R26"],     dtype=float)
            V_err = np.asarray(data["V_0p4R26_ERR"], dtype=float)
            absmag = np.asarray(data["R_ABSMAG_SB26"], dtype=float)

            if "R_ABSMAG_SB26_ERR" in names:
                absmag_err = np.asarray(data["R_ABSMAG_SB26_ERR"], dtype=float)
            else:
                print("  Warning: R_ABSMAG_SB26_ERR absent; falling back to R_MAG_SB26_ERR")
                absmag_err = np.asarray(data["R_MAG_SB26_ERR"], dtype=float)

            x_raw   = np.log10(np.where(V > 0, V, np.nan) / 100.0)
            sigma_x = V_err / (np.where(V > 0, V, np.nan) * np.log(10.0))
            y_raw   = absmag
            sigma_y = absmag_err

            # Redshift for DESI
            zobs = None
            for col in ("Z_DESI", "Z_DESI_CMB", "ZOBS", "Z_OBS", "zobs", "Z", "ZHELIO"):
                if col in names:
                    zobs = np.asarray(data[col], dtype=float)
                    break

    valid = (
        np.isfinite(x_raw)
        & np.isfinite(sigma_x)
        & np.isfinite(y_raw)
        & np.isfinite(sigma_y)
        & (sigma_x > 0)
        & (sigma_y >= 0)
        & (y_raw >= haty_min)
        & (y_raw <= haty_max)
    )

    if zobs is not None:
        finite_mask = np.isfinite(zobs)
        if z_obs_min is not None:
            valid &= finite_mask & (zobs >= z_obs_min)
        if z_obs_max is not None:
            valid &= finite_mask & (zobs <= z_obs_max)

        lo = f"z >= {z_obs_min}" if z_obs_min is not None else None
        hi = f"z <= {z_obs_max}" if z_obs_max is not None else None
        bounds = " and ".join(filter(None, [lo, hi]))
        if bounds:
            print(f"  Redshift cut {bounds} applied")


    print(f"  Valid rows after validity + pre-filter: {valid.sum()}")
    return x_raw[valid], y_raw[valid], sigma_x[valid], sigma_y[valid]


# ---------------------------------------------------------------------------
# Vectorised per-point bivariate normal pdf
# ---------------------------------------------------------------------------

def _bvn_pdf(xy, mu, Sigma_k, sigma2_x, sigma2_y):
    """Bivariate normal pdf with per-point effective covariance.

    Sigma_eff_i = Sigma_k + diag(sigma2_x[i], sigma2_y[i])

    Returns array of shape (N,).
    """
    # Effective variance components (all shape (N,))
    a = Sigma_k[0, 0] + sigma2_x          # Var(x)_eff
    b = Sigma_k[0, 1]                      # Cov(x,y)  — same for all i
    d = Sigma_k[1, 1] + sigma2_y          # Var(y)_eff

    det = a * d - b * b
    det = np.maximum(det, 1e-300)

    dx = xy[:, 0] - mu[0]
    dy = xy[:, 1] - mu[1]

    quad = (d * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------

def detect_truncation(x, y):
    """Return (x_hi, y_lo): data extremes defining the truncation boundary.

    The survey is truncated at high velocity (x_hi = x.max()) and at the
    bright end (y_lo = y.min(), the more-negative magnitude limit).
    """
    return float(x.max()), float(y.min())


def _rect_prob(mu, Sigma, x_hi, y_lo):
    """P(X1 <= x_hi, X2 >= y_lo) for N(mu, Sigma).

    = P(X1 <= x_hi) - P(X1 <= x_hi, X2 <= y_lo)
    """
    from scipy.stats import norm
    rv    = mvnorm(mean=mu, cov=Sigma)
    p_x   = norm.cdf(x_hi, loc=mu[0], scale=np.sqrt(Sigma[0, 0]))
    p_lo  = rv.cdf([x_hi, y_lo])
    return float(np.clip(p_x - p_lo, 1e-10, 1.0))


# ---------------------------------------------------------------------------
# Truncated noisy GMM optimisation
# ---------------------------------------------------------------------------

def _unpack(theta):
    """Flat parameter vector → (w1, mu1, Sigma1, mu2, Sigma2)."""
    w1 = 1.0 / (1.0 + np.exp(-theta[0]))
    mu1 = theta[1:3]
    L1 = np.array([[np.exp(theta[3]), 0.0],
                   [theta[4],         np.exp(theta[5])]])
    mu2 = theta[6:8]
    L2 = np.array([[np.exp(theta[8]), 0.0],
                   [theta[9],         np.exp(theta[10])]])
    return w1, mu1, L1 @ L1.T, mu2, L2 @ L2.T


def _pack(gmm):
    """sklearn GMM → flat parameter vector."""
    w1  = float(gmm.weights_[0])
    mu1, mu2 = gmm.means_[0], gmm.means_[1]
    L1  = np.linalg.cholesky(gmm.covariances_[0])
    L2  = np.linalg.cholesky(gmm.covariances_[1])
    return np.array([
        np.log(w1 / (1.0 - w1)),
        mu1[0], mu1[1], np.log(L1[0, 0]), L1[1, 0], np.log(L1[1, 1]),
        mu2[0], mu2[1], np.log(L2[0, 0]), L2[1, 0], np.log(L2[1, 1]),
    ])


def _neg_log_lik(theta, xy, sigma2_x, sigma2_y,
                 x_hi, y_lo, mean_sigma2_x, mean_sigma2_y):
    """Negative truncated noisy GMM log-likelihood.

    Per-point likelihood uses Sigma_k + diag(sigma2_x[i], sigma2_y[i]).
    Truncation normalisation Z_k uses the mean noise level.
    """
    try:
        w1, mu1, S1, mu2, S2 = _unpack(theta)
    except np.linalg.LinAlgError:
        return 1e10

    w2 = 1.0 - w1

    # Truncation normalisation at mean noise level
    S1_mn = S1 + np.diag([mean_sigma2_x, mean_sigma2_y])
    S2_mn = S2 + np.diag([mean_sigma2_x, mean_sigma2_y])
    Z1 = _rect_prob(mu1, S1_mn, x_hi, y_lo)
    Z2 = _rect_prob(mu2, S2_mn, x_hi, y_lo)

    # Per-point density (noise-convolved)
    p = (w1 * _bvn_pdf(xy, mu1, S1, sigma2_x, sigma2_y) / Z1 +
         w2 * _bvn_pdf(xy, mu2, S2, sigma2_x, sigma2_y) / Z2)

    return -float(np.sum(np.log(np.clip(p, 1e-300, None))))


class _TruncGMM:
    """Truncated noisy 2-component GMM result with sklearn-compatible interface."""

    def __init__(self, weights, means, covs, x_hi, y_lo,
                 sigma_x, sigma_y, mean_sigma2_x, mean_sigma2_y):
        self.weights_     = np.asarray(weights)
        self.means_       = np.asarray(means)
        self.covariances_ = np.asarray(covs)
        self._x_hi = x_hi
        self._y_lo = y_lo
        self._sigma2_x = sigma_x ** 2
        self._sigma2_y = sigma_y ** 2
        self._mean_s2x = mean_sigma2_x
        self._mean_s2y = mean_sigma2_y

    def predict_proba(self, xy):
        """Posterior component probabilities, accounting for per-point noise."""
        cols = []
        for k in range(2):
            mu_k = self.means_[k]
            S_k  = self.covariances_[k]
            S_mn = S_k + np.diag([self._mean_s2x, self._mean_s2y])
            Z_k  = _rect_prob(mu_k, S_mn, self._x_hi, self._y_lo)
            cols.append(
                self.weights_[k]
                * _bvn_pdf(xy, mu_k, S_k, self._sigma2_x, self._sigma2_y)
                / Z_k
            )
        probs = np.column_stack(cols)
        return probs / probs.sum(axis=1, keepdims=True)


def fit_truncated_gmm(x, y, sigma_x, sigma_y, n_init, x_hi, y_lo):
    """Fit a 2-component GMM truncated to (-∞, x_hi] × [y_lo, +∞),
    accounting for per-galaxy measurement noise.

    Starts from a standard sklearn GMM (on observed positions only),
    then maximises the truncated noisy log-likelihood with Powell.
    """
    xy       = np.column_stack([x, y])
    sigma2_x = sigma_x ** 2
    sigma2_y = sigma_y ** 2
    msig2_x  = float(np.mean(sigma2_x))
    msig2_y  = float(np.mean(sigma2_y))

    print("  Initial sklearn GMM fit …")
    init_gmm = GaussianMixture(
        n_components=2, covariance_type="full",
        n_init=n_init, random_state=0,
    )
    init_gmm.fit(xy)
    theta0 = _pack(init_gmm)

    print(f"  Optimising noise+truncation-corrected log-likelihood "
          f"(x_hi={x_hi:.4f}, y_lo={y_lo:.4f}) …")
    res = minimize(
        _neg_log_lik, theta0,
        args=(xy, sigma2_x, sigma2_y, x_hi, y_lo, msig2_x, msig2_y),
        method="Powell",
        options={"maxiter": 10000, "ftol": 1e-9},
    )
    if not res.success:
        print(f"  Warning: optimiser did not converge — {res.message}")

    w1, mu1, S1, mu2, S2 = _unpack(res.x)
    return _TruncGMM(
        weights=[w1, 1.0 - w1],
        means=[mu1, mu2],
        covs=[S1, S2],
        x_hi=x_hi, y_lo=y_lo,
        sigma_x=sigma_x, sigma_y=sigma_y,
        mean_sigma2_x=msig2_x, mean_sigma2_y=msig2_y,
    )


def core_component(gmm):
    """Return index of the component with the smaller covariance determinant."""
    dets = [np.linalg.det(C) for C in gmm.covariances_]
    return int(np.argmin(dets))


# ---------------------------------------------------------------------------
# Ellipse geometry
# ---------------------------------------------------------------------------

def ellipse_params(sigma):
    """Return (semi_axes, angle_deg) for the 1-sigma ellipse of covariance Sigma.

    semi_axes = [sigma_minor, sigma_major]  (ascending eigenvalues).
    angle is counter-clockwise rotation of the major axis from the x-axis.
    """
    vals, vecs = np.linalg.eigh(sigma)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    return np.sqrt(vals), angle


def derived_cuts(mu, sigma):
    """Derive selection boundary parameters from the 1-sigma ellipse.

    haty_min / haty_max  — extreme y-values of the 1σ ellipse (horizontal cuts).
    slope                — dy/dx of major axis (TFR orientation).
    intercept1/2         — oblique lines with that slope through the ± ends
                           of the semi-minor axis.

    Returns (haty_min, haty_max, slope, intercept1, intercept2).
    """
    vals, vecs = np.linalg.eigh(sigma)
    sigma_minor = np.sqrt(vals[0])
    sigma_major = np.sqrt(vals[1])
    angle_rad   = np.arctan2(vecs[1, -1], vecs[0, -1])

    # Extreme y-values reached by the 1σ ellipse
    y_extent = np.sqrt(sigma_major**2 * np.sin(angle_rad)**2
                       + sigma_minor**2 * np.cos(angle_rad)**2)
    haty_min = float(mu[1] - y_extent)
    haty_max = float(mu[1] + y_extent)

    # Slope from major-axis orientation
    slope = float(np.tan(angle_rad))

    # Oblique lines through ± minor-axis endpoints
    minor_vec  = vecs[:, 0]                      # unit vector along minor axis
    p1 = mu + sigma_minor * minor_vec
    p2 = mu - sigma_minor * minor_vec
    intercept1 = float(p1[1] - slope * p1[0])
    intercept2 = float(p2[1] - slope * p2[0])

    return haty_min, haty_max, slope, intercept1, intercept2


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def make_plot(x, y, gmm, core_idx, out_path, x_hi, y_lo):
    xy        = np.column_stack([x, y])
    core_prob = gmm.predict_proba(xy)[:, core_idx]

    mu        = gmm.means_[core_idx]
    sigma     = gmm.covariances_[core_idx]
    semi_axes, angle = ellipse_params(sigma)

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        x, y,
        c=core_prob,
        cmap="coolwarm",
        s=1,
        alpha=0.4,
        vmin=0, vmax=1,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="P(core component)")

    sigma_colors = ["gold", "red"]
    sigma_styles = ["-", ":"]
    sigma_lws    = [1.5, 2.5]
    for n_sigma, color, ls, lw in zip([1, 3], sigma_colors, sigma_styles, sigma_lws):
        ell = mpatches.Ellipse(
            mu,
            width=2 * n_sigma * semi_axes[1],   # major axis full diameter
            height=2 * n_sigma * semi_axes[0],  # minor axis full diameter
            angle=angle,
            edgecolor=color,
            facecolor="none",
            linewidth=lw,
            linestyle=ls,
            label=f"{n_sigma}σ ellipse",
            zorder=5,
        )
        ax.add_patch(ell)

    # Truncation boundary used in fit
    ax.axvline(x_hi, color="white", lw=1.0, ls="-")
    ax.axhline(y_lo, color="white", lw=1.0, ls="-")

    # Selection boundaries derived from the 3σ ellipse (bold solid)
    cuts3 = _cuts_at_nsigma(mu, sigma, 3.0)
    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 300)
    ax.axhline(cuts3["haty_max"], color="cyan", lw=2.0, ls="-",
               label=f"haty_max={cuts3['haty_max']:.2f}")
    ax.axhline(cuts3["haty_min"], color="cyan", lw=2.0, ls="-",
               label=f"haty_min={cuts3['haty_min']:.2f}")
    ax.plot(x_line, cuts3["slope_plane"] * x_line + cuts3["intercept_plane"],
            color="lime", lw=2.0, ls="-",
            label=f"plane1: b={cuts3['intercept_plane']:.2f}")
    ax.plot(x_line, cuts3["slope_plane"] * x_line + cuts3["intercept_plane2"],
            color="lime", lw=2.0, ls="-",
            label=f"plane2: b={cuts3['intercept_plane2']:.2f}")

    ax.set_xlabel(r"$x = \log_{10}(V/100\,\mathrm{km\,s}^{-1})$")
    ax.set_ylabel(r"$y = R\mathrm{-band\ absolute\ magnitude}$")

    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(x.min(), x.max())
    y_pad = 0.3
    ax.set_ylim(min(y.min(), cuts3["haty_min"]) - y_pad,
                max(y.max(), cuts3["haty_max"]) + y_pad)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# MLE at n-sigma
# ---------------------------------------------------------------------------

def _cuts_at_nsigma(mu, sigma, n_sigma):
    """Compute selection cut parameters for a GMM ellipse scaled by n_sigma.

    Returns a dict with keys: haty_min, haty_max, slope_plane,
    intercept_plane, intercept_plane2.
    """
    vals, vecs = np.linalg.eigh(sigma)
    sigma_minor = np.sqrt(vals[0])
    sigma_major = np.sqrt(vals[1])
    angle_rad   = np.arctan2(vecs[1, -1], vecs[0, -1])

    y_extent = np.sqrt(sigma_major**2 * np.sin(angle_rad)**2
                       + sigma_minor**2 * np.cos(angle_rad)**2)
    haty_min = float(mu[1] - n_sigma * y_extent)
    haty_max = float(mu[1] + n_sigma * y_extent)
    slope    = float(np.tan(angle_rad))

    minor_vec = vecs[:, 0]
    p1 = mu + n_sigma * sigma_minor * minor_vec
    p2 = mu - n_sigma * sigma_minor * minor_vec
    i1 = float(p1[1] - slope * p1[0])
    i2 = float(p2[1] - slope * p2[0])

    return dict(
        haty_min=haty_min,
        haty_max=haty_max,
        slope_plane=slope,
        intercept_plane=min(i1, i2),
        intercept_plane2=max(i1, i2),
    )


def _build_stan_dicts_from_arrays(x, sx, y, sy, cuts):
    """Apply selection cuts to raw arrays and build Stan data/init dicts.

    Returns (data_dict, init_dict), or (None, None) if the sample is too
    small or the cuts are geometrically invalid.
    """
    haty_min = cuts["haty_min"]
    haty_max = cuts["haty_max"]
    slope_p  = cuts["slope_plane"]
    ip       = cuts["intercept_plane"]
    ip2      = cuts["intercept_plane2"]

    if haty_min >= haty_max or ip >= ip2:
        return None, None

    mask = (y > haty_min) & (y < haty_max)
    lb = np.maximum(haty_min, slope_p * x + ip)
    ub = np.minimum(haty_max, slope_p * x + ip2)
    mask &= (y >= lb) & (y <= ub)

    x_sel = x[mask]; sx_sel = sx[mask]
    y_sel = y[mask]; sy_sel = sy[mask]

    if len(x_sel) < 30:
        return None, None

    mean_x = float(np.mean(x_sel))
    sd_x   = float(np.std(x_sel, ddof=1))
    if sd_x < 1e-6:
        return None, None

    x_std = (x_sel - mean_x) / sd_x
    slope_std, intercept_std = np.polyfit(x_std, y_sel, 1)
    slope_std = float(np.clip(slope_std, -9.0 * sd_x + 1e-4, -4.0 * sd_x - 1e-4))

    data_dict = {
        "N_bins":           1,
        "N_total":          len(x_sel),
        "x":                x_sel.tolist(),
        "sigma_x":          sx_sel.tolist(),
        "y":                y_sel.tolist(),
        "sigma_y":          sy_sel.tolist(),
        "haty_min":         float(haty_min),
        "haty_max":         float(haty_max),
        "y_min":            float(haty_min) - 0.5,
        "y_max":            float(haty_max) + 1.0,
        "slope_plane":      float(slope_p),
        "intercept_plane":  float(ip),
        "intercept_plane2": float(ip2),
    }
    init_dict = {
        "slope_std":     float(slope_std),
        "intercept_std": [float(intercept_std)],
        "sigma_int_x":   0.1,
        "sigma_int_y":   0.1,
    }
    return data_dict, init_dict


def _run_stan_mle_full(data_dict, init_dict, exe_file, tmp_dir):
    """Run Stan optimize and return all MLE parameters as a dict, or None on failure."""
    input_path  = os.path.join(tmp_dir, "input.json")
    init_path   = os.path.join(tmp_dir, "init.json")
    output_path = os.path.join(tmp_dir, "optimize.csv")

    with open(input_path, "w") as f:
        json.dump(data_dict, f)
    with open(init_path, "w") as f:
        json.dump(init_dict, f)

    result = subprocess.run(
        [exe_file, "optimize",
         "data",   f"file={input_path}",
         f"init={init_path}",
         "output", f"file={output_path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None

    try:
        with open(output_path) as f:
            lines = [l.strip() for l in f
                     if not l.startswith("#") and l.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        values = lines[1].split(",")
        return {k: float(v) for k, v in zip(header, values)}
    except (KeyError, ValueError, FileNotFoundError):
        return None


def mle_at_nsigma(x, y, sigma_x, sigma_y, mu, cov, n_sigma, exe_file,
                  run_dir, x_hi, y_lo, gmm, core_idx):
    """Fit TFR MLE within n-sigma selection region derived from GMM ellipse.

    Saves output/<run>/mle_nsigma.json and output/<run>/mle_nsigma.png.
    Returns the result dict, or None on failure.
    """
    cuts = _cuts_at_nsigma(mu, cov, n_sigma)
    data_dict, init_dict = _build_stan_dicts_from_arrays(
        x, sigma_x, y, sigma_y, cuts)
    if data_dict is None:
        print(f"Warning: too few galaxies (<30) in {n_sigma}σ selection — skipping MLE")
        return None

    print(f"\nRunning Stan MLE at {n_sigma}σ (N={data_dict['N_total']}) …")

    # Resolve exe to absolute path
    exe = exe_file
    if not os.path.isabs(exe):
        if os.path.exists(exe):
            exe = os.path.abspath(exe)
        else:
            candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), exe)
            if os.path.exists(candidate):
                exe = candidate

    with tempfile.TemporaryDirectory() as tmp_dir_path:
        mle_params = _run_stan_mle_full(data_dict, init_dict, exe, tmp_dir_path)

    if mle_params is None:
        print("Warning: Stan optimize failed")
        return None

    mle_slope = mle_params.get("slope")
    if mle_slope is None:
        print("Warning: 'slope' not found in Stan optimize output")
        return None

    # Use intercept from CSV if present; otherwise compute from data centroid
    if "intercept.1" in mle_params:
        mle_intercept = mle_params["intercept.1"]
    else:
        slope_p = cuts["slope_plane"]
        ip      = cuts["intercept_plane"]
        ip2     = cuts["intercept_plane2"]
        mask = (y > cuts["haty_min"]) & (y < cuts["haty_max"])
        lb = np.maximum(cuts["haty_min"], slope_p * x + ip)
        ub = np.minimum(cuts["haty_max"], slope_p * x + ip2)
        mask &= (y >= lb) & (y <= ub)
        mle_intercept = float(np.mean(y[mask]) - mle_slope * np.mean(x[mask]))

    result = {
        "n_sigma":          float(n_sigma),
        "N":                int(data_dict["N_total"]),
        "mle_slope":        float(mle_slope),
        "mle_intercept":    float(mle_intercept),
        "haty_min":         cuts["haty_min"],
        "haty_max":         cuts["haty_max"],
        "slope_plane":      cuts["slope_plane"],
        "intercept_plane":  cuts["intercept_plane"],
        "intercept_plane2": cuts["intercept_plane2"],
    }

    json_path = os.path.join(run_dir, "mle_nsigma.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  MLE slope={mle_slope:.4f}  intercept={mle_intercept:.4f}  N={result['N']}")
    print(f"  Saved: {json_path}")

    out_path = os.path.join(run_dir, "mle_nsigma.png")
    make_mle_plot(x, y, gmm, core_idx, cuts, mle_slope, mle_intercept,
                  n_sigma, out_path, x_hi, y_lo)
    return result


def make_mle_plot(x, y, gmm, core_idx, cuts, mle_slope, mle_intercept,
                  n_sigma, out_path, x_hi, y_lo):
    """Scatter + GMM ellipses + n-sigma selection boundary + MLE line."""
    xy        = np.column_stack([x, y])
    core_prob = gmm.predict_proba(xy)[:, core_idx]

    mu        = gmm.means_[core_idx]
    sigma     = gmm.covariances_[core_idx]
    semi_axes, angle = ellipse_params(sigma)

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        x, y,
        c=core_prob,
        cmap="coolwarm",
        s=1,
        alpha=0.4,
        vmin=0, vmax=1,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="P(core component)")

    # 1σ/2σ/3σ ellipses for context
    sigma_colors = ["gold", "orange", "red"]
    sigma_styles = ["-", "--", ":"]
    for n, color, ls in zip([1, 2, 3], sigma_colors, sigma_styles):
        ell = mpatches.Ellipse(
            mu,
            width=2 * n * semi_axes[1],
            height=2 * n * semi_axes[0],
            angle=angle,
            edgecolor=color,
            facecolor="none",
            linewidth=1.5,
            linestyle=ls,
            label=f"{n}σ ellipse",
            zorder=5,
        )
        ax.add_patch(ell)

    # Truncation boundary
    ax.axvline(x_hi, color="white", lw=1.0, ls="-",
               label=f"x_trunc={x_hi:.3f}")
    ax.axhline(y_lo, color="white", lw=1.0, ls="-",
               label=f"y_trunc={y_lo:.2f}")

    # n-sigma selection boundary (white, thick)
    haty_min = cuts["haty_min"]
    haty_max = cuts["haty_max"]
    slope_p  = cuts["slope_plane"]
    ip       = cuts["intercept_plane"]
    ip2      = cuts["intercept_plane2"]

    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 300)
    ax.axhline(haty_max, color="white", lw=2.0, ls="--",
               label=f"haty_max={haty_max:.2f}")
    ax.axhline(haty_min, color="white", lw=2.0, ls=":",
               label=f"haty_min={haty_min:.2f}")
    ax.plot(x_line, slope_p * x_line + ip,
            color="white", lw=2.0, ls="--",
            label=f"plane1: slope={slope_p:.2f}, b={ip:.2f}")
    ax.plot(x_line, slope_p * x_line + ip2,
            color="white", lw=2.0, ls=":",
            label=f"plane2: slope={slope_p:.2f}, b={ip2:.2f}")

    # MLE line
    ax.plot(x_line, mle_slope * x_line + mle_intercept,
            color="red", lw=2.0,
            label=f"MLE slope={mle_slope:.3f}")

    ax.set_xlabel(r"$x = \log_{10}(V/100\,\mathrm{km\,s}^{-1})$")
    ax.set_ylabel(r"$y = R\mathrm{-band\ absolute\ magnitude}$")
    ax.set_title(f"TFR core ellipses + MLE at {n_sigma:.1f}σ selection")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit noise+truncation-corrected GMM to TFR (x,y) "
                    "and draw selection ellipses."
    )
    parser.add_argument("--file",     required=True,  help="Path to FITS file")
    parser.add_argument("--run",      required=True,  help="Output run directory under output/")
    parser.add_argument("--source",   choices=["fullmocks", "DESI"], default="fullmocks",
                        help="Data source: fullmocks (default) or DESI")
    parser.add_argument("--haty_min", type=float, default=-23.0,
                        help="Loose lower magnitude pre-filter (default: -23.0)")
    parser.add_argument("--haty_max", type=float, default=-18.0,
                        help="Loose upper magnitude pre-filter (default: -18.0)")
    parser.add_argument("--n_init",   type=int,   default=20,
                        help="GMM random restarts for initialisation (default: 20)")
    parser.add_argument("--z_obs_min", type=float, default=0.03,
                        help="Minimum redshift cut (default: 0.03)")
    parser.add_argument("--z_obs_max", type=float, default=0.1,
                        help="Maximum redshift cut (default: 0.1)")    
    parser.add_argument("--exe",      default=None,
                        help="Path to compiled Stan tophat executable; if given, run MLE at --n_sigma")
    parser.add_argument("--n_sigma",  type=float, default=3.0,
                        help="Sigma scaling for MLE selection region (default: 3.0)")
    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # Load data (now includes sigma_x, sigma_y)
    x, y, sigma_x, sigma_y = load_data(args.file, args.haty_min, args.haty_max, args.source,
                                        z_obs_min=args.z_obs_min, z_obs_max=args.z_obs_max)

    # Detect truncation from data extremes
    x_hi, y_lo = detect_truncation(x, y)
    print(f"  Truncation bounds from data: x_hi={x_hi:.4f}, y_lo={y_lo:.4f}")

    # Fit noise + truncation-corrected GMM
    print(f"Fitting noise+truncation-corrected 2-component GMM (n_init={args.n_init}) …")
    gmm = fit_truncated_gmm(x, y, sigma_x, sigma_y, args.n_init, x_hi, y_lo)
    core = core_component(gmm)

    mu        = gmm.means_[core]
    sigma     = gmm.covariances_[core]
    semi_axes, angle = ellipse_params(sigma)
    weight    = gmm.weights_[core]

    haty_min, haty_max, slope, intercept1, intercept2 = derived_cuts(mu, sigma)

    print("\n--- Core component (noise + truncation-corrected) ---")
    print(f"  Weight : {weight:.4f}")
    print(f"  Mean   : x={mu[0]:.4f}, y={mu[1]:.4f}")
    print(f"  Semi-axes (σ_minor, σ_major): {semi_axes[0]:.4f}, {semi_axes[1]:.4f}")
    print(f"  Rotation angle: {angle:.2f} deg")
    print("\n--- Derived selection cuts ---")
    print(f"  haty_min={haty_min:.4f}  haty_max={haty_max:.4f}")
    print(f"  slope_plane={slope:.4f}")
    print(f"  intercept_plane={intercept1:.4f}  intercept_plane2={intercept2:.4f}")

    # Save core Gaussian fit
    gmm_path = os.path.join(run_dir, "selection_ellipse.json")
    gmm_out = {
        "weight":       float(weight),
        "mean":         mu.tolist(),
        "covariance":   sigma.tolist(),
        "semi_axes":    semi_axes.tolist(),
        "angle_deg":    float(angle),
        "x_trunc":      x_hi,
        "y_trunc":      y_lo,
        "haty_min":     haty_min,
        "haty_max":     haty_max,
        "slope_plane":  slope,
        "intercept_plane":  intercept1,
        "intercept_plane2": intercept2,
    }
    with open(gmm_path, "w") as f:
        json.dump(gmm_out, f, indent=2)
    print(f"  Saved GMM fit: {gmm_path}")

    # Plot
    out_path = os.path.join(run_dir, "selection_ellipse.png")
    make_plot(x, y, gmm, core, out_path, x_hi, y_lo)

    # Optional MLE at n-sigma
    if args.exe:
        mle_at_nsigma(x, y, sigma_x, sigma_y, mu, sigma,
                      args.n_sigma, args.exe, run_dir, x_hi, y_lo, gmm, core)


if __name__ == "__main__":
    main()
