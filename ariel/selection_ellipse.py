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

def load_data(fits_file, haty_min, haty_max):
    print(f"Reading FITS file: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data  # type: ignore[union-attr]
        total_rows = len(data)
        main_mask = np.asarray(data["MAIN"], dtype=bool)
        data_main = data[main_mask]

    print(f"  Total rows: {total_rows}  |  MAIN=True: {np.sum(main_mask)}")

    logvrot     = np.asarray(data_main["LOGVROT"],           dtype=float)
    logvrot_err = np.asarray(data_main["LOGVROT_ERR"],       dtype=float)
    absmag      = np.asarray(data_main["R_ABSMAG_SB26"],     dtype=float)
    absmag_err  = np.asarray(data_main["R_ABSMAG_SB26_ERR"], dtype=float)

    x_raw     = logvrot - 2.0
    sigma_x   = logvrot_err
    y_raw     = absmag
    sigma_y   = absmag_err

    valid = (
        np.isfinite(x_raw)
        & np.isfinite(sigma_x)
        & np.isfinite(y_raw)
        & np.isfinite(sigma_y)
        & (logvrot > 0)
        & (sigma_x > 0)
        & (sigma_y >= 0)
        & (y_raw >= haty_min)
        & (y_raw <= haty_max)
    )

    print(f"  Valid rows after MAIN + validity + pre-filter: {valid.sum()}")
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

    sigma_colors = ["gold", "orange", "red"]
    sigma_styles = ["-", "--", ":"]
    for n_sigma, color, ls in zip([1, 2, 3], sigma_colors, sigma_styles):
        ell = mpatches.Ellipse(
            mu,
            width=2 * n_sigma * semi_axes[1],   # major axis full diameter
            height=2 * n_sigma * semi_axes[0],  # minor axis full diameter
            angle=angle,
            edgecolor=color,
            facecolor="none",
            linewidth=1.5,
            linestyle=ls,
            label=f"{n_sigma}σ ellipse",
            zorder=5,
        )
        ax.add_patch(ell)

    # Truncation boundary used in fit
    ax.axvline(x_hi, color="white", lw=1.0, ls="-",
               label=f"x_trunc={x_hi:.3f}")
    ax.axhline(y_lo, color="white", lw=1.0, ls="-",
               label=f"y_trunc={y_lo:.2f}")

    # Selection boundaries derived from the 1σ ellipse
    haty_min, haty_max, slope, intercept1, intercept2 = \
        derived_cuts(mu, sigma)

    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 300)
    ax.axhline(haty_max, color="cyan", lw=1.2, ls="--",
               label=f"haty_max={haty_max:.2f}")
    ax.axhline(haty_min, color="cyan", lw=1.2, ls=":",
               label=f"haty_min={haty_min:.2f}")
    ax.plot(x_line, slope * x_line + intercept1,
            color="lime", lw=1.2, ls="--",
            label=f"plane1: slope={slope:.2f}, b={intercept1:.2f}")
    ax.plot(x_line, slope * x_line + intercept2,
            color="lime", lw=1.2, ls=":",
            label=f"plane2: slope={slope:.2f}, b={intercept2:.2f}")

    ax.set_xlabel(r"$x = \log_{10}(V/100\,\mathrm{km\,s}^{-1})$")
    ax.set_ylabel(r"$y = R\mathrm{-band\ absolute\ magnitude}$")
    ax.set_title("Noise + truncation-corrected GMM — TFR core ellipses")
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
    parser.add_argument("--haty_min", type=float, default=-23.0,
                        help="Loose lower magnitude pre-filter (default: -23.0)")
    parser.add_argument("--haty_max", type=float, default=-18.0,
                        help="Loose upper magnitude pre-filter (default: -18.0)")
    parser.add_argument("--n_init",   type=int,   default=20,
                        help="GMM random restarts for initialisation (default: 20)")
    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # Load data (now includes sigma_x, sigma_y)
    x, y, sigma_x, sigma_y = load_data(args.file, args.haty_min, args.haty_max)

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


if __name__ == "__main__":
    main()
