"""
color_predict.py — Posterior predictive predictions using the color-correction TFR model.

Implements the algorithm in paper/main.tex §"Posterior Predictive Distribution"
(Appendix C, §C.4), using both x̂ and ẑ to predict absolute magnitude ŷ.
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import norm

from predict import (
    read_cmdstan_posterior,
    create_average_grid_image,
    _apply_main_cuts,
    _load_rz_color_from_desi,
    _load_gr_color_from_desi,
)
from mag_utils import get_mag_cols

# ---------------------------------------------------------------------------
# Systematic off-diagonal covariance terms (dust + photometric calibration)
# ---------------------------------------------------------------------------

# Std of internal-dust slope d from iron_internalDust z<0.1 MCMC
# (iron_internalDust_z0p1_mcmc_nokcorr.pickle, chain 0 filtered to d ∈ (−1.5, 0))
_D_ERR_R = 0.17680325261483004   # mag

# Photometric calibration floor for the DESI North footprint
_D_A_SYS = 0.02                  # mag


def _systematic_offdiag_terms(ba, photsys):
    """Per-galaxy systematic sensitivity vectors (MU / mag units).

    Parameters
    ----------
    ba : (G,) array-like — axis ratio b/a for each galaxy
    photsys : (G,) array-like of str — 'N' or 'S' per galaxy

    Returns
    -------
    v_dust : (G,) ndarray — internal-dust sensitivity  d_err_r × (BA − 1)
    v_phot : (G,) ndarray — photsys calibration floor  dAsys × 1_{N}
    """
    ba = np.asarray(ba, dtype=float)
    photsys = np.asarray(photsys)
    v_dust = _D_ERR_R * (ba - 1.0)
    v_phot = np.where(photsys == 'N', _D_A_SYS, 0.0)
    return v_dust, v_phot


def _add_systematic_offdiag(cov, ba, photsys):
    """Add dust and photsys off-diagonal covariance terms in-place.

    The diagonal is preserved exactly; only true off-diagonal elements change.

    Parameters
    ----------
    cov : (G, G) ndarray — covariance matrix, modified in-place
    ba : (G,) array-like — axis ratio b/a
    photsys : (G,) array-like of str — 'N' or 'S'

    Returns
    -------
    cov : same array, modified in-place
    """
    v_dust, v_phot = _systematic_offdiag_terms(ba, photsys)
    diag = np.diag(cov).copy()
    cov += np.outer(v_dust, v_dust) + np.outer(v_phot, v_phot)
    np.fill_diagonal(cov, diag)   # restore diagonal exactly
    return cov


def load_xyz_and_uncertainties_from_desi(
    fits_path,
    *,
    V0=100.0,
    vel_col="V_0p4R26",
    vel_err_col="V_0p4R26_ERR",
    mag_col=None,
    mag_err_col=None,
    z_col="Z_DESI",
    z_col_candidates=("zobs", "ZOBS", "Z", "ZHELIO", "Z_CMB", "ZDESI", "ZTRUE"),
    apply_valid_mask=True,
):
    """
    Load x̂, σ_x, ŷ, σ_y, ẑ, σ_z, z_obs from a DESI FITS catalog.

    ẑ is the z-band absolute magnitude (Z_ABSMAG_SB26 or derived from Z_MAG_SB26).
    σ_z is Z_MAG_SB26_ERR (photometric noise on z-band).

    Returns
    -------
    xhat, sigma_x, yhat, sigma_y, zhat, sigma_z, zobs : np.ndarray
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data  # type: ignore[union-attr]
        names = set(data.dtype.names or ())

        # Resolve redshift column
        if z_col not in names:
            found = None
            for cand in z_col_candidates:
                if cand in names:
                    found = cand
                    break
            if found is None:
                raise ValueError(
                    f"Could not find redshift column. Tried z_col={z_col!r} and candidates "
                    f"{z_col_candidates}. Available: {sorted(list(names))[:30]} ..."
                )
            z_col_use = found
        else:
            z_col_use = z_col

        # r-band magnitude columns
        if mag_col is None or mag_err_col is None:
            col_abs, col_abs_err, _ = get_mag_cols(names)
            mag_col = mag_col or col_abs
            mag_err_col = mag_err_col or col_abs_err

        # z-band magnitude columns
        if "Z_ABSMAG_SB26_CORR" in names:
            z_abs_col = "Z_ABSMAG_SB26_CORR"
            z_abs_err_col = "Z_ABSMAG_SB26_ERR_CORR"
        elif "Z_ABSMAG_SB26" in names:
            z_abs_col = "Z_ABSMAG_SB26"
            z_abs_err_col = "Z_ABSMAG_SB26_ERR"
        else:
            # Derive from apparent magnitudes
            if "Z_MAG_SB26_CORR" in names:
                z_app_col: str = "Z_MAG_SB26_CORR"
                z_app_err_col: str = "Z_MAG_SB26_ERR_CORR"
            elif "Z_MAG_SB26" in names:
                z_app_col = "Z_MAG_SB26"
                z_app_err_col = "Z_MAG_SB26_ERR"
            else:
                raise ValueError(
                    "No z-band magnitude column found (tried Z_ABSMAG_SB26_CORR, "
                    "Z_ABSMAG_SB26, Z_MAG_SB26_CORR, Z_MAG_SB26)."
                )
            z_abs_col = None
            z_abs_err_col = None

        # Load velocity
        V = np.asarray(data[vel_col], dtype=float)
        V_err = np.asarray(data[vel_err_col], dtype=float)
        yhat = np.asarray(data[mag_col], dtype=float)
        sigma_y = np.asarray(data[mag_err_col], dtype=float)
        zobs = np.asarray(data[z_col_use], dtype=float)

        # Load z-band absolute magnitude
        if z_abs_col is not None:
            zhat = np.asarray(data[z_abs_col], dtype=float)
            sigma_z = np.asarray(data[z_abs_err_col], dtype=float)
        else:
            # Derive: Z_ABSMAG = R_ABSMAG - (R_MAG - Z_MAG)
            # Need apparent r-band
            _, _, app_mag_col = get_mag_cols(names)
            R_app = np.asarray(data[app_mag_col], dtype=float)
            Z_app = np.asarray(data[z_app_col], dtype=float)
            zhat = yhat - (R_app - Z_app)
            sigma_z = np.asarray(data[z_app_err_col], dtype=float)

    # Convert to xhat and sigma_x
    xhat = np.log10(V / V0)
    sigma_x = V_err / (V * np.log(10))

    if apply_valid_mask:
        mask = (
            np.isfinite(V)
            & np.isfinite(V_err)
            & np.isfinite(yhat)
            & np.isfinite(sigma_y)
            & np.isfinite(zhat)
            & np.isfinite(sigma_z)
            & np.isfinite(zobs)
            & (V > 0)
            & (V_err > 0)
            & (sigma_y >= 0)
            & (sigma_z >= 0)
        )
        xhat = xhat[mask]
        sigma_x = sigma_x[mask]
        yhat = yhat[mask]
        sigma_y = sigma_y[mask]
        zhat = zhat[mask]
        sigma_z = sigma_z[mask]
        zobs = zobs[mask]

    return xhat, sigma_x, yhat, sigma_y, zhat, sigma_z, zobs


def load_gband_from_desi(fits_path, *, apply_valid_mask=True):
    """
    Load g-band absolute magnitude and uncertainty from a DESI FITS catalog.

    Uses the same validity mask as load_xyz_and_uncertainties_from_desi so
    arrays are aligned.

    Returns
    -------
    ghat : (G,) array — g-band absolute magnitudes
    sigma_g : (G,) array — g-band uncertainties
    """
    from mag_utils import get_mag_cols

    z_col_candidates = ("Z_DESI", "zobs", "ZOBS", "Z", "ZHELIO", "Z_CMB", "ZDESI", "ZTRUE")
    with fits.open(fits_path) as hdul:
        data = hdul[1].data  # type: ignore[union-attr]
        names = set(data.dtype.names or ())

        # r-band columns (needed for derivation)
        col_abs, col_abs_err, app_mag_col = get_mag_cols(names)
        yhat_raw = np.asarray(data[col_abs], dtype=float)
        sigma_y_raw = np.asarray(data[col_abs_err], dtype=float)
        R_app = np.asarray(data[app_mag_col], dtype=float)

        # g-band apparent magnitude
        if "G_MAG_SB26_CORR" in names:
            G_app = np.asarray(data["G_MAG_SB26_CORR"], dtype=float)
            G_err = np.asarray(data["G_MAG_SB26_ERR_CORR"], dtype=float)
        elif "G_MAG_SB26" in names:
            G_app = np.asarray(data["G_MAG_SB26"], dtype=float)
            G_err = np.asarray(data["G_MAG_SB26_ERR"], dtype=float)
        else:
            raise ValueError("No g-band column found (tried G_MAG_SB26_CORR, G_MAG_SB26).")

        # g-band absolute magnitude: G_ABSMAG = R_ABSMAG - (R_MAG - G_MAG)
        ghat_raw = yhat_raw - (R_app - G_app)
        sigma_g_raw = G_err

        # z-band (needed for validity mask alignment)
        if "Z_MAG_SB26_CORR" in names:
            Z_app = np.asarray(data["Z_MAG_SB26_CORR"], dtype=float)
            Z_err = np.asarray(data["Z_MAG_SB26_ERR_CORR"], dtype=float)
        elif "Z_MAG_SB26" in names:
            Z_app = np.asarray(data["Z_MAG_SB26"], dtype=float)
            Z_err = np.asarray(data["Z_MAG_SB26_ERR"], dtype=float)
        else:
            Z_app = np.zeros_like(R_app)
            Z_err = np.zeros_like(R_app)
        zhat_raw = yhat_raw - (R_app - Z_app)
        sigma_z_raw = Z_err

        # Velocity and redshift (for mask alignment)
        V = np.asarray(data["V_0p4R26"], dtype=float)
        V_err = np.asarray(data["V_0p4R26_ERR"], dtype=float)
        z_col_use = next((c for c in ("Z_DESI",) + z_col_candidates if c in names), None)
        zobs_raw = np.asarray(data[z_col_use], dtype=float) if z_col_use else np.ones(len(V))

    if apply_valid_mask:
        mask = (
            np.isfinite(V)
            & np.isfinite(V_err)
            & np.isfinite(yhat_raw)
            & np.isfinite(sigma_y_raw)
            & np.isfinite(zhat_raw)
            & np.isfinite(sigma_z_raw)
            & np.isfinite(zobs_raw)
            & (V > 0)
            & (V_err > 0)
            & (sigma_y_raw >= 0)
            & (sigma_z_raw >= 0)
            & np.isfinite(ghat_raw)
            & np.isfinite(sigma_g_raw)
        )
        ghat_raw = ghat_raw[mask]
        sigma_g_raw = sigma_g_raw[mask]

    return ghat_raw, sigma_g_raw


def load_xyz_and_uncertainties_from_stan_json(json_path, *, apply_valid_mask=True):
    """
    Load x̂, σ_x, ŷ, σ_y, ẑ, σ_z, z_obs from a Stan-style input JSON
    (as produced by desi_data.py for color.stan).

    Returns
    -------
    xhat, sigma_x, yhat, sigma_y, zhat, sigma_z, zobs : np.ndarray
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        stan_data = json.load(f)

    xhat = np.asarray(stan_data["x"], dtype=float)
    sigma_x = np.asarray(stan_data["sigma_x"], dtype=float)
    yhat = np.asarray(stan_data["y"], dtype=float)
    sigma_y = np.asarray(stan_data["sigma_y"], dtype=float)
    zhat = np.asarray(stan_data["z"], dtype=float)
    sigma_z = np.asarray(stan_data["sigma_z"], dtype=float)
    zobs = np.asarray(stan_data["z_obs"], dtype=float)

    if apply_valid_mask:
        mask = (
            np.isfinite(xhat)
            & np.isfinite(sigma_x)
            & np.isfinite(yhat)
            & np.isfinite(sigma_y)
            & np.isfinite(zhat)
            & np.isfinite(sigma_z)
            & np.isfinite(zobs)
            & (sigma_x > 0)
            & (sigma_y >= 0)
            & (sigma_z >= 0)
        )
        xhat = xhat[mask]
        sigma_x = sigma_x[mask]
        yhat = yhat[mask]
        sigma_y = sigma_y[mask]
        zhat = zhat[mask]
        sigma_z = sigma_z[mask]
        zobs = zobs[mask]

    return xhat, sigma_x, yhat, sigma_y, zhat, sigma_z, zobs


def ystar_pp_mean_sd_color_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    zhat_star,
    sigma_z_star,
    *,
    sigma_y_star,
    x_bar,
    y_min,
    y_max,
    zobs_star,
    mean_log1pz,
    on_bad_Z="raise",
    Z_floor=1e-300,
):
    """
    Posterior predictive mean and SD of ŷ_* for all galaxies under the
    color-correction model, using observed (x̂_*, ẑ_*) to predict ŷ_*.

    Implements paper/main.tex Appendix C §C.4, Eqs. (C.33)–(C.39).

    Parameters
    ----------
    draws : DataFrame
        MCMC posterior with columns: "slope", "intercept.1", "sigma_int_x",
        "sigma_int_y", "sigma_int_z", "gamma", "delta_c", "mu_c", "tau_c",
        "alpha_kcorr_r", "alpha_kcorr_z".
    xhat_star : (G,) array — observed log-velocity
    sigma_x_star : (G,) array — uncertainty on x̂
    zhat_star : (G,) array — observed z-band absolute magnitude
    sigma_z_star : (G,) array — uncertainty on ẑ
    sigma_y_star : (G,) array — measurement uncertainty on ŷ (enters A₁₁)
    x_bar : float — sample mean of x̂ (from training data)
    y_min, y_max : float — tophat prior bounds on y_TF
    on_bad_Z : {"raise", "floor"}
    Z_floor : float

    Returns
    -------
    mean_y : (G,) array — posterior predictive mean of ŷ_*
    sd_y : (G,) array — posterior predictive SD of ŷ_*
    """
    xhat_star = np.asarray(xhat_star, dtype=float)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)
    zhat_star = np.asarray(zhat_star, dtype=float)
    sigma_z_star = np.asarray(sigma_z_star, dtype=float)
    sigma_y_star = np.asarray(sigma_y_star, dtype=float)

    a = float(y_min)
    b = float(y_max)
    if not (a < b):
        raise ValueError(f"Require y_min < y_max; got y_min={a}, y_max={b}.")

    # Extract draws (M,)
    alpha = draws["slope"].to_numpy(float)
    beta = draws["intercept.1"].to_numpy(float)
    six = draws["sigma_int_x"].to_numpy(float)
    siy = draws["sigma_int_y"].to_numpy(float)
    siz = draws["sigma_int_z"].to_numpy(float)
    gamma = draws["gamma"].to_numpy(float)
    delta = draws["delta_c"].to_numpy(float)
    mu_c = draws["mu_c"].to_numpy(float)
    tau_c = draws["tau_c"].to_numpy(float)
    alpha_k_r = draws["alpha_kcorr_r"].to_numpy(float)  # (M,) r-band k-correction slope
    alpha_k_z = draws["alpha_kcorr_z"].to_numpy(float)  # (M,) z-band k-correction slope

    zobs_star = np.asarray(zobs_star, dtype=float)

    if np.any(alpha == 0):
        raise ValueError("Found slope == 0 in draws; model requires α ≠ 0.")

    # Broadcast to (M, G)
    aMG = alpha[:, None]
    bMG = beta[:, None]
    sixMG = six[:, None]
    siyMG = siy[:, None]
    sizMG = siz[:, None]
    gMG = gamma[:, None]
    dMG = delta[:, None]
    mcMG = mu_c[:, None]
    tcMG = tau_c[:, None]

    # Per-galaxy, per-draw quantities
    sigma1_sq = sixMG**2 + sigma_x_star[None, :] ** 2  # (M, G)

    # A matrix entries (Eq. C.17); A₁₁ includes σ²_{y,★} (measurement noise)
    A11 = gMG**2 * tcMG**2 + siyMG**2 + sigma_y_star[None, :] ** 2  # (M, G)
    A12 = gMG * (gMG - 1) * tcMG**2  # (M, G)
    A22 = (gMG - 1) ** 2 * tcMG**2 + sizMG**2 + sigma_z_star[None, :] ** 2  # (M, G)

    # b vector from D^{-1} [0, A12]^T (Eq. C.28)
    # D = [[σ1², -δ·σ²_{int,x}], [-δ·σ²_{int,x}, A22 + δ²·σ²_{int,x}]]
    # (off-diagonal uses σ²_{int,x} because ẑ depends on latent x, not x̂)
    # det(D) = σ1²·A22 + δ²·σ²_{int,x}·σ²_x
    # b = D^{-1} [0, A12]^T
    # b[0] = (δ·σ²_{int,x} · A12) / det_D
    # b[1] = (σ1² · A12) / det_D
    sigma_intx_sq = sixMG**2  # (M, G)
    sigma_x_sq = sigma_x_star[None, :] ** 2  # (M, G)
    det_D = sigma1_sq * A22 + dMG**2 * sigma_intx_sq * sigma_x_sq  # (M, G)
    b0 = dMG * sigma_intx_sq * A12 / det_D  # (M, G)
    b1 = sigma1_sq * A12 / det_D  # (M, G)

    # Conditional variance σ²_{y|x̂ẑ} = A11 - [0, A12]·D^{-1}·[0, A12]^T
    # = A11 - σ1² · A12² / det_D
    sigma_y_given_xz_sq = A11 - sigma1_sq * A12**2 / det_D  # (M, G)

    # ---- Step 4: Truncated normal posterior for y_TF | x̂, ẑ, θ ----
    # Bivariate likelihood (x̂, ẑ) | y_TF ~ N_2(m(y_TF), D_★) (paper Eq. cc:bivariate_xz),
    # with m(y_TF) = m(0) + y_TF · b_xz, b_xz = (1/α, 1 - δ/α).
    # Posterior on y_TF (uniform prior) is N(μ^†_xz, 1/ξ_xz) truncated to [y_min, y_max],
    # where ξ_xz = b_xz^T D^{-1} b_xz and
    #       μ^†_xz = ξ_xz^{-1} · b_xz^T D^{-1} (o - m(0))
    # with o = (x̂, ẑ)^T and m(0) = (-β/α, Δ - μ_c + δβ/α + δ·x̄)^T.

    # Band-dependent k-correction terms per (M, G)
    log1pz_centered = np.log1p(zobs_star[None, :]) - mean_log1pz  # (1, G)
    alpha_zn_r = alpha_k_r[:, None] * log1pz_centered  # (M, G) r-band
    alpha_zn_z = alpha_k_z[:, None] * log1pz_centered  # (M, G) z-band

    # adj(D) = [[A22 + δ²σ²_intx, δσ²_intx], [δσ²_intx, σ1²]]
    adjD_11 = A22 + dMG**2 * sigma_intx_sq  # (M, G)
    adjD_12 = dMG * sigma_intx_sq           # (M, G)
    adjD_22 = sigma1_sq                     # (M, G)

    # b_xz components
    bxz_0 = 1.0 / aMG          # (M, G)
    bxz_1 = 1.0 - dMG / aMG    # (M, G)

    # Residuals at y_TF = 0: o - m(0)
    r0_x = xhat_star[None, :] + bMG / aMG                                      # (M, G)
    r0_z = zhat_star[None, :] - alpha_zn_z + mcMG - dMG * bMG / aMG - dMG * x_bar  # (M, G)

    # b_xz^T adj(D) b_xz and b_xz^T adj(D) (o - m(0)); divide by det_D once.
    bAb = bxz_0 * (adjD_11 * bxz_0 + adjD_12 * bxz_1) + bxz_1 * (
        adjD_12 * bxz_0 + adjD_22 * bxz_1
    )  # (M, G)
    bAo = bxz_0 * (adjD_11 * r0_x + adjD_12 * r0_z) + bxz_1 * (
        adjD_12 * r0_x + adjD_22 * r0_z
    )  # (M, G)

    xi_xz = bAb / det_D            # (M, G); always > 0 since D is PD
    mu_L = bAo / bAb               # (M, G); = (bAo/det_D) / (bAb/det_D)
    sigma_L_sq = 1.0 / xi_xz
    sigma_L = np.sqrt(sigma_L_sq)  # (M, G)

    # Compute truncated normal mean and variance on [a, b]
    mean_yTF = np.empty_like(mu_L)
    var_yTF = np.empty_like(mu_L)

    deg = sigma_L == 0.0
    if np.any(deg):
        mu_deg = mu_L[deg]
        ok = (mu_deg >= a) & (mu_deg <= b)
        if not np.all(ok):
            raise ValueError(
                "Encountered sigma_L == 0 with mu_L outside [y_min,y_max]."
            )
        mean_yTF[deg] = mu_deg
        var_yTF[deg] = 0.0

    nd = ~deg
    if np.any(nd):
        mu = mu_L[nd]
        sig = sigma_L[nd]

        alpha_tn = (a - mu) / sig
        beta_tn = (b - mu) / sig

        # Numerically stable log(Z) computation
        use_sf = alpha_tn >= 0.0
        log_sf_a = norm.logsf(alpha_tn)
        log_sf_b = norm.logsf(beta_tn)
        log_cdf_a = norm.logcdf(alpha_tn)
        log_cdf_b = norm.logcdf(beta_tn)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_Z_sf = log_sf_a + np.log1p(
                -np.exp(np.clip(log_sf_b - log_sf_a, -np.inf, 0.0))
            )
            log_Z_cdf = log_cdf_b + np.log1p(
                -np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0))
            )
        log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)

        if on_bad_Z == "raise":
            if np.any(~np.isfinite(log_Z)):
                raise ValueError("log(Z) is non-finite for some (draw, galaxy).")
        elif on_bad_Z == "floor":
            log_Z = np.maximum(log_Z, np.log(Z_floor))
        else:
            raise ValueError("on_bad_Z must be 'raise' or 'floor'.")

        log_phi_a = norm.logpdf(alpha_tn)
        log_phi_b = norm.logpdf(beta_tn)
        la = np.exp(log_phi_a - log_Z)
        lb = np.exp(log_phi_b - log_Z)

        t = la - lb
        m = mu + sig * t
        u = alpha_tn * la - beta_tn * lb
        v = (sig**2) * (1.0 + u - t**2)
        v = np.maximum(v, 0.0)

        mean_yTF[nd] = m
        var_yTF[nd] = v

    # ---- Step 5: Conditional mean E[ŷ_* | x̂, ẑ, θ] ----
    # μ_{y|x̂ẑ}(y_TF) = y_TF + Δ + b^T · [x̂ - μ_x(y_TF), ẑ - (y_TF + Δ - μ_c - δ(μ_x(y_TF) - x̄))]
    # where μ_x(y_TF) = (y_TF - β) / α, Δ = α_k·ln(1+z_obs)
    #
    # This is linear in y_TF, so E[μ_{y|x̂ẑ}(y_TF)] = μ_{y|x̂ẑ}(E[y_TF]) exactly.
    #
    # Compute μ_x(mean_yTF):
    mu_x_at_mean = (mean_yTF - bMG) / aMG  # (M, G)

    # Residual vector at y_TF = mean_yTF:
    res0 = xhat_star[None, :] - mu_x_at_mean  # (M, G)
    res1 = zhat_star[None, :] - (
        mean_yTF + alpha_zn_z - mcMG - dMG * (mu_x_at_mean - x_bar)
    )  # (M, G)

    cond_mean = mean_yTF + alpha_zn_r + b0 * res0 + b1 * res1  # (M, G)

    # ---- Step 6: Conditional variance Var[ŷ | x̂, ẑ, θ] ----
    # ∂μ_{y|x̂ẑ}/∂y_TF = 1 + b^T · ∂residuals/∂y_TF
    # ∂res0/∂y_TF = -1/α
    # ∂res1/∂y_TF = -(1 - δ/α)
    dres0_dyTF = -1.0 / aMG  # (M, G)
    dres1_dyTF = -(1.0 - dMG / aMG)  # (M, G)

    dmu_dyTF = 1.0 + b0 * dres0_dyTF + b1 * dres1_dyTF  # (M, G)

    # Var[ŷ | x̂, ẑ, θ] = σ²_{y|x̂ẑ} + (∂μ/∂y_TF)² · Var(y_TF)
    cond_var = sigma_y_given_xz_sq + dmu_dyTF**2 * var_yTF  # (M, G)

    # ---- Step 7: Mix over MCMC draws ----
    mean_y = cond_mean.mean(axis=0)  # (G,)
    var_y = cond_var.mean(axis=0) + (cond_mean**2).mean(axis=0) - mean_y**2
    sd_y = np.sqrt(np.maximum(var_y, 0.0))

    return mean_y, sd_y


def ystar_pp_mean_sd_2color_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    zhat_star,
    sigma_z_star,
    ghat_star,
    sigma_g_star,
    *,
    sigma_y_star,
    x_bar,
    y_min,
    y_max,
    zobs_star,
    mean_log1pz,
    on_bad_Z="raise",
    Z_floor=1e-300,
):
    """
    Posterior predictive mean and SD of ŷ_* for the 2color model,
    conditioning on observed (x̂_*, ẑ_*, ĝ_*) to predict ŷ_*.

    Extends ystar_pp_mean_sd_color_vectorized from a 2×2 D matrix (x, z)
    to a 3×3 D matrix (x, z, g) with independent color factors.

    Parameters
    ----------
    draws : DataFrame
        MCMC posterior with columns: "slope", "intercept.1", "sigma_int_x",
        "sigma_int_y", "sigma_int_z", "sigma_int_g", "gamma", "gamma_g",
        "delta_c", "delta_g", "mu_c", "mu_g", "tau_c", "tau_g",
        "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g".
    xhat_star : (G,) array — observed log-velocity
    sigma_x_star : (G,) array — uncertainty on x̂
    zhat_star : (G,) array — observed z-band absolute magnitude
    sigma_z_star : (G,) array — uncertainty on ẑ
    ghat_star : (G,) array — observed g-band absolute magnitude
    sigma_g_star : (G,) array — uncertainty on ĝ
    sigma_y_star : (G,) array — measurement uncertainty on ŷ (enters A₁₁)
    x_bar : float — sample mean of x̂ (from training data)
    y_min, y_max : float — tophat prior bounds on y_TF
    zobs_star : (G,) array — observed redshift
    mean_log1pz : float — mean log(1+z) from training sample

    Returns
    -------
    mean_y : (G,) array — posterior predictive mean of ŷ_*
    sd_y : (G,) array — posterior predictive SD of ŷ_*
    """
    xhat_star = np.asarray(xhat_star, dtype=float)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)
    zhat_star = np.asarray(zhat_star, dtype=float)
    sigma_z_star = np.asarray(sigma_z_star, dtype=float)
    ghat_star = np.asarray(ghat_star, dtype=float)
    sigma_g_star = np.asarray(sigma_g_star, dtype=float)
    sigma_y_star = np.asarray(sigma_y_star, dtype=float)
    zobs_star = np.asarray(zobs_star, dtype=float)

    a = float(y_min)
    b = float(y_max)
    if not (a < b):
        raise ValueError(f"Require y_min < y_max; got y_min={a}, y_max={b}.")

    # Extract draws (M,)
    alpha = draws["slope"].to_numpy(float)
    beta = draws["intercept.1"].to_numpy(float)
    six = draws["sigma_int_x"].to_numpy(float)
    siy = draws["sigma_int_y"].to_numpy(float)
    siz = draws["sigma_int_z"].to_numpy(float)
    sig_d = draws["sigma_int_g"].to_numpy(float)
    gamma_c = draws["gamma"].to_numpy(float)
    gamma_g = draws["gamma_g"].to_numpy(float)
    delta_c = draws["delta_c"].to_numpy(float)
    delta_g = draws["delta_g"].to_numpy(float)
    mu_c = draws["mu_c"].to_numpy(float)
    mu_g = draws["mu_g"].to_numpy(float)
    tau_c = draws["tau_c"].to_numpy(float)
    tau_g = draws["tau_g"].to_numpy(float)
    alpha_k_r = draws["alpha_kcorr_r"].to_numpy(float)
    alpha_k_z = draws["alpha_kcorr_z"].to_numpy(float)
    alpha_k_g = draws["alpha_kcorr_g"].to_numpy(float)

    if np.any(alpha == 0):
        raise ValueError("Found slope == 0 in draws; model requires α ≠ 0.")

    # Broadcast to (M, G)
    aMG = alpha[:, None]
    bMG = beta[:, None]
    sixMG = six[:, None]
    siyMG = siy[:, None]
    sizMG = siz[:, None]
    sigMG = sig_d[:, None]
    gcMG = gamma_c[:, None]
    ggMG = gamma_g[:, None]
    dcMG = delta_c[:, None]
    dgMG = delta_g[:, None]
    mcMG = mu_c[:, None]
    mgMG = mu_g[:, None]
    tcMG = tau_c[:, None]
    tgMG = tau_g[:, None]

    sigma_intx_sq = sixMG**2  # (M, G)
    sigma_x_sq = sigma_x_star[None, :] ** 2  # (1, G)
    sigma1_sq = sigma_intx_sq + sigma_x_sq  # (M, G)

    # A matrix entries (from 4×4 B in 2color.stan)
    A11 = gcMG**2 * tcMG**2 + ggMG**2 * tgMG**2 + siyMG**2 + sigma_y_star[None, :] ** 2
    A12 = gcMG * (gcMG - 1) * tcMG**2  # B[2,3] in Stan indexing
    A14 = ggMG * (ggMG - 1) * tgMG**2  # B[2,4] in Stan indexing
    A22 = (gcMG - 1) ** 2 * tcMG**2 + sizMG**2 + sigma_z_star[None, :] ** 2
    A44 = (ggMG - 1) ** 2 * tgMG**2 + sigMG**2 + sigma_g_star[None, :] ** 2

    # D matrix (3×3): sub-block of B for rows/cols {x, z, g}
    # D[0,0] = σ1², D[0,1] = -δ_c·s², D[0,2] = -δ_g·s²
    # D[1,1] = A22 + δ_c²·s², D[1,2] = 0, D[2,2] = A44 + δ_g²·s²
    D00 = sigma1_sq
    D01 = -dcMG * sigma_intx_sq
    D02 = -dgMG * sigma_intx_sq
    D11 = A22 + dcMG**2 * sigma_intx_sq
    D22 = A44 + dgMG**2 * sigma_intx_sq
    # D12 = 0 (independent factors)

    # Compute D^{-1} analytically. Since D12=0, the cofactors simplify:
    # det(D) = D00·D11·D22 - D01²·D22 - D02²·D11
    det_D = D00 * D11 * D22 - D01**2 * D22 - D02**2 * D11

    # Inverse of D (symmetric, D12=0):
    # Dinv[0,0] = D11·D22 / det
    # Dinv[1,1] = (D00·D22 - D02²) / det
    # Dinv[2,2] = (D00·D11 - D01²) / det
    # Dinv[0,1] = -D01·D22 / det
    # Dinv[0,2] = -D02·D11 / det
    # Dinv[1,2] = D01·D02 / det
    inv_det = 1.0 / det_D
    Di00 = D11 * D22 * inv_det
    Di11 = (D00 * D22 - D02**2) * inv_det
    Di22 = (D00 * D11 - D01**2) * inv_det
    Di01 = -D01 * D22 * inv_det
    Di02 = -D02 * D11 * inv_det
    Di12 = D01 * D02 * inv_det

    # Cross-vector c = B[y, {x,z,g}] = [0, A12, A14]
    # b_cond = D^{-1} c  (regression coefficients for E[ŷ|x̂,ẑ,ĝ,y_TF])
    # b_cond[0] = Di00·0 + Di01·A12 + Di02·A14 = Di01·A12 + Di02·A14
    # b_cond[1] = Di01·0 + Di11·A12 + Di12·A14 = Di11·A12 + Di12·A14
    # b_cond[2] = Di02·0 + Di12·A12 + Di22·A14 = Di12·A12 + Di22·A14
    bc0 = Di01 * A12 + Di02 * A14
    bc1 = Di11 * A12 + Di12 * A14
    bc2 = Di12 * A12 + Di22 * A14

    # Conditional variance σ²_{y|x̂ẑĝ} = A11 - c^T D^{-1} c
    cDinvc = A12 * (Di11 * A12 + Di12 * A14) + A14 * (Di12 * A12 + Di22 * A14)
    sigma_y_given_xzg_sq = A11 - cDinvc

    # b_xzg vector: how y_TF enters the observation means
    # b_xzg = [1/α, 1-δ_c/α, 1-δ_g/α]^T
    bxzg_0 = 1.0 / aMG
    bxzg_1 = 1.0 - dcMG / aMG
    bxzg_2 = 1.0 - dgMG / aMG

    # ξ = b_xzg^T D^{-1} b_xzg
    xi = (bxzg_0 * (Di00 * bxzg_0 + Di01 * bxzg_1 + Di02 * bxzg_2)
          + bxzg_1 * (Di01 * bxzg_0 + Di11 * bxzg_1 + Di12 * bxzg_2)
          + bxzg_2 * (Di02 * bxzg_0 + Di12 * bxzg_1 + Di22 * bxzg_2))

    # Band-dependent k-correction mean shifts
    log1pz_centered = np.log1p(zobs_star[None, :]) - mean_log1pz
    alpha_zn_r = alpha_k_r[:, None] * log1pz_centered
    alpha_zn_z = alpha_k_z[:, None] * log1pz_centered
    alpha_zn_g = alpha_k_g[:, None] * log1pz_centered

    # Residual at y_TF=0: o - a_vec (observation minus mean at y_TF=0)
    # a_vec = [-β/α, Δ_z - μ_c + δ_c·β/α, Δ_g - μ_g + δ_g·β/α]
    # o = [x̂, ẑ, ĝ]
    # r0 = o - a_vec
    r0_x = xhat_star[None, :] + bMG / aMG
    r0_z = zhat_star[None, :] - alpha_zn_z + mcMG - dcMG * bMG / aMG - dcMG * x_bar
    r0_g = ghat_star[None, :] - alpha_zn_g + mgMG - dgMG * bMG / aMG - dgMG * x_bar

    # φ = b_xzg^T D^{-1} r0
    Dinv_r0_0 = Di00 * r0_x + Di01 * r0_z + Di02 * r0_g
    Dinv_r0_1 = Di01 * r0_x + Di11 * r0_z + Di12 * r0_g
    Dinv_r0_2 = Di02 * r0_x + Di12 * r0_z + Di22 * r0_g
    phi = bxzg_0 * Dinv_r0_0 + bxzg_1 * Dinv_r0_1 + bxzg_2 * Dinv_r0_2

    # Posterior on y_TF: N(μ^†, 1/ξ) truncated to [a, b]
    mu_L = phi / xi
    sigma_L_sq = 1.0 / xi
    sigma_L = np.sqrt(sigma_L_sq)

    # Compute truncated normal mean and variance
    mean_yTF = np.empty_like(mu_L)
    var_yTF = np.empty_like(mu_L)

    deg = sigma_L == 0.0
    if np.any(deg):
        mu_deg = mu_L[deg]
        ok = (mu_deg >= a) & (mu_deg <= b)
        if not np.all(ok):
            raise ValueError("Encountered sigma_L == 0 with mu_L outside [y_min,y_max].")
        mean_yTF[deg] = mu_deg
        var_yTF[deg] = 0.0

    nd = ~deg
    if np.any(nd):
        mu = mu_L[nd]
        sig = sigma_L[nd]

        alpha_tn = (a - mu) / sig
        beta_tn = (b - mu) / sig

        use_sf = alpha_tn >= 0.0
        log_sf_a = norm.logsf(alpha_tn)
        log_sf_b = norm.logsf(beta_tn)
        log_cdf_a = norm.logcdf(alpha_tn)
        log_cdf_b = norm.logcdf(beta_tn)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_Z_sf = log_sf_a + np.log1p(
                -np.exp(np.clip(log_sf_b - log_sf_a, -np.inf, 0.0))
            )
            log_Z_cdf = log_cdf_b + np.log1p(
                -np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0))
            )
        log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)

        if on_bad_Z == "raise":
            if np.any(~np.isfinite(log_Z)):
                raise ValueError("log(Z) is non-finite for some (draw, galaxy).")
        elif on_bad_Z == "floor":
            log_Z = np.maximum(log_Z, np.log(Z_floor))
        else:
            raise ValueError("on_bad_Z must be 'raise' or 'floor'.")

        log_phi_a = norm.logpdf(alpha_tn)
        log_phi_b = norm.logpdf(beta_tn)
        la = np.exp(log_phi_a - log_Z)
        lb = np.exp(log_phi_b - log_Z)

        t = la - lb
        m = mu + sig * t
        u = alpha_tn * la - beta_tn * lb
        v = (sig**2) * (1.0 + u - t**2)
        v = np.maximum(v, 0.0)

        mean_yTF[nd] = m
        var_yTF[nd] = v

    # Conditional mean E[ŷ | x̂, ẑ, ĝ, θ]
    # = y_TF + Δ_r + bc · (o - m(y_TF))
    mu_x_at_mean = (mean_yTF - bMG) / aMG
    res0 = xhat_star[None, :] - mu_x_at_mean
    res1 = zhat_star[None, :] - (mean_yTF + alpha_zn_z - mcMG - dcMG * (mu_x_at_mean - x_bar))
    res2 = ghat_star[None, :] - (mean_yTF + alpha_zn_g - mgMG - dgMG * (mu_x_at_mean - x_bar))

    cond_mean = mean_yTF + alpha_zn_r + bc0 * res0 + bc1 * res1 + bc2 * res2

    # Conditional variance: ∂μ/∂y_TF contributions
    dres0_dyTF = -1.0 / aMG
    dres1_dyTF = -(1.0 - dcMG / aMG)
    dres2_dyTF = -(1.0 - dgMG / aMG)
    dmu_dyTF = 1.0 + bc0 * dres0_dyTF + bc1 * dres1_dyTF + bc2 * dres2_dyTF

    cond_var = sigma_y_given_xzg_sq + dmu_dyTF**2 * var_yTF

    # Mix over MCMC draws
    mean_y = cond_mean.mean(axis=0)
    var_y = cond_var.mean(axis=0) + (cond_mean**2).mean(axis=0) - mean_y**2
    sd_y = np.sqrt(np.maximum(var_y, 0.0))

    return mean_y, sd_y


def ystar_pp_mean_sd_color_xonly_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    *,
    sigma_y_star,
    y_min,
    y_max,
    zobs_star,
    mean_log1pz,
    on_bad_Z="raise",
    Z_floor=1e-300,
):
    """
    Posterior predictive mean and SD of ŷ_* using x̂ and redshift (no z-band).

    Marginalizes ẑ out of the trivariate distribution (Eq. C.trivariate).
    Since B[1,2]=0, ŷ ⊥ x̂ | y_TF, giving:

        ŷ_* | y_TF ~ N(y_TF + Δ_r, A₁₁)

    where A₁₁ = γ²τ_c² + σ²_{int,y} + σ²_{y,*} and Δ_r = α_{k,r}·[log(1+z) - mean].
    See paper/main.tex §sec:cc:x_only.

    Parameters
    ----------
    draws : DataFrame
        MCMC posterior with columns: "slope", "intercept.1", "sigma_int_x",
        "sigma_int_y", "gamma", "tau_c", "alpha_kcorr_r".
    xhat_star : (G,) array — observed log-velocity
    sigma_x_star : (G,) array — uncertainty on x̂
    sigma_y_star : (G,) array — measurement uncertainty on ŷ (enters A₁₁)
    y_min, y_max : float — tophat prior bounds on y_TF
    zobs_star : (G,) array — observed redshift (for k-correction)
    on_bad_Z : {"raise", "floor"}
    Z_floor : float

    Returns
    -------
    mean_y : (G,) array — posterior predictive mean of ŷ_*
    sd_y : (G,) array — posterior predictive SD of ŷ_*
    """
    xhat_star = np.asarray(xhat_star, dtype=float)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)
    sigma_y_star = np.asarray(sigma_y_star, dtype=float)
    zobs_star = np.asarray(zobs_star, dtype=float)

    a = float(y_min)
    b = float(y_max)
    if not (a < b):
        raise ValueError(f"Require y_min < y_max; got y_min={a}, y_max={b}.")

    # Extract draws (M,)
    alpha = draws["slope"].to_numpy(float)
    beta = draws["intercept.1"].to_numpy(float)
    six = draws["sigma_int_x"].to_numpy(float)
    siy = draws["sigma_int_y"].to_numpy(float)
    gamma = draws["gamma"].to_numpy(float)
    tau_c = draws["tau_c"].to_numpy(float)
    alpha_k_r = draws["alpha_kcorr_r"].to_numpy(float)

    if np.any(alpha == 0):
        raise ValueError("Found slope == 0 in draws; model requires α ≠ 0.")

    # Broadcast to (M, G)
    aMG = alpha[:, None]
    bMG = beta[:, None]
    sixMG = six[:, None]
    siyMG = siy[:, None]
    gMG = gamma[:, None]
    tcMG = tau_c[:, None]

    # A₁₁ = γ²τ_c² + σ²_{int,y} + σ²_{y,*}  (Eq. C.A, marginalizing ẑ)
    # For 2color model, also includes γ²_g·τ²_g
    A11 = gMG**2 * tcMG**2 + siyMG**2 + sigma_y_star[None, :] ** 2  # (M, G)
    if "gamma_g" in draws.columns and "tau_g" in draws.columns:
        gg = draws["gamma_g"].to_numpy(float)[:, None]
        tg = draws["tau_g"].to_numpy(float)[:, None]
        A11 = A11 + gg**2 * tg**2

    # Truncated normal posterior for y_TF | x̂  (identical to baseline)
    sigma1_sq = sixMG**2 + sigma_x_star[None, :] ** 2  # (M, G)
    mu_L = bMG + aMG * xhat_star[None, :]  # (M, G)
    sigma_L_sq = aMG**2 * sigma1_sq  # (M, G)
    sigma_L = np.sqrt(sigma_L_sq)  # (M, G)

    mean_yTF = np.empty_like(mu_L)
    var_yTF = np.empty_like(mu_L)

    deg = sigma_L == 0.0
    if np.any(deg):
        mu_deg = mu_L[deg]
        ok = (mu_deg >= a) & (mu_deg <= b)
        if not np.all(ok):
            raise ValueError("Encountered sigma_L == 0 with mu_L outside [y_min,y_max].")
        mean_yTF[deg] = mu_deg
        var_yTF[deg] = 0.0

    nd = ~deg
    if np.any(nd):
        mu = mu_L[nd]
        sig = sigma_L[nd]

        alpha_tn = (a - mu) / sig
        beta_tn = (b - mu) / sig

        use_sf = alpha_tn >= 0.0
        log_sf_a = norm.logsf(alpha_tn)
        log_sf_b = norm.logsf(beta_tn)
        log_cdf_a = norm.logcdf(alpha_tn)
        log_cdf_b = norm.logcdf(beta_tn)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_Z_sf = log_sf_a + np.log1p(
                -np.exp(np.clip(log_sf_b - log_sf_a, -np.inf, 0.0))
            )
            log_Z_cdf = log_cdf_b + np.log1p(
                -np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0))
            )
        log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)

        if on_bad_Z == "raise":
            if np.any(~np.isfinite(log_Z)):
                raise ValueError("log(Z) is non-finite for some (draw, galaxy).")
        elif on_bad_Z == "floor":
            log_Z = np.maximum(log_Z, np.log(Z_floor))
        else:
            raise ValueError("on_bad_Z must be 'raise' or 'floor'.")

        log_phi_a = norm.logpdf(alpha_tn)
        log_phi_b = norm.logpdf(beta_tn)
        la = np.exp(log_phi_a - log_Z)
        lb = np.exp(log_phi_b - log_Z)

        t = la - lb
        m = mu + sig * t
        u = alpha_tn * la - beta_tn * lb
        v = (sig**2) * (1.0 + u - t**2)
        v = np.maximum(v, 0.0)

        mean_yTF[nd] = m
        var_yTF[nd] = v

    # E[ŷ | x̂, z_obs, θ] = mean_yTF + Δ_r (r-band k-correction only)
    # Var[ŷ | x̂, z_obs, θ] = A₁₁ + V_*(θ)
    cond_mean = mean_yTF + alpha_k_r[:, None] * (np.log1p(zobs_star[None, :]) - mean_log1pz)  # (M, G)
    cond_var = A11 + var_yTF  # (M, G)

    # Mix over draws
    mean_y = cond_mean.mean(axis=0)  # (G,)
    var_y = cond_var.mean(axis=0) + (cond_mean**2).mean(axis=0) - mean_y**2
    sd_y = np.sqrt(np.maximum(var_y, 0.0))

    return mean_y, sd_y


def DESI_color(
    run_dir=None,
    grid_resolution_x=50,
    grid_resolution_y=50,
    make_residual_grid=True,
    make_redshift_grid=True,
    xonly=False,
    model="color",
):
    """
    Run color-correction model predictions and produce diagnostic plots.
    """
    _p = lambda name: os.path.join(run_dir, name) if run_dir else name

    # Load config
    with open(_p("config.json"), "r") as f:
        cfg = json.load(f)
    galaxy_fits = cfg["fits_file"]

    # Load input.json for bounds and x_bar
    with open(_p("input.json"), "r") as f:
        input_data = json.load(f)
    y_min = input_data["y_min"]
    y_max = input_data["y_max"]
    x_bar = input_data.get("mean_x", None)
    mean_log1pz = float(np.mean(np.log1p(input_data["z_obs"])))
    # Load galaxy data
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zhat_star, sigma_z_star, zobs_star = (
        load_xyz_and_uncertainties_from_desi(galaxy_fits)
    )

    # Compute x_bar from fitting sample if not in input.json
    if x_bar is None:
        x_bar = float(np.mean(input_data["x"]))

    # Load posterior draws and compute predictions
    if model == "2color":
        draws = read_cmdstan_posterior(
            _p(f"{model}_?.csv"),
            keep=[
                "slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                "sigma_int_z", "sigma_int_g", "gamma", "gamma_g",
                "delta_c", "delta_g", "mu_c", "mu_g", "tau_c", "tau_g",
                "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g",
            ],
            drop_diagnostics=True,
        )
        ghat_star, sigma_g_star = load_gband_from_desi(galaxy_fits)

        mean_pred, sd_pred = ystar_pp_mean_sd_2color_vectorized(
            draws, xhat_star, sigma_x_star, zhat_star, sigma_z_star,
            ghat_star, sigma_g_star,
            sigma_y_star=sigma_y_star, x_bar=x_bar,
            y_min=y_min, y_max=y_max,
            zobs_star=zobs_star, mean_log1pz=mean_log1pz,
            on_bad_Z="floor", Z_floor=1e-300,
        )
    else:
        draws = read_cmdstan_posterior(
            _p(f"{model}_?.csv"),
            keep=[
                "slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                "sigma_int_z", "gamma", "delta_c", "mu_c", "tau_c",
                "alpha_kcorr_r", "alpha_kcorr_z",
            ],
            drop_diagnostics=True,
        )
        ghat_star = sigma_g_star = None

        mean_pred, sd_pred = ystar_pp_mean_sd_color_vectorized(
            draws, xhat_star, sigma_x_star, zhat_star, sigma_z_star,
            sigma_y_star=sigma_y_star, x_bar=x_bar,
            y_min=y_min, y_max=y_max,
            zobs_star=zobs_star, mean_log1pz=mean_log1pz,
            on_bad_Z="floor", Z_floor=1e-300,
        )

    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    # MAIN sample mask
    rz_color_desi = _load_rz_color_from_desi(galaxy_fits)
    main_mask = _apply_main_cuts(cfg, xhat_star, yhat_star, rz_color=rz_color_desi)

    xhat_main = xhat_star[main_mask]
    sigma_x_main = sigma_x_star[main_mask]
    yhat_main = yhat_star[main_mask]
    sigma_y_main = sigma_y_star[main_mask]
    zhat_main = zhat_star[main_mask]
    sigma_z_main = sigma_z_star[main_mask]
    zobs_main = zobs_star[main_mask]

    if model == "2color":
        ghat_main = ghat_star[main_mask]
        sigma_g_main = sigma_g_star[main_mask]
        mean_pred_main, _ = ystar_pp_mean_sd_2color_vectorized(
            draws, xhat_main, sigma_x_main, zhat_main, sigma_z_main,
            ghat_main, sigma_g_main,
            sigma_y_star=sigma_y_main, x_bar=x_bar,
            y_min=y_min, y_max=y_max,
            zobs_star=zobs_main, mean_log1pz=mean_log1pz,
            on_bad_Z="floor", Z_floor=1e-300,
        )
    else:
        mean_pred_main, _ = ystar_pp_mean_sd_color_vectorized(
            draws, xhat_main, sigma_x_main, zhat_main, sigma_z_main,
            sigma_y_star=sigma_y_main, x_bar=x_bar,
            y_min=y_min, y_max=y_max,
            zobs_star=zobs_main, mean_log1pz=mean_log1pz,
            on_bad_Z="floor", Z_floor=1e-300,
        )
    mean_y_main = mean_pred_main - yhat_main

    # --- Residual grid: MAIN sample ---
    if make_residual_grid:
        fig, ax, img = create_average_grid_image(
            xhat_main,
            yhat_main,
            mean_y_main,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r"$\log{V/V_0}$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(img, ax=ax, label="Average Magnitude Difference")
        fig.savefig(_p("color_grid.png"), dpi=300)
        plt.close(fig)

        # Full sample
        fig, ax, img = create_average_grid_image(
            xhat_star,
            yhat_star,
            mean_y,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r"$\log{V/V_0}$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(img, ax=ax, label="Average Magnitude Difference")
        fig.savefig(_p("color_grid_full.png"), dpi=300)
        plt.close(fig)

    # --- Redshift grid ---
    if make_redshift_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star,
            yhat_star,
            zobs_star,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r"$\log{V/V_0}$")
        ax.set_ylabel(r"$M$")
        fig.colorbar(img, ax=ax, label="Average Redshift")
        fig.savefig(_p("redshift_grid_color.png"), dpi=300)
        plt.close(fig)

    # --- Redshift scatter plot ---
    plt.scatter(zobs_star, mean_y, marker=".", alpha=0.2, label="DR2 PV Spirals")
    plt.scatter(zobs_main, mean_y_main, marker=".", alpha=0.2, label="Main Sample")
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[\hat{y}_* | \hat{x}_*, \hat{z}_*] - \hat{y}_{\text{obs}}$ (mag)")
    plt.axhline(y=0, color="gray", linestyle="dashed", linewidth=1.5)
    plt.legend()
    # Set y-limits based on MAIN sample range with 10% padding
    y_min_main, y_max_main = np.min(mean_y_main), np.max(mean_y_main)
    y_range = y_max_main - y_min_main
    y_pad = 0.1 * y_range if y_range > 0 else 1.0
    plt.ylim((y_min_main - y_pad, y_max_main + y_pad))
    plt.savefig(_p("redshift_color.png"), dpi=300)
    plt.clf()

    # --- Variance vs redshift ---
    var_pred = sd_pred**2
    plt.scatter(zobs_star, var_pred, marker=".", alpha=0.15, s=4, label="Prediction")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"Magnitude variance (mag$^2$)")
    plt.legend()
    plt.savefig(_p("variance_redshift_color.png"), dpi=300)
    plt.clf()

    # --- Variance vs xhat ---
    plt.scatter(xhat_star, var_pred, marker=".", alpha=0.15, s=4, label="Prediction")
    plt.yscale("log")
    plt.xlabel(r"$\log(V/V_0)$")
    plt.ylabel(r"Magnitude variance (mag$^2$)")
    plt.legend()
    plt.savefig(_p("variance_xhat_color.png"), dpi=300)
    plt.clf()

    # --- x-only diagnostic plots (reuse loaded data and draws) ---
    if xonly:
        mean_pred_xo, sd_pred_xo = ystar_pp_mean_sd_color_xonly_vectorized(
            draws,
            xhat_star,
            sigma_x_star,
            sigma_y_star=sigma_y_star,
            y_min=y_min,
            y_max=y_max,
            zobs_star=zobs_star,
            mean_log1pz=mean_log1pz,
        )
        mean_pred_main_xo, _ = ystar_pp_mean_sd_color_xonly_vectorized(
            draws,
            xhat_main,
            sigma_x_main,
            sigma_y_star=sigma_y_main,
            y_min=y_min,
            y_max=y_max,
            zobs_star=zobs_main,
            mean_log1pz=mean_log1pz,
        )
        mean_y_xo = mean_pred_xo - yhat_star
        mean_y_main_xo = mean_pred_main_xo - yhat_main

        # Redshift scatter — x-only
        plt.scatter(zobs_star, mean_y_xo, marker=".", alpha=0.2, label="DR2 PV Spirals")
        plt.scatter(zobs_main, mean_y_main_xo, marker=".", alpha=0.2, label="Main Sample")
        plt.xscale("log")
        plt.xlabel(r"$z_{\text{obs}}$")
        plt.ylabel(r"$\mathbb{E}[\hat{y}_* | \hat{x}_*] - \hat{y}_{\text{obs}}$ (mag)")
        plt.axhline(y=0, color="gray", linestyle="dashed", linewidth=1.5)
        plt.legend()
        y_min_xo, y_max_xo = np.min(mean_y_main_xo), np.max(mean_y_main_xo)
        y_range_xo = y_max_xo - y_min_xo
        y_pad_xo = 0.1 * y_range_xo if y_range_xo > 0 else 1.0
        plt.ylim((y_min_xo - y_pad_xo, y_max_xo + y_pad_xo))
        plt.savefig(_p("redshift_color_xonly.png"), dpi=300)
        plt.clf()

        # g-r color residual — x-only
        gr_color = _load_gr_color_from_desi(galaxy_fits)
        if gr_color is not None:
            gr_main = gr_color[main_mask]
            plt.scatter(gr_color, mean_y_xo, marker=".", alpha=0.2, label="DR2 PV Spirals")
            plt.scatter(gr_main, mean_y_main_xo, marker=".", alpha=0.2, label="Main Sample")
            plt.xlabel(r"$g - r$ (mag)")
            plt.ylabel(r"$\mathbb{E}[\hat{y}_* | \hat{x}_*] - \hat{y}_{\rm obs}$ (mag)")
            plt.axhline(y=0, color="gray", linestyle="dashed", linewidth=1.5)
            plt.legend()
            y_min_xo2, y_max_xo2 = np.nanmin(mean_y_main_xo), np.nanmax(mean_y_main_xo)
            y_range_xo2 = y_max_xo2 - y_min_xo2
            y_pad_xo2 = 0.1 * y_range_xo2 if y_range_xo2 > 0 else 1.0
            plt.ylim((y_min_xo2 - y_pad_xo2, y_max_xo2 + y_pad_xo2))
            plt.savefig(_p("gr_color_xonly.png"), dpi=300)
            plt.clf()

        # Variance vs redshift — x-only
        var_pred_xo = sd_pred_xo**2
        plt.scatter(zobs_star, var_pred_xo, marker=".", alpha=0.15, s=4, label="Prediction")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$z_{\text{obs}}$")
        plt.ylabel(r"Magnitude variance (mag$^2$)")
        plt.legend()
        plt.savefig(_p("variance_redshift_color_xonly.png"), dpi=300)
        plt.clf()

        # Variance vs xhat — x-only
        plt.scatter(xhat_star, var_pred_xo, marker=".", alpha=0.15, s=4, label="Prediction")
        plt.yscale("log")
        plt.xlabel(r"$\log(V/V_0)$")
        plt.ylabel(r"Magnitude variance (mag$^2$)")
        plt.legend()
        plt.savefig(_p("variance_xhat_color_xonly.png"), dpi=300)
        plt.clf()

    return mean_y, sigma_y, zobs_star


def write_desi_catalog_color(run_dir, fits_path, cfg=None, model="color"):
    """
    Augment a DESI FITS catalog with color-model TFR-derived quantities and write
    to output/<run>/color_catalog.fits.

    New columns added (matching predict.py write_desi_catalog):
      MU_TF        = R_MAG_SB26_CORR - mean_pred
      MU_ERR       = sd_pred  (sd_pred already includes σ_{y,★} via A₁₁)
      LOGDIST      = 0.2 * ((R_MAG_SB26 - R_ABSMAG_SB26) - MU_TF)
      LOGDIST_ERR  = 0.2 * MU_ERR
      MAIN         = bool (True if passes selection cuts from config.json)
    """
    _p = lambda name: os.path.join(run_dir, name)

    z_col_candidates = ("Z_DESI", "zobs", "ZOBS", "Z", "ZHELIO", "Z_CMB", "ZDESI", "ZTRUE")

    with fits.open(fits_path) as hdul:
        primary_hdu = hdul[0].copy()
        table_hdu = hdul[1].copy()
        data = hdul[1].data  # type: ignore[union-attr]
        names = set(data.dtype.names or ())
        n_rows = len(data)

        z_col_use = None
        for cand in z_col_candidates:
            if cand in names:
                z_col_use = cand
                break
        if z_col_use is None:
            raise ValueError(
                f"Could not find redshift column. Tried: {z_col_candidates}. "
                f"Available: {sorted(list(names))[:30]} ..."
            )

        col_abs, col_abs_err, col_app = get_mag_cols(names)

        V = np.asarray(data["V_0p4R26"], dtype=float)
        V_err = np.asarray(data["V_0p4R26_ERR"], dtype=float)
        app = np.asarray(data[col_app], dtype=float)
        app_err = np.asarray(data[col_abs_err], dtype=float)
        abs_mag = np.asarray(data[col_abs], dtype=float)
        zobs = np.asarray(data[z_col_use], dtype=float)

        # z-band: load absolute magnitude column or derive from apparent
        if "Z_ABSMAG_SB26_CORR" in names:
            zhat_full = np.asarray(data["Z_ABSMAG_SB26_CORR"], dtype=float)
            sigma_z_full = np.asarray(data["Z_ABSMAG_SB26_ERR_CORR"], dtype=float)
        elif "Z_ABSMAG_SB26" in names:
            zhat_full = np.asarray(data["Z_ABSMAG_SB26"], dtype=float)
            sigma_z_full = np.asarray(data["Z_ABSMAG_SB26_ERR"], dtype=float)
        elif "Z_MAG_SB26_CORR" in names:
            z_app = np.asarray(data["Z_MAG_SB26_CORR"], dtype=float)
            sigma_z_full = np.asarray(data["Z_MAG_SB26_ERR_CORR"], dtype=float)
            zhat_full = abs_mag - (app - z_app)
        elif "Z_MAG_SB26" in names:
            z_app = np.asarray(data["Z_MAG_SB26"], dtype=float)
            sigma_z_full = np.asarray(data["Z_MAG_SB26_ERR"], dtype=float)
            zhat_full = abs_mag - (app - z_app)
        else:
            raise ValueError("No z-band magnitude column found for color model.")

        # r-z apparent color for the MAIN selection cut
        if "R_MAG_SB26_CORR" in names and "Z_MAG_SB26_CORR" in names:
            rz_color: np.ndarray | None = (
                np.asarray(data["R_MAG_SB26_CORR"], dtype=float)
                - np.asarray(data["Z_MAG_SB26_CORR"], dtype=float)
            )
        elif "R_MAG_SB26" in names and "Z_MAG_SB26" in names:
            rz_color = (
                np.asarray(data["R_MAG_SB26"], dtype=float)
                - np.asarray(data["Z_MAG_SB26"], dtype=float)
            )
        else:
            rz_color = None

    with np.errstate(invalid="ignore", divide="ignore"):
        xhat = np.where(V > 0, np.log10(V / 100.0), np.nan)
        sigma_x = np.where(V > 0, V_err / (V * np.log(10.0)), np.nan)

    valid = (
        np.isfinite(V)
        & (V > 0)
        & np.isfinite(V_err)
        & (V_err > 0)
        & np.isfinite(xhat)
        & np.isfinite(sigma_x)
        & (sigma_x > 0)
        & np.isfinite(zhat_full)
        & np.isfinite(sigma_z_full)
    )

    with open(_p("input.json"), "r") as f:
        input_data = json.load(f)
    y_min = input_data["y_min"]
    y_max = input_data["y_max"]
    x_bar = input_data.get("mean_x", float(np.mean(input_data["x"])))
    mean_log1pz = float(np.mean(np.log1p(input_data["z_obs"])))

    if model == "2color":
        draws = read_cmdstan_posterior(
            _p(f"{model}_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                  "sigma_int_z", "sigma_int_g", "gamma", "gamma_g",
                  "delta_c", "delta_g", "mu_c", "mu_g", "tau_c", "tau_g",
                  "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g"],
            drop_diagnostics=True,
        )
        ghat_full, sigma_g_full = load_gband_from_desi(fits_path, apply_valid_mask=False)
        valid = valid & np.isfinite(ghat_full) & np.isfinite(sigma_g_full)

        mean_pred_valid, sd_pred_valid = ystar_pp_mean_sd_2color_vectorized(
            draws,
            xhat[valid], sigma_x[valid],
            zhat_full[valid], sigma_z_full[valid],
            ghat_full[valid], sigma_g_full[valid],
            sigma_y_star=app_err[valid], x_bar=x_bar,
            y_min=y_min, y_max=y_max,
            zobs_star=zobs[valid], mean_log1pz=mean_log1pz,
            on_bad_Z="floor", Z_floor=1e-300,
        )
    else:
        draws = read_cmdstan_posterior(
            _p(f"{model}_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                  "sigma_int_z", "gamma", "delta_c", "mu_c", "tau_c",
                  "alpha_kcorr_r", "alpha_kcorr_z"],
            drop_diagnostics=True,
        )

        mean_pred_valid, sd_pred_valid = ystar_pp_mean_sd_color_vectorized(
            draws,
            xhat[valid], sigma_x[valid],
            zhat_full[valid], sigma_z_full[valid],
            sigma_y_star=app_err[valid], x_bar=x_bar,
            y_min=y_min, y_max=y_max,
            zobs_star=zobs[valid], mean_log1pz=mean_log1pz,
            on_bad_Z="floor", Z_floor=1e-300,
        )

    mean_pred_full = np.full(n_rows, np.nan)
    sd_pred_full = np.full(n_rows, np.nan)
    mean_pred_full[valid] = mean_pred_valid
    sd_pred_full[valid] = sd_pred_valid

    MU_TF = app - mean_pred_full
    MU_ERR = sd_pred_full  # sd_pred already includes σ_{y,★} via A₁₁
    MU_ZCMB = app - abs_mag
    LOGDIST = 0.2 * (MU_ZCMB - MU_TF)
    LOGDIST_ERR = 0.2 * MU_ERR

    if not cfg:
        with open(_p("config.json"), "r") as f:
            cfg = json.load(f)

    main = valid & _apply_main_cuts(cfg, xhat, abs_mag, zobs=zobs, rz_color=rz_color)

    new_cols = [
        fits.Column(name="MU_TF", format="E", array=MU_TF.astype(np.float32)),
        fits.Column(name="MU_ERR", format="E", array=MU_ERR.astype(np.float32)),
        fits.Column(name="LOGDIST", format="E", array=LOGDIST.astype(np.float32)),
        fits.Column(name="LOGDIST_ERR", format="E", array=LOGDIST_ERR.astype(np.float32)),
        fits.Column(name="MAIN", format="L", array=main),
    ]
    all_cols = fits.ColDefs(list(table_hdu.columns) + new_cols)
    new_table_hdu = fits.BinTableHDU.from_columns(all_cols)
    out_hdul = fits.HDUList([primary_hdu, new_table_hdu])
    out_path = _p("color_catalog.fits")
    out_hdul.writeto(out_path, overwrite=True)

    print(f"Written {n_rows} rows to {out_path}")
    print(f"  MAIN: {main.sum()} objects pass selection cuts")
    print(f"  MU_TF finite: {np.isfinite(MU_TF).sum()} objects")


def write_desi_catalog_color_xonly(run_dir, fits_path, cfg=None, model="color"):
    """
    Augment a DESI FITS catalog with color-model TFR predictions using x̂ and
    redshift (no z-band), writing to output/<run>/color_xonly_catalog.fits.

    Uses ystar_pp_mean_sd_color_xonly_vectorized: ŷ conditioned on x̂ and z_obs
    (k-correction), with A₁₁ = γ²τ_c² + σ²_{int,y} + σ²_{y,★} replacing σ²_{2,★}.
    See paper/main.tex §sec:cc:x_only.

    New columns (same as color_catalog.fits):
      MU_TF, MU_ERR, LOGDIST, LOGDIST_ERR, MAIN
    """
    _p = lambda name: os.path.join(run_dir, name)

    z_col_candidates = ("Z_DESI", "zobs", "ZOBS", "Z", "ZHELIO", "Z_CMB", "ZDESI", "ZTRUE")

    with fits.open(fits_path) as hdul:
        primary_hdu = hdul[0].copy()
        table_hdu = hdul[1].copy()
        data = hdul[1].data  # type: ignore[union-attr]
        names = set(data.dtype.names or ())
        n_rows = len(data)

        z_col_use = None
        for cand in z_col_candidates:
            if cand in names:
                z_col_use = cand
                break
        if z_col_use is None:
            raise ValueError(
                f"Could not find redshift column. Tried: {z_col_candidates}. "
                f"Available: {sorted(list(names))[:30]} ..."
            )

        col_abs, col_abs_err, col_app = get_mag_cols(names)

        V = np.asarray(data["V_0p4R26"], dtype=float)
        V_err = np.asarray(data["V_0p4R26_ERR"], dtype=float)
        app = np.asarray(data[col_app], dtype=float)
        app_err = np.asarray(data[col_abs_err], dtype=float)
        abs_mag = np.asarray(data[col_abs], dtype=float)
        zobs = np.asarray(data[z_col_use], dtype=float)

        if "R_MAG_SB26_CORR" in names and "Z_MAG_SB26_CORR" in names:
            rz_color: np.ndarray | None = (
                np.asarray(data["R_MAG_SB26_CORR"], dtype=float)
                - np.asarray(data["Z_MAG_SB26_CORR"], dtype=float)
            )
        elif "R_MAG_SB26" in names and "Z_MAG_SB26" in names:
            rz_color = (
                np.asarray(data["R_MAG_SB26"], dtype=float)
                - np.asarray(data["Z_MAG_SB26"], dtype=float)
            )
        else:
            rz_color = None

    with np.errstate(invalid="ignore", divide="ignore"):
        xhat = np.where(V > 0, np.log10(V / 100.0), np.nan)
        sigma_x = np.where(V > 0, V_err / (V * np.log(10.0)), np.nan)

    valid = (
        np.isfinite(V)
        & (V > 0)
        & np.isfinite(V_err)
        & (V_err > 0)
        & np.isfinite(xhat)
        & np.isfinite(sigma_x)
        & (sigma_x > 0)
    )

    keep_cols = ["slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                 "gamma", "tau_c", "alpha_kcorr_r"]
    if model == "2color":
        keep_cols += ["gamma_g", "tau_g"]
    draws = read_cmdstan_posterior(
        _p(f"{model}_?.csv"),
        keep=keep_cols,
        drop_diagnostics=True,
    )

    with open(_p("input.json"), "r") as f:
        input_data = json.load(f)
    y_min = input_data["y_min"]
    y_max = input_data["y_max"]
    mean_log1pz = float(np.mean(np.log1p(input_data["z_obs"])))

    mean_pred_valid, sd_pred_valid = ystar_pp_mean_sd_color_xonly_vectorized(
        draws,
        xhat[valid],
        sigma_x[valid],
        sigma_y_star=app_err[valid],
        y_min=y_min,
        y_max=y_max,
        zobs_star=zobs[valid],
        mean_log1pz=mean_log1pz,
        on_bad_Z="floor",
        Z_floor=1e-300,
    )

    mean_pred_full = np.full(n_rows, np.nan)
    sd_pred_full = np.full(n_rows, np.nan)
    mean_pred_full[valid] = mean_pred_valid
    sd_pred_full[valid] = sd_pred_valid

    MU_TF = app - mean_pred_full
    MU_ERR = sd_pred_full  # sd_pred includes σ_{y,★} via A₁₁
    MU_ZCMB = app - abs_mag
    LOGDIST = 0.2 * (MU_ZCMB - MU_TF)
    LOGDIST_ERR = 0.2 * MU_ERR

    if not cfg:
        with open(_p("config.json"), "r") as f:
            cfg = json.load(f)

    main = valid & _apply_main_cuts(cfg, xhat, abs_mag, zobs=zobs, rz_color=rz_color)

    new_cols = [
        fits.Column(name="MU_TF", format="E", array=MU_TF.astype(np.float32)),
        fits.Column(name="MU_ERR", format="E", array=MU_ERR.astype(np.float32)),
        fits.Column(name="LOGDIST", format="E", array=LOGDIST.astype(np.float32)),
        fits.Column(name="LOGDIST_ERR", format="E", array=LOGDIST_ERR.astype(np.float32)),
        fits.Column(name="MAIN", format="L", array=main),
    ]
    all_cols = fits.ColDefs(list(table_hdu.columns) + new_cols)
    new_table_hdu = fits.BinTableHDU.from_columns(all_cols)
    out_hdul = fits.HDUList([primary_hdu, new_table_hdu])
    out_path = _p("color_xonly_catalog.fits")
    out_hdul.writeto(out_path, overwrite=True)

    print(f"Written {n_rows} rows to {out_path}")
    print(f"  MAIN: {main.sum()} objects pass selection cuts")
    print(f"  MU_TF finite: {np.isfinite(MU_TF).sum()} objects")


def ystar_pp_cov_color_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    zhat_star,
    sigma_z_star,
    *,
    sigma_y_star,
    x_bar,
    y_min,
    y_max,
    zobs_star,
    mean_log1pz,
    on_bad_Z="floor",
    Z_floor=1e-300,
    chunk_size=200,
):
    """
    Posterior predictive covariance Cov(ŷ*[g1], ŷ*[g2]) — color-correction model.

    Off-diagonal elements arise solely from the shared uncertainty in the TFR
    parameters θ (law of total covariance, conditional independence given θ).

    Parameters
    ----------
    draws : DataFrame with columns slope, intercept.1, sigma_int_x, sigma_int_y,
            sigma_int_z, gamma, delta_c, mu_c, tau_c, alpha_kcorr_r, alpha_kcorr_z
    xhat_star, sigma_x_star, zhat_star, sigma_z_star : (G,) arrays
    sigma_y_star : (G,) array — measurement uncertainty on ŷ (enters A₁₁)
    x_bar : float — sample mean of x̂ (from training data)
    y_min, y_max : float — tophat prior bounds
    zobs_star : (G,) array — observed redshifts (for k-correction shift)
    chunk_size : int — draws per chunk to limit memory

    Returns
    -------
    cov : (G, G) ndarray
    """
    xhat_star = np.asarray(xhat_star, dtype=float)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)
    zhat_star = np.asarray(zhat_star, dtype=float)
    sigma_z_star = np.asarray(sigma_z_star, dtype=float)
    sigma_y_star = np.asarray(sigma_y_star, dtype=float)
    G = xhat_star.size

    mean_y, _ = ystar_pp_mean_sd_color_vectorized(
        draws, xhat_star, sigma_x_star, zhat_star, sigma_z_star,
        sigma_y_star=sigma_y_star,
        x_bar=x_bar, y_min=y_min, y_max=y_max,
        zobs_star=zobs_star, mean_log1pz=mean_log1pz,
        on_bad_Z=on_bad_Z, Z_floor=Z_floor,
    )

    a = float(y_min)
    b = float(y_max)

    alpha_d = draws["slope"].to_numpy(float)
    beta_d = draws["intercept.1"].to_numpy(float)
    six_d = draws["sigma_int_x"].to_numpy(float)
    siy_d = draws["sigma_int_y"].to_numpy(float)
    siz_d = draws["sigma_int_z"].to_numpy(float)
    gamma_d = draws["gamma"].to_numpy(float)
    delta_d = draws["delta_c"].to_numpy(float)
    mu_c_d = draws["mu_c"].to_numpy(float)
    tau_c_d = draws["tau_c"].to_numpy(float)
    alpha_k_r_d = draws["alpha_kcorr_r"].to_numpy(float)
    alpha_k_z_d = draws["alpha_kcorr_z"].to_numpy(float)
    M = len(draws)

    accum = np.zeros((G, G), dtype=float)
    var_accum = np.zeros(G, dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)

        aMG = alpha_d[start:end, None]
        bMG = beta_d[start:end, None]
        sixMG = six_d[start:end, None]
        siyMG = siy_d[start:end, None]
        sizMG = siz_d[start:end, None]
        gMG = gamma_d[start:end, None]
        dMG = delta_d[start:end, None]
        mcMG = mu_c_d[start:end, None]
        tcMG = tau_c_d[start:end, None]

        sigma1_sq = sixMG**2 + sigma_x_star[None, :]**2
        sigma_intx_sq = sixMG**2
        sigma_x_sq = sigma_x_star[None, :]**2
        A11 = gMG**2 * tcMG**2 + siyMG**2 + sigma_y_star[None, :]**2
        A12 = gMG * (gMG - 1) * tcMG**2
        A22 = (gMG - 1)**2 * tcMG**2 + sizMG**2 + sigma_z_star[None, :]**2

        det_D = sigma1_sq * A22 + dMG**2 * sigma_intx_sq * sigma_x_sq
        b0 = dMG * sigma_intx_sq * A12 / det_D
        b1 = sigma1_sq * A12 / det_D
        sigma_y_given_xz_sq = A11 - sigma1_sq * A12**2 / det_D

        # Band-dependent k-corrections
        log1pz_centered = np.log1p(zobs_star[None, :]) - mean_log1pz
        alpha_zn_r_chunk = alpha_k_r_d[start:end, None] * log1pz_centered
        alpha_zn_z_chunk = alpha_k_z_d[start:end, None] * log1pz_centered

        # Joint posterior y_TF | x̂, ẑ, θ  (paper Eq. cc:T_post_xz)
        adjD_11 = A22 + dMG**2 * sigma_intx_sq
        adjD_12 = dMG * sigma_intx_sq
        adjD_22 = sigma1_sq
        bxz_0 = 1.0 / aMG
        bxz_1 = 1.0 - dMG / aMG
        r0_x = xhat_star[None, :] + bMG / aMG
        r0_z = zhat_star[None, :] - alpha_zn_z_chunk + mcMG - dMG * bMG / aMG - dMG * x_bar
        bAb = bxz_0 * (adjD_11 * bxz_0 + adjD_12 * bxz_1) + bxz_1 * (
            adjD_12 * bxz_0 + adjD_22 * bxz_1
        )
        bAo = bxz_0 * (adjD_11 * r0_x + adjD_12 * r0_z) + bxz_1 * (
            adjD_12 * r0_x + adjD_22 * r0_z
        )
        xi_xz = bAb / det_D
        mu_L = bAo / bAb
        sigma_L_sq = 1.0 / xi_xz
        sigma_L = np.sqrt(sigma_L_sq)

        mu_chunk = np.empty_like(mu_L)
        var_chunk = np.empty_like(mu_L)

        deg = sigma_L == 0.0
        if np.any(deg):
            mu_deg = mu_L[deg]
            mu_chunk[deg] = mu_deg
            var_chunk[deg] = 0.0

        nd = ~deg
        if np.any(nd):
            mu = mu_L[nd]
            sig = sigma_L[nd]
            alpha_tn = (a - mu) / sig
            beta_tn = (b - mu) / sig

            use_sf = alpha_tn >= 0.0
            log_sf_a = norm.logsf(alpha_tn)
            log_sf_b = norm.logsf(beta_tn)
            log_cdf_a = norm.logcdf(alpha_tn)
            log_cdf_b = norm.logcdf(beta_tn)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_Z_sf = log_sf_a + np.log1p(
                    -np.exp(np.clip(log_sf_b - log_sf_a, -np.inf, 0.0))
                )
                log_Z_cdf = log_cdf_b + np.log1p(
                    -np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0))
                )
            log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)
            log_Z = np.maximum(log_Z, np.log(Z_floor))

            la = np.exp(norm.logpdf(alpha_tn) - log_Z)
            lb = np.exp(norm.logpdf(beta_tn) - log_Z)
            t = la - lb
            u = alpha_tn * la - beta_tn * lb
            mu_chunk[nd] = mu + sig * t
            var_chunk[nd] = np.maximum(sig**2 * (1.0 + u - t**2), 0.0)

        # Conditional mean of ŷ given (x̂, ẑ, θ) at y_TF = mu_chunk
        mu_x = (mu_chunk - bMG) / aMG
        res0 = xhat_star[None, :] - mu_x
        res1 = zhat_star[None, :] - (mu_chunk + alpha_zn_z_chunk - mcMG - dMG * (mu_x - x_bar))
        cond_mean_chunk = mu_chunk + alpha_zn_r_chunk + b0 * res0 + b1 * res1  # (B, G)

        # Conditional variance contribution at fixed θ
        dres0_dyTF = -1.0 / aMG
        dres1_dyTF = -(1.0 - dMG / aMG)
        dmu_dyTF = 1.0 + b0 * dres0_dyTF + b1 * dres1_dyTF
        cond_var_chunk = sigma_y_given_xz_sq + dmu_dyTF**2 * var_chunk  # (B, G)

        # Rank-B update to covariance accumulator (calibration variance term)
        mu_centered = cond_mean_chunk - mean_y[None, :]
        accum += mu_centered.T @ mu_centered

        # Expected conditional variance
        var_accum += cond_var_chunk.sum(axis=0)

    cov = accum / M
    np.fill_diagonal(cov, np.diag(cov) + var_accum / M)
    return cov


def ystar_pp_cov_2color_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    zhat_star,
    sigma_z_star,
    ghat_star,
    sigma_g_star,
    *,
    sigma_y_star,
    x_bar,
    y_min,
    y_max,
    zobs_star,
    mean_log1pz,
    on_bad_Z="floor",
    Z_floor=1e-300,
    chunk_size=200,
):
    """
    Posterior predictive covariance Cov(ŷ*[g1], ŷ*[g2]) — 2color model.

    Uses the 3×3 D matrix (conditioning on x̂, ẑ, ĝ).
    """
    xhat_star = np.asarray(xhat_star, dtype=float)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)
    zhat_star = np.asarray(zhat_star, dtype=float)
    sigma_z_star = np.asarray(sigma_z_star, dtype=float)
    ghat_star = np.asarray(ghat_star, dtype=float)
    sigma_g_star = np.asarray(sigma_g_star, dtype=float)
    sigma_y_star = np.asarray(sigma_y_star, dtype=float)
    zobs_star = np.asarray(zobs_star, dtype=float)
    G = xhat_star.size

    mean_y, _ = ystar_pp_mean_sd_2color_vectorized(
        draws, xhat_star, sigma_x_star, zhat_star, sigma_z_star,
        ghat_star, sigma_g_star,
        sigma_y_star=sigma_y_star,
        x_bar=x_bar, y_min=y_min, y_max=y_max,
        zobs_star=zobs_star, mean_log1pz=mean_log1pz,
        on_bad_Z=on_bad_Z, Z_floor=Z_floor,
    )

    a = float(y_min)
    b = float(y_max)

    alpha_d = draws["slope"].to_numpy(float)
    beta_d = draws["intercept.1"].to_numpy(float)
    six_d = draws["sigma_int_x"].to_numpy(float)
    siy_d = draws["sigma_int_y"].to_numpy(float)
    siz_d = draws["sigma_int_z"].to_numpy(float)
    sig_d = draws["sigma_int_g"].to_numpy(float)
    gamma_c_d = draws["gamma"].to_numpy(float)
    gamma_g_d = draws["gamma_g"].to_numpy(float)
    delta_c_d = draws["delta_c"].to_numpy(float)
    delta_g_d = draws["delta_g"].to_numpy(float)
    mu_c_d = draws["mu_c"].to_numpy(float)
    mu_g_d = draws["mu_g"].to_numpy(float)
    tau_c_d = draws["tau_c"].to_numpy(float)
    tau_g_d = draws["tau_g"].to_numpy(float)
    alpha_k_r_d = draws["alpha_kcorr_r"].to_numpy(float)
    alpha_k_z_d = draws["alpha_kcorr_z"].to_numpy(float)
    alpha_k_g_d = draws["alpha_kcorr_g"].to_numpy(float)
    M = len(draws)

    accum = np.zeros((G, G), dtype=float)
    var_accum = np.zeros(G, dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)

        aMG = alpha_d[start:end, None]
        bMG = beta_d[start:end, None]
        sixMG = six_d[start:end, None]
        siyMG = siy_d[start:end, None]
        sizMG = siz_d[start:end, None]
        sigMG = sig_d[start:end, None]
        gcMG = gamma_c_d[start:end, None]
        ggMG = gamma_g_d[start:end, None]
        dcMG = delta_c_d[start:end, None]
        dgMG = delta_g_d[start:end, None]
        mcMG = mu_c_d[start:end, None]
        mgMG = mu_g_d[start:end, None]
        tcMG = tau_c_d[start:end, None]
        tgMG = tau_g_d[start:end, None]

        sigma_intx_sq = sixMG**2
        sigma1_sq = sigma_intx_sq + sigma_x_star[None, :]**2

        A11 = gcMG**2 * tcMG**2 + ggMG**2 * tgMG**2 + siyMG**2 + sigma_y_star[None, :]**2
        A12 = gcMG * (gcMG - 1) * tcMG**2
        A14 = ggMG * (ggMG - 1) * tgMG**2
        A22 = (gcMG - 1)**2 * tcMG**2 + sizMG**2 + sigma_z_star[None, :]**2
        A44 = (ggMG - 1)**2 * tgMG**2 + sigMG**2 + sigma_g_star[None, :]**2

        D00 = sigma1_sq
        D01 = -dcMG * sigma_intx_sq
        D02 = -dgMG * sigma_intx_sq
        D11 = A22 + dcMG**2 * sigma_intx_sq
        D22 = A44 + dgMG**2 * sigma_intx_sq

        det_D = D00 * D11 * D22 - D01**2 * D22 - D02**2 * D11
        inv_det = 1.0 / det_D
        Di00 = D11 * D22 * inv_det
        Di11 = (D00 * D22 - D02**2) * inv_det
        Di22 = (D00 * D11 - D01**2) * inv_det
        Di01 = -D01 * D22 * inv_det
        Di02 = -D02 * D11 * inv_det
        Di12 = D01 * D02 * inv_det

        bc0 = Di01 * A12 + Di02 * A14
        bc1 = Di11 * A12 + Di12 * A14
        bc2 = Di12 * A12 + Di22 * A14

        cDinvc = A12 * (Di11 * A12 + Di12 * A14) + A14 * (Di12 * A12 + Di22 * A14)
        sigma_y_given_xzg_sq = A11 - cDinvc

        bxzg_0 = 1.0 / aMG
        bxzg_1 = 1.0 - dcMG / aMG
        bxzg_2 = 1.0 - dgMG / aMG

        xi = (bxzg_0 * (Di00 * bxzg_0 + Di01 * bxzg_1 + Di02 * bxzg_2)
              + bxzg_1 * (Di01 * bxzg_0 + Di11 * bxzg_1 + Di12 * bxzg_2)
              + bxzg_2 * (Di02 * bxzg_0 + Di12 * bxzg_1 + Di22 * bxzg_2))

        log1pz_centered = np.log1p(zobs_star[None, :]) - mean_log1pz
        alpha_zn_r_chunk = alpha_k_r_d[start:end, None] * log1pz_centered
        alpha_zn_z_chunk = alpha_k_z_d[start:end, None] * log1pz_centered
        alpha_zn_g_chunk = alpha_k_g_d[start:end, None] * log1pz_centered

        r0_x = xhat_star[None, :] + bMG / aMG
        r0_z = zhat_star[None, :] - alpha_zn_z_chunk + mcMG - dcMG * bMG / aMG - dcMG * x_bar
        r0_g = ghat_star[None, :] - alpha_zn_g_chunk + mgMG - dgMG * bMG / aMG - dgMG * x_bar

        Dinv_r0_0 = Di00 * r0_x + Di01 * r0_z + Di02 * r0_g
        Dinv_r0_1 = Di01 * r0_x + Di11 * r0_z + Di12 * r0_g
        Dinv_r0_2 = Di02 * r0_x + Di12 * r0_z + Di22 * r0_g
        phi = bxzg_0 * Dinv_r0_0 + bxzg_1 * Dinv_r0_1 + bxzg_2 * Dinv_r0_2

        mu_L = phi / xi
        sigma_L = np.sqrt(1.0 / xi)

        mu_chunk = np.empty_like(mu_L)
        var_chunk = np.empty_like(mu_L)

        deg = sigma_L == 0.0
        if np.any(deg):
            mu_chunk[deg] = mu_L[deg]
            var_chunk[deg] = 0.0

        nd = ~deg
        if np.any(nd):
            mu = mu_L[nd]
            sig = sigma_L[nd]
            alpha_tn = (a - mu) / sig
            beta_tn = (b - mu) / sig

            use_sf = alpha_tn >= 0.0
            log_sf_a = norm.logsf(alpha_tn)
            log_sf_b = norm.logsf(beta_tn)
            log_cdf_a = norm.logcdf(alpha_tn)
            log_cdf_b = norm.logcdf(beta_tn)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_Z_sf = log_sf_a + np.log1p(
                    -np.exp(np.clip(log_sf_b - log_sf_a, -np.inf, 0.0))
                )
                log_Z_cdf = log_cdf_b + np.log1p(
                    -np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0))
                )
            log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)
            log_Z = np.maximum(log_Z, np.log(Z_floor))

            la = np.exp(norm.logpdf(alpha_tn) - log_Z)
            lb = np.exp(norm.logpdf(beta_tn) - log_Z)
            t = la - lb
            u = alpha_tn * la - beta_tn * lb
            mu_chunk[nd] = mu + sig * t
            var_chunk[nd] = np.maximum(sig**2 * (1.0 + u - t**2), 0.0)

        mu_x = (mu_chunk - bMG) / aMG
        res0 = xhat_star[None, :] - mu_x
        res1 = zhat_star[None, :] - (mu_chunk + alpha_zn_z_chunk - mcMG - dcMG * (mu_x - x_bar))
        res2 = ghat_star[None, :] - (mu_chunk + alpha_zn_g_chunk - mgMG - dgMG * (mu_x - x_bar))
        cond_mean_chunk = mu_chunk + alpha_zn_r_chunk + bc0 * res0 + bc1 * res1 + bc2 * res2

        dres0_dyTF = -1.0 / aMG
        dres1_dyTF = -(1.0 - dcMG / aMG)
        dres2_dyTF = -(1.0 - dgMG / aMG)
        dmu_dyTF = 1.0 + bc0 * dres0_dyTF + bc1 * dres1_dyTF + bc2 * dres2_dyTF
        cond_var_chunk = sigma_y_given_xzg_sq + dmu_dyTF**2 * var_chunk

        mu_centered = cond_mean_chunk - mean_y[None, :]
        accum += mu_centered.T @ mu_centered
        var_accum += cond_var_chunk.sum(axis=0)

    cov = accum / M
    np.fill_diagonal(cov, np.diag(cov) + var_accum / M)
    return cov


def ystar_pp_cov_color_xonly_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    *,
    sigma_y_star,
    y_min,
    y_max,
    zobs_star,
    mean_log1pz,
    on_bad_Z="floor",
    Z_floor=1e-300,
    chunk_size=200,
):
    """
    Posterior predictive covariance Cov(ŷ*[g1], ŷ*[g2]) — color model, x̂ + redshift (no z-band).

    Marginalizes ẑ out (§sec:cc:x_only). The conditional mean includes the
    k-correction α_kcorr·log(1+z). Off-diagonal elements arise from shared
    uncertainty in θ (same as full model).
    """
    xhat_star = np.asarray(xhat_star, dtype=float)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)
    sigma_y_star = np.asarray(sigma_y_star, dtype=float)
    zobs_star = np.asarray(zobs_star, dtype=float)
    G = xhat_star.size

    mean_y, _ = ystar_pp_mean_sd_color_xonly_vectorized(
        draws, xhat_star, sigma_x_star,
        sigma_y_star=sigma_y_star,
        y_min=y_min, y_max=y_max,
        zobs_star=zobs_star, mean_log1pz=mean_log1pz,
        on_bad_Z=on_bad_Z, Z_floor=Z_floor,
    )

    a = float(y_min)
    b = float(y_max)

    alpha_d = draws["slope"].to_numpy(float)
    beta_d = draws["intercept.1"].to_numpy(float)
    six_d = draws["sigma_int_x"].to_numpy(float)
    siy_d = draws["sigma_int_y"].to_numpy(float)
    gamma_d = draws["gamma"].to_numpy(float)
    tau_c_d = draws["tau_c"].to_numpy(float)
    alpha_k_r_d = draws["alpha_kcorr_r"].to_numpy(float)
    has_2color = "gamma_g" in draws.columns and "tau_g" in draws.columns
    if has_2color:
        gamma_g_d = draws["gamma_g"].to_numpy(float)
        tau_g_d = draws["tau_g"].to_numpy(float)
    M = len(draws)

    accum = np.zeros((G, G), dtype=float)
    var_accum = np.zeros(G, dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)

        aMG = alpha_d[start:end, None]
        bMG = beta_d[start:end, None]
        sixMG = six_d[start:end, None]
        siyMG = siy_d[start:end, None]
        gMG = gamma_d[start:end, None]
        tcMG = tau_c_d[start:end, None]

        A11 = gMG**2 * tcMG**2 + siyMG**2 + sigma_y_star[None, :]**2
        if has_2color:
            ggMG = gamma_g_d[start:end, None]
            tgMG = tau_g_d[start:end, None]
            A11 = A11 + ggMG**2 * tgMG**2

        sigma1_sq = sixMG**2 + sigma_x_star[None, :]**2
        mu_L = bMG + aMG * xhat_star[None, :]
        sigma_L_sq = aMG**2 * sigma1_sq
        sigma_L = np.sqrt(sigma_L_sq)

        mu_chunk = np.empty_like(mu_L)
        var_chunk = np.empty_like(mu_L)

        deg = sigma_L == 0.0
        if np.any(deg):
            mu_chunk[deg] = mu_L[deg]
            var_chunk[deg] = 0.0

        nd = ~deg
        if np.any(nd):
            mu = mu_L[nd]
            sig = sigma_L[nd]
            alpha_tn = (a - mu) / sig
            beta_tn = (b - mu) / sig

            use_sf = alpha_tn >= 0.0
            log_sf_a = norm.logsf(alpha_tn)
            log_sf_b = norm.logsf(beta_tn)
            log_cdf_a = norm.logcdf(alpha_tn)
            log_cdf_b = norm.logcdf(beta_tn)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_Z_sf = log_sf_a + np.log1p(
                    -np.exp(np.clip(log_sf_b - log_sf_a, -np.inf, 0.0))
                )
                log_Z_cdf = log_cdf_b + np.log1p(
                    -np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0))
                )
            log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)
            log_Z = np.maximum(log_Z, np.log(Z_floor))

            la = np.exp(norm.logpdf(alpha_tn) - log_Z)
            lb = np.exp(norm.logpdf(beta_tn) - log_Z)
            t = la - lb
            u = alpha_tn * la - beta_tn * lb
            mu_chunk[nd] = mu + sig * t
            var_chunk[nd] = np.maximum(sig**2 * (1.0 + u - t**2), 0.0)

        # Conditional mean = mean_yTF + Δ_r (r-band k-correction)
        cond_mean_chunk = mu_chunk + alpha_k_r_d[start:end, None] * (np.log1p(zobs_star[None, :]) - mean_log1pz)  # (B, G)

        # Conditional variance = A₁₁ + V_★ (dmu_dyTF = 1)
        cond_var_chunk = A11 + var_chunk  # (B, G)

        mu_centered = cond_mean_chunk - mean_y[None, :]
        accum += mu_centered.T @ mu_centered
        var_accum += cond_var_chunk.sum(axis=0)

    cov = accum / M
    np.fill_diagonal(cov, np.diag(cov) + var_accum / M)
    return cov


def write_cov_color_xonly(run_dir, fits_path, cfg=None, model="color"):
    """
    Compute and save the posterior predictive covariance matrix for the color
    model using x̂ only (no z-band).

    Outputs:
      output/<run>/color_xonly_cov.fits  — full (G, G) float32 covariance matrix
    """
    from predict import plot_cov

    _p = lambda name: os.path.join(run_dir, name)

    if not cfg:
        with open(_p("config.json")) as f:
            cfg = json.load(f)

    with open(_p("input.json")) as f:
        input_data = json.load(f)
    y_min = input_data["y_min"]
    y_max = input_data["y_max"]
    mean_log1pz = float(np.mean(np.log1p(input_data["z_obs"])))

    (xhat_full, sigma_x_full, yhat_full, sigma_y_full,
     _zhat_full, _sigma_z_full, zobs_full) = load_xyz_and_uncertainties_from_desi(fits_path)

    rz_color_full = _load_rz_color_from_desi(fits_path)
    main = _apply_main_cuts(cfg, xhat_full, yhat_full, rz_color=rz_color_full)
    xhat_star = xhat_full[main]
    sigma_x_star = sigma_x_full[main]
    sigma_y_star = sigma_y_full[main]
    zobs_star = zobs_full[main]

    keep_cols = ["slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                 "gamma", "tau_c", "alpha_kcorr_r"]
    if model == "2color":
        keep_cols += ["gamma_g", "tau_g"]
    draws = read_cmdstan_posterior(
        _p(f"{model}_?.csv"),
        keep=keep_cols,
        drop_diagnostics=True,
    )

    cov = ystar_pp_cov_color_xonly_vectorized(
        draws, xhat_star, sigma_x_star,
        sigma_y_star=sigma_y_star,
        y_min=y_min, y_max=y_max,
        zobs_star=zobs_star,
        mean_log1pz=mean_log1pz,
    )

    # Add dust and photometric-calibration off-diagonal systematics
    with fits.open(fits_path) as _hdul:
        _t = _hdul[1].data
    _tmain = _t[np.array(main, dtype=bool)]
    ba_star    = np.array(_tmain['BA'],      dtype=float)
    photsys_star = np.array(_tmain['PHOTSYS'])
    _add_systematic_offdiag(cov, ba_star, photsys_star)

    fits_out = _p("color_xonly_cov.fits")
    hdr = fits.Header()
    hdr["COMMENT"] = "Posterior predictive covariance matrix (float32), x-hat only"
    hdr["COMMENT"] = f"Row/col order: MAIN=True rows of color_xonly_catalog.fits"
    hdr["MODEL"] = "color_xonly"
    hdr["RUN"] = os.path.basename(run_dir)
    fits.writeto(fits_out, cov.astype(np.float32), header=hdr, overwrite=True)
    print(f"Saved xonly covariance FITS to {fits_out}")

    G = cov.shape[0]
    n_sub = min(512, G)
    rng = np.random.default_rng(0)
    idx = rng.choice(G, size=n_sub, replace=False)
    idx.sort()
    cov_sub = cov[np.ix_(idx, idx)]
    plot_cov(cov_sub, _p("color_xonly_cov_sub.png"))


def write_cov_color(run_dir, fits_path, cfg=None, model="color"):
    """
    Compute and save the posterior predictive covariance matrix for the color model.

    Outputs:
      output/<run>/color_cov.fits         — full (G, G) float32 covariance matrix
      output/<run>/color_cov.png          — covariance + correlation visualization
      output/<run>/color_cov_sub.png      — same for a random subset ≤512 galaxies
      output/<run>/color_cov_sub_noobs.png — subset without obs-magnitude diagonal
    """
    from predict import plot_cov

    _p = lambda name: os.path.join(run_dir, name)

    if not cfg:
        with open(_p("config.json")) as f:
            cfg = json.load(f)

    with open(_p("input.json")) as f:
        input_data = json.load(f)
    y_min = input_data["y_min"]
    y_max = input_data["y_max"]
    mean_log1pz = float(np.mean(np.log1p(input_data["z_obs"])))

    # Load MAIN-sample galaxies
    (xhat_full, sigma_x_full, yhat_full, sigma_y_full,
     zhat_full, sigma_z_full, zobs_full) = load_xyz_and_uncertainties_from_desi(fits_path)
    x_bar = input_data.get("mean_x", float(np.mean(input_data["x"])))

    rz_color_full = _load_rz_color_from_desi(fits_path)
    main = _apply_main_cuts(cfg, xhat_full, yhat_full, rz_color=rz_color_full)
    xhat_star = xhat_full[main]
    sigma_x_star = sigma_x_full[main]
    sigma_y_star = sigma_y_full[main]
    zhat_star = zhat_full[main]
    sigma_z_star = sigma_z_full[main]
    zobs_star = zobs_full[main]

    if model == "2color":
        draws = read_cmdstan_posterior(
            _p(f"{model}_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                  "sigma_int_z", "sigma_int_g", "gamma", "gamma_g",
                  "delta_c", "delta_g", "mu_c", "mu_g", "tau_c", "tau_g",
                  "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g"],
            drop_diagnostics=True,
        )
        ghat_full_g, sigma_g_full_g = load_gband_from_desi(fits_path)
        ghat_star = ghat_full_g[main]
        sigma_g_star = sigma_g_full_g[main]

        cov = ystar_pp_cov_2color_vectorized(
            draws, xhat_star, sigma_x_star, zhat_star, sigma_z_star,
            ghat_star, sigma_g_star,
            sigma_y_star=sigma_y_star,
            x_bar=x_bar, y_min=y_min, y_max=y_max,
            zobs_star=zobs_star,
            mean_log1pz=mean_log1pz,
        )
    else:
        draws = read_cmdstan_posterior(
            _p(f"{model}_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y",
                  "sigma_int_z", "gamma", "delta_c", "mu_c", "tau_c",
                  "alpha_kcorr_r", "alpha_kcorr_z"],
            drop_diagnostics=True,
        )

        cov = ystar_pp_cov_color_vectorized(
            draws, xhat_star, sigma_x_star, zhat_star, sigma_z_star,
            sigma_y_star=sigma_y_star,
            x_bar=x_bar, y_min=y_min, y_max=y_max,
            zobs_star=zobs_star,
            mean_log1pz=mean_log1pz,
        )

    # Add dust and photometric-calibration off-diagonal systematics
    with fits.open(fits_path) as _hdul:
        _t = _hdul[1].data
    _tmain = _t[np.array(main, dtype=bool)]
    ba_star    = np.array(_tmain['BA'],      dtype=float)
    photsys_star = np.array(_tmain['PHOTSYS'])
    _add_systematic_offdiag(cov, ba_star, photsys_star)

    G = cov.shape[0]
    n_sub = min(512, G)
    rng = np.random.default_rng(0)
    idx = rng.choice(G, size=n_sub, replace=False)
    idx.sort()

    # σ²_{y,★} is already included in the diagonal via A₁₁; no further addition needed.
    cov_sub_noobs = cov[np.ix_(idx, idx)]
    plot_cov(cov_sub_noobs, _p("color_cov_sub_noobs.png"))

    fits_out = _p("color_cov.fits")
    hdr = fits.Header()
    hdr["COMMENT"] = "Posterior predictive covariance matrix (float32)"
    hdr["COMMENT"] = f"Row/col order: MAIN=True rows of color_catalog.fits"
    hdr["MODEL"] = "color"
    hdr["RUN"] = os.path.basename(run_dir)
    fits.writeto(fits_out, cov.astype(np.float32), header=hdr, overwrite=True)
    print(f"Saved covariance FITS to {fits_out}")

    plot_cov(cov, _p("color_cov.png"))

    cov_sub = cov[np.ix_(idx, idx)]
    plot_cov(cov_sub, _p("color_cov_sub.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Color-correction model posterior predictions and diagnostics."
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory containing config.json, input.json, and color_?.csv files",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config (e.g. configs/dr1_v6_2color.json); sets run-dir",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Run name; reads output/<run>/ (alternative to --run-dir)",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=50,
        help="Grid resolution for diagnostic plots (default: 50)",
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        default=False,
        help="Skip writing color_catalog.fits",
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        default=False,
        help="Skip computing and writing color_cov.fits posterior predictive covariance matrix",
    )
    parser.add_argument(
        "--xonly",
        action="store_true",
        default=False,
        help="Also write color_xonly_catalog.fits using x̂ and redshift (no z-band)",
    )
    parser.add_argument(
        "--model",
        default="color",
        help="Model name for CSV glob pattern (default: color → color_?.csv)",
    )
    args = parser.parse_args()

    import json as _json
    if args.config or args.run:
        from config_utils import apply_config, run_dir_from_args
        apply_config(args)
        if not args.run_dir:
            args.run_dir = run_dir_from_args(args)
    _run_dir = args.run_dir or "."
    with open(os.path.join(_run_dir, "config.json")) as _f:
        _cfg = _json.load(_f)
    _fits_path = _cfg["fits_file"]

    DESI_color(
        run_dir=args.run_dir,
        grid_resolution_x=args.grid_resolution,
        grid_resolution_y=args.grid_resolution,
        xonly=args.xonly,
        model=args.model,
    )

    if not args.no_catalog:
        write_desi_catalog_color(_run_dir, _fits_path, cfg=_cfg, model=args.model)

    if not args.no_cov:
        write_cov_color(_run_dir, _fits_path, cfg=_cfg, model=args.model)

    if args.xonly:
        write_desi_catalog_color_xonly(_run_dir, _fits_path, cfg=_cfg, model=args.model)
        write_cov_color_xonly(_run_dir, _fits_path, cfg=_cfg, model=args.model)
