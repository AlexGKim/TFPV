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
)
from mag_utils import get_mag_cols


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
    x_bar,
    y_min,
    y_max,
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
        "sigma_int_y", "gamma", "delta_c", "mu_c", "tau_c".
    xhat_star : (G,) array — observed log-velocity
    sigma_x_star : (G,) array — uncertainty on x̂
    zhat_star : (G,) array — observed z-band absolute magnitude
    sigma_z_star : (G,) array — uncertainty on ẑ
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
    delta = draws["delta_c"].to_numpy(float)
    mu_c = draws["mu_c"].to_numpy(float)
    tau_c = draws["tau_c"].to_numpy(float)

    if np.any(alpha == 0):
        raise ValueError("Found slope == 0 in draws; model requires α ≠ 0.")

    # Broadcast to (M, G)
    aMG = alpha[:, None]
    bMG = beta[:, None]
    sixMG = six[:, None]
    siyMG = siy[:, None]
    gMG = gamma[:, None]
    dMG = delta[:, None]
    mcMG = mu_c[:, None]
    tcMG = tau_c[:, None]

    # Per-galaxy, per-draw quantities
    sigma1_sq = sixMG**2 + sigma_x_star[None, :] ** 2  # (M, G)

    # A matrix entries (Eq. C.17) — at prediction time, σ_{y,*} = 0
    A11 = gMG**2 * tcMG**2 + siyMG**2  # (M, G) scalar broadcast
    A12 = gMG * (gMG - 1) * tcMG**2 + siyMG**2  # (M, G)
    A22 = (gMG - 1) ** 2 * tcMG**2 + siyMG**2 + sigma_z_star[None, :] ** 2  # (M, G)

    # b vector from D^{-1} [0, A12]^T (Eq. C.28)
    # D = [[σ1², -δσ1²], [-δσ1², A22 + δ²σ1²]]
    # det(D) = σ1²·A22
    # b = D^{-1} [0, A12]^T
    # D^{-1} = (1/det) [[A22+δ²σ1², δσ1²], [δσ1², σ1²]]
    # b[0] = (δσ1² · A12) / det_D = δ · A12 / A22
    # b[1] = (σ1² · A12) / det_D = A12 / A22
    b0 = dMG * A12 / A22  # (M, G)
    b1 = A12 / A22  # (M, G)

    # Conditional variance σ²_{y|x̂ẑ} = A11 - b^T D b
    # = A11 - [b0, b1] · D · [b0, b1]^T
    # = A11 - A12² / A22  (simpler form from Schur complement)
    sigma_y_given_xz_sq = A11 - A12**2 / A22  # (M, G)

    # ---- Step 4: Truncated normal posterior for y_TF | x̂ ----
    # μ_L = β + α·x̂_*, σ_L² = α²·σ1²
    mu_L = bMG + aMG * xhat_star[None, :]  # (M, G)
    sigma_L_sq = aMG**2 * sigma1_sq  # (M, G)
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
    # μ_{y|x̂ẑ}(y_TF) = y_TF + b^T · [x̂ - μ_x(y_TF), ẑ - (y_TF - μ_c - δ(μ_x(y_TF) - x̄))]
    # where μ_x(y_TF) = (y_TF - β) / α
    #
    # This is linear in y_TF, so E[μ_{y|x̂ẑ}(y_TF)] = μ_{y|x̂ẑ}(E[y_TF]) exactly.
    #
    # Compute μ_x(mean_yTF):
    mu_x_at_mean = (mean_yTF - bMG) / aMG  # (M, G)

    # Residual vector at y_TF = mean_yTF:
    res0 = xhat_star[None, :] - mu_x_at_mean  # (M, G)
    res1 = zhat_star[None, :] - (
        mean_yTF - mcMG - dMG * (mu_x_at_mean - x_bar)
    )  # (M, G)

    # E[ŷ | x̂, ẑ, θ] = mean_yTF + b0*res0 + b1*res1
    cond_mean = mean_yTF + b0 * res0 + b1 * res1  # (M, G)

    # ---- Step 6: Conditional variance Var[ŷ | x̂, ẑ, θ] ----
    # ∂μ_{y|x̂ẑ}/∂y_TF = 1 + b^T · ∂residuals/∂y_TF
    # ∂res0/∂y_TF = -1/α
    # ∂res1/∂y_TF = -(1 - δ·(-1/α)) = -(1 + δ/α)
    dres0_dyTF = -1.0 / aMG  # (M, G)
    dres1_dyTF = -(1.0 + dMG / aMG)  # (M, G)

    dmu_dyTF = 1.0 + b0 * dres0_dyTF + b1 * dres1_dyTF  # (M, G)

    # Var[ŷ | x̂, ẑ, θ] = σ²_{y|x̂ẑ} + (∂μ/∂y_TF)² · Var(y_TF)
    cond_var = sigma_y_given_xz_sq + dmu_dyTF**2 * var_yTF  # (M, G)

    # ---- Step 7: Mix over MCMC draws ----
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

    # Load galaxy data
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zhat_star, sigma_z_star, zobs_star = (
        load_xyz_and_uncertainties_from_desi(galaxy_fits)
    )

    # Compute x_bar from data if not in input.json
    if x_bar is None:
        x_bar = float(np.mean(xhat_star))

    # Load posterior draws
    draws = read_cmdstan_posterior(
        _p("color_?.csv"),
        keep=[
            "slope",
            "intercept.1",
            "sigma_int_x",
            "sigma_int_y",
            "gamma",
            "delta_c",
            "mu_c",
            "tau_c",
        ],
        drop_diagnostics=True,
    )

    # Posterior predictive
    mean_pred, sd_pred = ystar_pp_mean_sd_color_vectorized(
        draws,
        xhat_star,
        sigma_x_star,
        zhat_star,
        sigma_z_star,
        x_bar=x_bar,
        y_min=y_min,
        y_max=y_max,
        on_bad_Z="floor",
        Z_floor=1e-300,
    )

    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    # MAIN sample mask
    main_mask = _apply_main_cuts(cfg, xhat_star, yhat_star)

    xhat_main = xhat_star[main_mask]
    sigma_x_main = sigma_x_star[main_mask]
    yhat_main = yhat_star[main_mask]
    zhat_main = zhat_star[main_mask]
    sigma_z_main = sigma_z_star[main_mask]
    zobs_main = zobs_star[main_mask]

    mean_pred_main, sd_pred_main = ystar_pp_mean_sd_color_vectorized(
        draws,
        xhat_main,
        sigma_x_main,
        zhat_main,
        sigma_z_main,
        x_bar=x_bar,
        y_min=y_min,
        y_max=y_max,
        on_bad_Z="floor",
        Z_floor=1e-300,
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
    plt.ylim((-8, 4))
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

    return mean_y, sigma_y, zobs_star


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
        "--grid-resolution",
        type=int,
        default=50,
        help="Grid resolution for diagnostic plots (default: 50)",
    )
    args = parser.parse_args()

    DESI_color(
        run_dir=args.run_dir,
        grid_resolution_x=args.grid_resolution,
        grid_resolution_y=args.grid_resolution,
    )
