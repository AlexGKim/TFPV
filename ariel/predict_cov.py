"""
predict_cov.py — memory-efficient posterior predictive covariance of absolute magnitudes.

Cov(y*[g1], y*[g2]) = (1/M) Σ_m μ[m,g1]·μ[m,g2]  −  mean_y[g1]·mean_y[g2]

where μ[m,g] is the per-draw conditional mean E[y*[g] | xhat[g], θ_m].

Uses chunked matrix multiply so only O(chunk_size * G) intermediate memory
is needed rather than O(M * G).
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from predict import (
    ystar_pp_mean_sd_normal_vectorized,
    ystar_pp_mean_sd_tophat_vectorized,
    _phi,
)


def ystar_pp_cov_normal_vectorized(draws, xhat_star, sigma_x_star, chunk_size=200):
    """
    Posterior predictive covariance Cov(y*[g1], y*[g2]) — Normal y_TF prior.

    Uses chunked matrix multiply to avoid storing the full (M, G) mu_post matrix.
    Peak intermediate memory: O(chunk_size * G).

    Parameters
    ----------
    draws : pandas.DataFrame
        Must contain columns: "slope", "intercept.1", "sigma_int_x",
        "sigma_int_y", "mu_y_TF", "tau".
    xhat_star, sigma_x_star : array-like, shape (G,)
    chunk_size : int
        Number of draws to process per chunk.

    Returns
    -------
    cov : (G, G) ndarray
    """
    xhat_star    = np.asarray(xhat_star,    dtype=float)  # (G,)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)  # (G,)
    G = xhat_star.size
    M = len(draws)

    mean_y, _ = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)

    s   = draws["slope"].to_numpy(float)        # (M,)
    c   = draws["intercept.1"].to_numpy(float)  # (M,)
    six = draws["sigma_int_x"].to_numpy(float)  # (M,)
    mu0 = draws["mu_y_TF"].to_numpy(float)      # (M,)
    tau = draws["tau"].to_numpy(float)          # (M,)

    accum = np.zeros((G, G), dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        sc  = s[start:end][:, None]
        cc  = c[start:end][:, None]
        six_c = six[start:end][:, None]
        mu0_c = mu0[start:end][:, None]
        V0    = tau[start:end][:, None] ** 2

        sigma_x_tot2 = six_c**2 + sigma_x_star[None, :]**2  # (B, G)
        mu_L = cc + sc * xhat_star[None, :]                  # (B, G)
        V_L  = sc**2 * sigma_x_tot2                          # (B, G)

        Vp        = 1.0 / (1.0 / V0 + 1.0 / V_L)
        mu_chunk  = Vp * (mu0_c / V0 + mu_L / V_L)          # (B, G)

        accum += mu_chunk.T @ mu_chunk  # (G, G) rank-B update

    cov = accum / M - np.outer(mean_y, mean_y)
    return cov


def ystar_pp_cov_tophat_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    *,
    bounds_json=None,
    y_min=None,
    y_max=None,
    on_bad_Z="floor",
    Z_floor=1e-300,
    chunk_size=200,
):
    """
    Posterior predictive covariance Cov(y*[g1], y*[g2]) — Top-Hat y_TF prior.

    Uses chunked matrix multiply to avoid storing the full (M, G) mean_yTF matrix.
    Peak intermediate memory: O(chunk_size * G).

    Parameters
    ----------
    draws : pandas.DataFrame
        Must contain columns: "slope", "intercept.1", "sigma_int_x", "sigma_int_y".
    xhat_star, sigma_x_star : array-like, shape (G,)
    bounds_json : str/path-like or None
        JSON file containing scalar y_min and y_max (only read if y_min/y_max
        are not provided directly).
    y_min, y_max : float or None
        Scalar bounds of the Top-Hat prior on y_TF.
    on_bad_Z : {"raise", "floor"}
        Handling when Z = Phi(beta)-Phi(alpha) is non-positive/non-finite.
    Z_floor : float
        Floor used when on_bad_Z="floor".
    chunk_size : int
        Number of draws to process per chunk.

    Returns
    -------
    cov : (G, G) ndarray
    """
    xhat_star    = np.asarray(xhat_star,    dtype=float)  # (G,)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)  # (G,)
    G = xhat_star.size
    M = len(draws)

    # Load scalar bounds if not provided directly
    if y_min is None or y_max is None:
        if bounds_json is None:
            raise ValueError("Provide scalar y_min and y_max, or set bounds_json to a JSON file path.")
        with Path(bounds_json).open("r") as f:
            stan_data = json.load(f)
        y_min = stan_data["y_min"]
        y_max = stan_data["y_max"]

    a = float(y_min)
    b = float(y_max)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("y_min and y_max must be finite scalars.")
    if not (a < b):
        raise ValueError(f"Require y_min < y_max; got y_min={a}, y_max={b}.")

    mean_y, _ = ystar_pp_mean_sd_tophat_vectorized(
        draws, xhat_star, sigma_x_star,
        y_min=y_min, y_max=y_max,
        on_bad_Z=on_bad_Z, Z_floor=Z_floor,
    )

    s   = draws["slope"].to_numpy(float)        # (M,)
    c   = draws["intercept.1"].to_numpy(float)  # (M,)
    six = draws["sigma_int_x"].to_numpy(float)  # (M,)

    accum = np.zeros((G, G), dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        sc    = s[start:end][:, None]
        cc    = c[start:end][:, None]
        six_c = six[start:end][:, None]

        sigma_x_tot2 = six_c**2 + sigma_x_star[None, :]**2  # (B, G)
        mu_L    = cc + sc * xhat_star[None, :]               # (B, G)
        sigma_L = np.sqrt(sc**2 * sigma_x_tot2)             # (B, G)

        # Truncated-normal conditional mean for each (draw, galaxy)
        mu_chunk = np.empty_like(mu_L)

        deg = (sigma_L == 0.0)
        if np.any(deg):
            mu_deg = mu_L[deg]
            ok = (mu_deg >= a) & (mu_deg <= b)
            if not np.all(ok):
                raise ValueError(
                    "Encountered sigma_L == 0 with mu_L outside [y_min, y_max] "
                    "for at least one (draw, galaxy)."
                )
            mu_chunk[deg] = mu_deg

        nd = ~deg
        if np.any(nd):
            mu  = mu_L[nd]
            sig = sigma_L[nd]

            alpha = (a - mu) / sig
            beta  = (b - mu) / sig

            Z = norm.cdf(beta) * (1.0 - np.exp(norm.logcdf(alpha) - norm.logcdf(beta)))

            if on_bad_Z == "raise":
                if np.any(~np.isfinite(Z)) or np.any(Z <= 0.0):
                    raise ValueError(
                        "Truncation normalizer Z is non-finite or non-positive "
                        "for some (draw, galaxy)."
                    )
            elif on_bad_Z == "floor":
                Z = np.where(np.isfinite(Z), np.maximum(Z, Z_floor), Z_floor)
            else:
                raise ValueError("on_bad_Z must be 'raise' or 'floor'.")

            t = (_phi(alpha) - _phi(beta)) / Z
            mu_chunk[nd] = mu + sig * t

        accum += mu_chunk.T @ mu_chunk  # (G, G) rank-B update

    cov = accum / M - np.outer(mean_y, mean_y)
    return cov
