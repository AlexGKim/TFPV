#!/usr/bin/env python3
"""
cut_sweep.py — Quantitative determination of TFR selection cuts.

For each combination of cut parameters on a 5-D grid:
  1. Apply cuts to the raw galaxy data
  2. Fit the tophat model likelihood by fast Python MLE (scipy)
  3. Record slope ± σ_slope, profile log-likelihood, N_sel

Computes two complementary optimality criteria:

  volatility(g)   = mean |Δslope / sqrt(σ_g² + σ_g'²)| over adjacent
                    neighbors g'.  Minimum → most stable, least cut-sensitive.

  loglike(g)      = profile log-likelihood at the MLE.
                    Maximum → selection model most consistent with data.

On fullmocks, an optional third criterion is available when --true_slope is
supplied:

  bias(g)         = (slope(g) − slope_true) / σ_slope(g).
                    |bias| → 0 at the optimal cuts.

Usage:
  # Step 1 — run the sweep (writes cut_sweep.csv, no plots):
  python cut_sweep.py sweep --source fullmocks \\
      --fits_file data/TF_extended_AbacusSummit_base_c000_ph000_r001_z0.11.fits \\
      --run c000_ph000_r001

  python cut_sweep.py sweep --source DESI --run DESI

  # Override grid range/resolution for any parameter:
  python cut_sweep.py sweep --source fullmocks --fits_file ... --run test \\
      --haty_max_range -21.5 -19.0 --haty_max_n 4 \\
      --slope_plane_range -8.0 -5.0 --slope_plane_n 4

  # Step 2 — generate plots and recommendations from the saved CSV:
  python cut_sweep.py recommend --run c000_ph000_r001

  # Write the best config JSON:
  python cut_sweep.py recommend --run c000_ph000_r001 --write_best

Output (all in output/<run>/):
  cut_sweep.csv                   — full grid results
  cut_sweep_1d.png                — 1-D slope profiles (one panel per parameter)
  cut_sweep_2d_<p1>_<p2>.png     — 2-D slices (slope / volatility / loglike)
  cut_sweep_corner.png            — corner plot (1-D diagonal + 2-D lower triangle)
  cut_sweep_best_config.json      — optimal cut parameters
"""

import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import multivariate_normal as _mvn

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_fullmocks(fits_file):
    """Load raw (uncut, MAIN=True) galaxy data from an AbacusSummit FITS file.

    Returns a dict with keys: x, sigma_x, y, sigma_y, z_obs  (1-D float arrays).
    x = log10(V_rot / 100 km/s) = LOGVROT − 2.
    """
    print(f"Loading fullmocks FITS: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        main_mask = np.asarray(data["MAIN"], dtype=bool)
        d = data[main_mask]

    logvrot     = np.asarray(d["LOGVROT"],           dtype=float)
    logvrot_err = np.asarray(d["LOGVROT_ERR"],       dtype=float)
    absmag      = np.asarray(d["R_ABSMAG_SB26"],     dtype=float)
    absmag_err  = np.asarray(d["R_ABSMAG_SB26_ERR"], dtype=float)
    zobs        = np.asarray(d["ZOBS"],              dtype=float)

    x_raw = logvrot - 2.0
    valid = (
        np.isfinite(x_raw) & np.isfinite(logvrot_err)
        & np.isfinite(absmag) & np.isfinite(absmag_err)
        & np.isfinite(zobs)
        & (logvrot > 0) & (logvrot_err > 0) & (absmag_err >= 0)
    )
    print(f"  MAIN=True: {int(main_mask.sum())}  |  valid: {int(valid.sum())}")
    return dict(
        x       = x_raw[valid],
        sigma_x = logvrot_err[valid],
        y       = absmag[valid],
        sigma_y = absmag_err[valid],
        z_obs   = zobs[valid],
    )


def load_desi(fits_file="data/DESI-DR1_TF_pv_cat_v15.fits"):
    """Load raw DESI TF galaxy data from FITS file.

    Columns: V_0p4R26, V_0p4R26_ERR, R_ABSMAG_SB26, R_ABSMAG_SB26_ERR, ZOBS.
    x = log10(V_0p4R26 / 100 km/s).
    """
    V0 = 100.0
    print(f"Loading DESI FITS: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data

    V     = np.asarray(data["V_0p4R26"],         dtype=float)
    V_err = np.asarray(data["V_0p4R26_ERR"],     dtype=float)
    mag   = np.asarray(data["R_ABSMAG_SB26"],    dtype=float)
    mag_e = np.asarray(data["R_ABSMAG_SB26_ERR"], dtype=float)

    # Redshift — try several candidate column names
    zobs = None
    for col in ("ZOBS", "Z_OBS", "zobs", "Z", "ZHELIO"):
        if col in data.names:
            zobs = np.asarray(data[col], dtype=float)
            break
    if zobs is None:
        zobs = np.zeros(len(V))

    valid = (
        np.isfinite(V) & np.isfinite(V_err)
        & np.isfinite(mag) & np.isfinite(mag_e)
        & (V > 0) & (V_err > 0) & (mag_e >= 0)
    )
    x     = np.log10(V[valid] / V0)
    sx    = V_err[valid] / (V[valid] * np.log(10.0))
    print(f"  Valid rows: {int(valid.sum())}")
    return dict(
        x       = x,
        sigma_x = sx,
        y       = mag[valid],
        sigma_y = mag_e[valid],
        z_obs   = zobs[valid],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SELECTION CUTS
# ─────────────────────────────────────────────────────────────────────────────

def apply_cuts(raw_data, cuts):
    """Apply selection cuts and return filtered (x, sigma_x, y, sigma_y) arrays.

    cuts is a dict with keys:
      haty_max, haty_min           — magnitude window
      slope_plane, intercept_plane — lower oblique boundary (required)
      intercept_plane2             — upper oblique boundary (optional)
      z_obs_min, z_obs_max         — redshift window (optional)
    """
    x   = raw_data["x"]
    sx  = raw_data["sigma_x"]
    y   = raw_data["y"]
    sy  = raw_data["sigma_y"]
    z   = raw_data.get("z_obs", np.zeros(len(x)))

    mask = (y > cuts["haty_min"]) & (y < cuts["haty_max"])

    z_obs_min = cuts.get("z_obs_min")
    z_obs_max = cuts.get("z_obs_max")
    if z_obs_min is not None:
        mask &= z > z_obs_min
    if z_obs_max is not None:
        mask &= z <= z_obs_max

    sp  = cuts.get("slope_plane")
    ip  = cuts.get("intercept_plane")
    ip2 = cuts.get("intercept_plane2")
    if sp is not None and ip is not None:
        lb = np.maximum(cuts["haty_min"], sp * x + ip)
        mask &= y >= lb
        if ip2 is not None:
            ub = np.minimum(cuts["haty_max"], sp * x + ip2)
            mask &= y <= ub

    return x[mask], sx[mask], y[mask], sy[mask]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: TOPHAT MODEL LOG-LIKELIHOOD (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _log_diff_exp(log_a, log_b):
    """Numerically stable log(exp(log_a) − exp(log_b)) assuming log_a >= log_b."""
    diff = np.clip(log_b - log_a, -700.0, 0.0)
    result = log_a + np.log1p(-np.exp(diff))
    return np.where(np.isfinite(result), result, -700.0)


# 8-point GL nodes/weights matching Stan's gl_x_arr_8 / gl_w_arr_8
_GL_X8 = np.array([-0.9602898564975362317, -0.7966664774136267396,
                   -0.5255324099163289858, -0.1834346424956498049,
                    0.1834346424956498049,  0.5255324099163289858,
                    0.7966664774136267396,  0.9602898564975362317])
_GL_W8 = np.array([ 0.1012285362903762591,  0.2223810344533744861,
                    0.3137066458778872873,  0.3626837833783619830,
                    0.3626837833783619830,  0.3137066458778872873,
                    0.2223810344533744861,  0.1012285362903762591])


def _log_psel_bvn(y_min, y_max, haty_min, haty_max,
                  slope_std, intercept_std,
                  slope_plane_std, c1_std, c2_std,
                  sigma1, sigma2):
    """Log selection probability via bivariate normal strip integral.

    Computes log[(1/(y_max-y_min)) * integral_{y_min}^{y_max} P_sel(y_TF) dy_TF]

    where P_sel(y_TF) = P(haty_min <= haty <= haty_max,
                          c1_std <= haty - slope_plane_std*hatx_std <= c2_std | y_TF)

    Uses 8-point GL quadrature split at the midpoint of [haty_min, haty_max].
    Parameters are in standardised x units; sigma1 and sigma2 are total
    (intrinsic + observational) uncertainties in x_std and y respectively.
    """
    sigma_strip = np.sqrt(sigma2**2 + slope_plane_std**2 * sigma1**2)
    rho = float(np.clip(sigma2 / sigma_strip, -1 + 1e-9, 1 - 1e-9))
    cov = [[1.0, rho], [rho, 1.0]]

    y_range = y_max - y_min
    y_star = float(np.clip(0.5 * (haty_min + haty_max), y_min, y_max))

    def _integrate_piece(lo, hi):
        if hi <= lo:
            return 0.0
        mid  = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        ytf  = mid + half * _GL_X8                          # shape (8,)

        x_tf      = (ytf - intercept_std) / slope_std
        mu_strip  = ytf - slope_plane_std * x_tf

        z1_lo = (haty_min - ytf)    / sigma2
        z1_hi = (haty_max - ytf)    / sigma2
        z2_lo = (c1_std - mu_strip) / sigma_strip
        z2_hi = (c2_std - mu_strip) / sigma_strip

        # Batch all 4x8 = 32 bivariate normal CDF calls in one scipy call
        pts = np.column_stack([
            np.concatenate([z1_hi, z1_lo, z1_hi, z1_lo]),
            np.concatenate([z2_hi, z2_hi, z2_lo, z2_lo]),
        ])                                                   # shape (32, 2)
        cdfs = _mvn.cdf(pts, mean=[0.0, 0.0], cov=cov)
        p_rect = np.clip(cdfs[:8] - cdfs[8:16] - cdfs[16:24] + cdfs[24:32],
                         0.0, 1.0)
        return half * float(np.dot(_GL_W8, p_rect))

    p_sel = (_integrate_piece(y_min, y_star) +
             _integrate_piece(y_star, y_max)) / y_range
    return float(np.log(max(p_sel, 1e-300)))


def tophat_loglik(x_std, sigma_x_std, y, sigma_y,
                  slope_std, intercept_std, sigma_int_x, sigma_int_y,
                  y_min, y_max, haty_min, haty_max,
                  slope_plane=None, intercept_plane=None, intercept_plane2=None,
                  mean_x=0.0, sd_x=1.0):
    """Total tophat model log-likelihood, vectorised over N galaxies.

    Implements the Stan tophat.stan likelihood in Python/numpy:

      (1) y ~ N(intercept_std + slope_std * x_std,  sqrt(sigmasq_tot))
          [marginal likelihood with y_TF integrated out over flat prior]
      (2) + log|slope_std| * N                       [Jacobian]
      (3) + log[Φ((y_max−μ*)/σ*) − Φ((y_min−μ*)/σ*)]  [y_TF limits]
      (4) − log P_sel_i                              [selection correction]

    If slope_plane/intercept_plane/intercept_plane2 are provided, the selection
    bounds are tightened per-galaxy to account for the oblique cuts.
    """
    sigmasq1   = sigma_int_x**2 + sigma_x_std**2    # variance in x_std
    sigmasq2   = sigma_int_y**2 + sigma_y**2         # variance in y
    s2         = slope_std**2
    sigmasq_tot = s2 * sigmasq1 + sigmasq2
    sqrt_tot   = np.sqrt(sigmasq_tot)

    yfromx = intercept_std + slope_std * x_std

    # (1) marginal likelihood in y
    ll = norm.logpdf(y, yfromx, sqrt_tot)

    # (2) Jacobian for change-of-variables from y_TF to x (matches Stan)
    ll = ll + np.log(np.abs(slope_std))

    # (3) y_TF limits correction
    mu_star        = (yfromx * sigmasq2 + y * s2 * sigmasq1) / sigmasq_tot
    sigmasq_star   = s2 * sigmasq1 * sigmasq2 / sigmasq_tot
    sqrt_star      = np.sqrt(np.maximum(sigmasq_star, 1e-16))
    log_Phi_max    = norm.logcdf((y_max - mu_star) / sqrt_star)
    log_Phi_min    = norm.logcdf((y_min - mu_star) / sqrt_star)
    ll            += _log_diff_exp(log_Phi_max, log_Phi_min)

    # (4) Selection correction via bivariate normal strip integral
    if slope_plane is not None and intercept_plane is not None:
        # Convert to standardised plane parameters (matches Stan transformed data)
        sp_std = slope_plane * sd_x
        c1_std = intercept_plane  + sp_std * mean_x / sd_x
        c2_std = (intercept_plane2 + sp_std * mean_x / sd_x
                  if intercept_plane2 is not None else haty_max + 10.0)
        mean_sig1 = float(np.sqrt(np.mean(sigmasq1)))
        mean_sig2 = float(np.sqrt(np.mean(sigmasq2)))
        log_psel = _log_psel_bvn(y_min, y_max, haty_min, haty_max,
                                  slope_std, intercept_std,
                                  sp_std, c1_std, c2_std,
                                  mean_sig1, mean_sig2)
        ll -= log_psel          # same scalar subtracted from every galaxy
    else:
        log_Psel_max = norm.logcdf((haty_max - yfromx) / sqrt_tot)
        log_Psel_min = norm.logcdf((haty_min - yfromx) / sqrt_tot)
        ll -= _log_diff_exp(log_Psel_max, log_Psel_min)

    return float(np.sum(ll))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MLE FITTING
# ─────────────────────────────────────────────────────────────────────────────

_N_MIN = 30   # minimum galaxies to attempt a fit
_SLOPE_LO = -9.0   # MLE slope lower bound (original units)
_SLOPE_HI = -4.0   # MLE slope upper bound (original units)
_STABLE_SIGMA = 2.0  # stable if |slope - ref| < _STABLE_SIGMA * sigma


def fit_mle(x, sigma_x, y, sigma_y, haty_min, haty_max,
            slope_plane=None, intercept_plane=None, intercept_plane2=None):
    """Fit tophat model parameters by MLE for the given (already cut) sample.

    Returns a dict:
      slope, sigma_slope, intercept, sigma_int_x, sigma_int_y, loglike, N
    or None if the fit fails or N < _N_MIN.

    slope and sigma_slope are in original (unstandardised) units.
    """
    N = len(x)
    if N < _N_MIN:
        return None

    mean_x = float(np.mean(x))
    sd_x   = float(np.std(x, ddof=1))
    if sd_x < 1e-6:
        return None

    x_std    = (x - mean_x) / sd_x
    sx_std   = sigma_x / sd_x

    y_min = float(haty_min) - 0.5
    y_max = float(haty_max) + 1.0

    # Initial guess: OLS on standardised x
    p = np.polyfit(x_std, y, 1)
    slope0 = float(p[0])
    if slope0 >= 0:
        slope0 = -6.0 * sd_x
    intercept0 = float(p[1])
    theta0 = np.array([slope0, intercept0, 0.1, 0.1])

    # Bounds matching Stan priors (in standardised units)
    bounds = [
        (-9.0 * sd_x, -4.0 * sd_x),   # slope_std  (matches Stan prior)
        (-26.0, -10.0),                 # intercept_std
        (1e-4, 2.0),                    # sigma_int_x
        (1e-4, 2.0),                    # sigma_int_y
    ]

    def neg_ll(theta):
        s, b, six, siy = theta
        if s >= 0 or six <= 0 or siy <= 0:
            return 1e10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = -tophat_loglik(x_std, sx_std, y, sigma_y, s, b, six, siy,
                                 y_min, y_max, float(haty_min), float(haty_max),
                                 slope_plane=slope_plane,
                                 intercept_plane=intercept_plane,
                                 intercept_plane2=intercept_plane2,
                                 mean_x=mean_x, sd_x=sd_x)
        return val if np.isfinite(val) else 1e10

    res = minimize(neg_ll, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-6})

    if res.fun > 1e9:
        return None

    slope_std_mle, intercept_std_mle, six_mle, siy_mle = res.x

    # Diagonal Hessian only — 8 function evaluations instead of 64.
    # σ_slope ≈ 1 / sqrt(H[0,0]).  Sufficient for locating the stability plateau.
    eps      = np.maximum(1e-5 * np.abs(res.x), 1e-7)
    hess_diag = np.zeros(4)
    for i in range(4):
        ei        = np.zeros(4)
        ei[i]     = eps[i]
        hess_diag[i] = (neg_ll(res.x + ei) - 2.0 * res.fun
                        + neg_ll(res.x - ei)) / eps[i] ** 2

    sigma_slope_std = float(1.0 / max(np.sqrt(max(hess_diag[0], 1e-8)), 1e-8))

    # Convert back to original (unstandardised) units
    slope_orig       = slope_std_mle / sd_x
    sigma_slope_orig = sigma_slope_std / sd_x
    intercept_orig   = intercept_std_mle - slope_std_mle * mean_x / sd_x

    return dict(
        slope        = slope_orig,
        sigma_slope  = sigma_slope_orig,
        intercept    = intercept_orig,
        sigma_int_x  = float(six_mle),
        sigma_int_y  = float(siy_mle),
        loglike      = float(-res.fun),
        N            = N,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: GRID SWEEP (parallelised)
# ─────────────────────────────────────────────────────────────────────────────

# Worker-process global (avoids pickling raw_data for every task)
_WORKER_RAW_DATA = None

def _worker_init(raw_data):
    global _WORKER_RAW_DATA
    _WORKER_RAW_DATA = raw_data

def _worker_evaluate(cuts):
    x, sx, y, sy = apply_cuts(_WORKER_RAW_DATA, cuts)
    result = fit_mle(x, sx, y, sy, cuts["haty_min"], cuts["haty_max"],
                     slope_plane=cuts.get("slope_plane"),
                     intercept_plane=cuts.get("intercept_plane"),
                     intercept_plane2=cuts.get("intercept_plane2"))
    if result is None:
        result = dict(slope=np.nan, sigma_slope=np.nan, intercept=np.nan,
                      sigma_int_x=np.nan, sigma_int_y=np.nan,
                      loglike=np.nan, N=len(x))
    return result


def build_grid(grid_params, fixed_cuts):
    """Return list of cut dicts for all valid grid points."""
    keys   = list(grid_params.keys())
    combos = list(itertools.product(*[grid_params[k] for k in keys]))
    valid  = []
    for combo in combos:
        cuts = dict(zip(keys, combo))
        cuts.update(fixed_cuts)
        if cuts["haty_min"] >= cuts["haty_max"]:
            continue
        if ("intercept_plane2" in cuts
                and cuts["intercept_plane"] >= cuts["intercept_plane2"]):
            continue
        valid.append(cuts)
    return valid


def run_sweep(raw_data, grid_params, fixed_cuts, n_workers=None):
    """Sweep the grid and return a DataFrame of results."""
    combos  = build_grid(grid_params, fixed_cuts)
    n_total = len(combos)
    n_grid  = len(list(itertools.product(*grid_params.values())))
    print(f"Grid: {n_total} valid combinations ({n_grid} total).")

    n_workers = n_workers or min(mp.cpu_count(), n_total)
    print(f"Using {n_workers} worker(s).")

    results     = [None] * n_total
    print_every = max(1, n_total // 20)   # ~20 progress lines
    t_start     = time.perf_counter()

    with mp.Pool(n_workers,
                 initializer=_worker_init,
                 initargs=(raw_data,)) as pool:
        for i, res in enumerate(pool.imap(_worker_evaluate, combos, chunksize=1)):
            results[i] = res
            done = i + 1
            if done % print_every == 0 or done == n_total:
                elapsed   = time.perf_counter() - t_start
                rate      = done / elapsed
                remaining = (n_total - done) / rate
                print(f"  {done:4d}/{n_total}  ({100*done/n_total:3.0f}%)  "
                      f"elapsed {elapsed:5.0f}s  ETA {remaining:5.0f}s  "
                      f"({rate:.1f} pts/s)")

    param_cols = list(grid_params.keys())
    rows = []
    for cuts, res in zip(combos, results):
        row = {k: cuts[k] for k in param_cols}
        row.update({k: cuts.get(k) for k in fixed_cuts})
        row.update(res)
        rows.append(row)

    total_elapsed = time.perf_counter() - t_start
    print(f"Sweep complete: {total_elapsed:.1f}s total ({n_total/total_elapsed:.1f} pts/s)")
    return pd.DataFrame(rows), param_cols


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: STABILITY METRIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility(df, grid_params, param_cols):
    """Compute the volatility (slope stability) for each grid point.

    volatility(g) = mean over adjacent neighbors g' of
                      |slope(g) − slope(g')| / sqrt(σ(g)² + σ(g')²)

    'Adjacent' means differing in exactly one parameter by one grid step.
    """
    # Build index maps
    param_to_idx = {k: {round(v, 10): i
                        for i, v in enumerate(grid_params[k])}
                    for k in param_cols}
    n_dims = [len(grid_params[k]) for k in param_cols]

    # Build lookup dict: index_tuple -> (slope, sigma_slope)
    lookup = {}
    for _, row in df.iterrows():
        if np.isnan(row["slope"]) or np.isnan(row["sigma_slope"]):
            continue
        try:
            idx = tuple(param_to_idx[k][round(row[k], 10)] for k in param_cols)
        except KeyError:
            continue
        lookup[idx] = (float(row["slope"]), float(row["sigma_slope"]))

    volatilities = []
    for _, row in df.iterrows():
        if np.isnan(row["slope"]) or np.isnan(row["sigma_slope"]):
            volatilities.append(np.nan)
            continue
        try:
            idx = tuple(param_to_idx[k][round(row[k], 10)] for k in param_cols)
        except KeyError:
            volatilities.append(np.nan)
            continue

        slope_g, sigma_g = lookup.get(idx, (np.nan, np.nan))
        diffs = []
        for dim_i in range(len(param_cols)):
            for delta in (-1, +1):
                nbr = list(idx)
                nbr[dim_i] += delta
                if 0 <= nbr[dim_i] < n_dims[dim_i]:
                    nbr_key = tuple(nbr)
                    if nbr_key in lookup:
                        slope_n, sigma_n = lookup[nbr_key]
                        denom = np.sqrt(sigma_g**2 + sigma_n**2)
                        if denom > 0:
                            diffs.append(abs(slope_g - slope_n) / denom)
        volatilities.append(float(np.mean(diffs)) if diffs else np.nan)

    return volatilities


def compute_scores(df, lam=None):
    """Add 'volatility', 'score', and optionally 'bias' columns.

    score = loglike − lam * volatility   (higher is better).
    lam defaults to the median N_sel (so both terms have comparable magnitude).
    """
    if lam is None:
        lam = float(np.nanmedian(df["N"]))
    df["score"] = df["loglike"] - lam * df["volatility"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_LABELS = {
    "haty_max":         r"$\hat{y}_{\max}$",
    "haty_min":         r"$\hat{y}_{\min}$",
    "slope_plane":      r"$\bar{s}$ (plane slope)",
    "intercept_plane":  r"$c_1$ (lower intercept)",
    "intercept_plane2": r"$c_2$ (upper intercept)",
}


def _label(p):
    return _PARAM_LABELS.get(p, p)


def plot_1d(df, param_cols, best_cuts, true_slope, out_file, rec_cuts=None):
    """1-D slope ± σ profiles, one panel per parameter."""
    ncols = len(param_cols)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, p in zip(axes, param_cols):
        # Slice at best values of all other parameters
        mask = pd.Series(True, index=df.index)
        for q in param_cols:
            if q != p:
                mask &= df[q] == best_cuts[q]
        sub = df[mask].sort_values(p)
        ax.errorbar(sub[p], sub["slope"], yerr=sub["sigma_slope"],
                    fmt="o-", color="steelblue", capsize=4, linewidth=1.5,
                    label="slope ± σ")
        if true_slope is not None:
            ax.axhline(true_slope, color="crimson", linestyle="--",
                       linewidth=1.5, label=f"true slope = {true_slope:.2f}")
        ax.axhline(_SLOPE_LO, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                   label=f"bound ({_SLOPE_LO:.0f})")
        ax.axhline(_SLOPE_HI, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                   label=f"bound ({_SLOPE_HI:.0f})")
        ax.axvline(best_cuts[p], color="steelblue", linestyle=":", linewidth=1.5,
                   label=f"best ({best_cuts[p]:.2f})")
        if rec_cuts is not None and rec_cuts[p] != best_cuts[p]:
            ax.axvline(rec_cuts[p], color="orange", linestyle=":", linewidth=1.5,
                       label=f"rec ({rec_cuts[p]:.2f})")
        ax.set_xlabel(_label(p), fontsize=11)
        ax.set_ylabel("fitted TFR slope", fontsize=11)
        ax.set_title(_label(p), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("1-D slope profiles (others fixed at best cuts)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_2d(df, p1, p2, param_cols, best_cuts, out_file):
    """2-D heatmaps (slope, volatility, loglike) for a pair of parameters."""
    mask = pd.Series(True, index=df.index)
    for q in param_cols:
        if q not in (p1, p2):
            mask &= df[q] == best_cuts[q]
    sub = df[mask]
    if sub.empty:
        return

    v1 = sorted(sub[p1].unique())
    v2 = sorted(sub[p2].unique())
    shape = (len(v2), len(v1))

    def to_grid(col):
        grid = np.full(shape, np.nan)
        for _, row in sub.iterrows():
            if np.isnan(row[col]):
                continue
            i1 = v1.index(row[p1])
            i2 = v2.index(row[p2])
            grid[i2, i1] = row[col]
        return grid

    g_slope   = to_grid("slope")
    g_vol     = to_grid("volatility")
    g_ll      = to_grid("loglike")

    cols_info = [
        (g_slope,  "slope",      "RdYlGn",   False),
        (g_vol,    "volatility", "YlOrRd_r", False),
        (g_ll,     "log-like",   "viridis",  False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (grid, title, cmap, _) in zip(axes, cols_info):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                       extent=[min(v1), max(v1), min(v2), max(v2)])
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.axvline(best_cuts[p1], color="white", linewidth=1.2, linestyle="--")
        ax.axhline(best_cuts[p2], color="white", linewidth=1.2, linestyle="--")
        ax.set_xlabel(_label(p1), fontsize=10)
        ax.set_ylabel(_label(p2), fontsize=10)
        ax.set_title(title, fontsize=11)

    fig.suptitle(f"2-D slice: {_label(p1)} × {_label(p2)}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def plot_corner(df, param_cols, best_cuts, true_slope, out_file, rec_cuts=None):
    """Corner plot: 1-D slope profiles on diagonal, 2-D slope heatmaps on lower triangle."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    N = len(param_cols)
    if N == 0:
        return

    # Reference slope/sigma at best_cuts
    ref_mask = pd.Series(True, index=df.index)
    for p in param_cols:
        ref_mask &= df[p] == best_cuts[p]
    ref_rows = df[ref_mask].dropna(subset=["slope", "sigma_slope"])
    if ref_rows.empty:
        print("WARNING: plot_corner: no data at best_cuts — skipping.")
        return
    ref_slope = float(ref_rows["slope"].iloc[0])
    ref_sigma = float(ref_rows["sigma_slope"].iloc[0])
    all_slopes = df["slope"].dropna().values
    tol = 0.01
    interior_slopes = all_slopes[
        (all_slopes > _SLOPE_LO + tol) & (all_slopes < _SLOPE_HI - tol)
    ]
    color_slopes = interior_slopes if len(interior_slopes) >= 10 else all_slopes
    vmin_color = float(np.nanpercentile(color_slopes, 2))
    vmax_color = float(np.nanpercentile(color_slopes, 98))

    fig, axes = plt.subplots(N, N, figsize=(3.5 * N, 3.5 * N), constrained_layout=True)
    if N == 1:
        axes = np.array([[axes]])

    for i in range(N):
        for j in range(N):
            ax = axes[i, j]

            if j > i:
                ax.set_visible(False)
                continue

            pi = param_cols[i]
            pj = param_cols[j]

            if i == j:
                # Diagonal: 1-D slope profile
                p = param_cols[i]
                mask = pd.Series(True, index=df.index)
                for q in param_cols:
                    if q != p:
                        mask &= df[q] == best_cuts[q]
                sub = df[mask].sort_values(p)
                ax.errorbar(sub[p], sub["slope"], yerr=sub["sigma_slope"],
                            fmt="o-", color="steelblue", capsize=3, linewidth=1.2,
                            markersize=4)
                if true_slope is not None:
                    ax.axhline(true_slope, color="crimson", linestyle="--",
                               linewidth=1.2)
                ax.axvline(best_cuts[p], color="steelblue", linestyle=":",
                           linewidth=1.5)
                if rec_cuts is not None and rec_cuts[p] != best_cuts[p]:
                    ax.axvline(rec_cuts[p], color="orange", linestyle=":",
                               linewidth=1.5)
                ax.set_xlabel(_label(p), fontsize=8)
                ax.set_ylabel("slope", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

            else:
                # Lower triangle: 2-D slope heatmap
                # x-axis = pj (param_cols[j]), y-axis = pi (param_cols[i])
                mask = pd.Series(True, index=df.index)
                for q in param_cols:
                    if q not in (pi, pj):
                        mask &= df[q] == best_cuts[q]
                sub = df[mask]
                if sub.empty:
                    ax.set_visible(False)
                    continue

                v_x = sorted(sub[pj].unique())
                v_y = sorted(sub[pi].unique())
                shape = (len(v_y), len(v_x))
                slope_grid = np.full(shape, np.nan)
                for _, row in sub.iterrows():
                    if np.isnan(row["slope"]):
                        continue
                    ix = v_x.index(row[pj])
                    iy = v_y.index(row[pi])
                    slope_grid[iy, ix] = row["slope"]

                ax.imshow(slope_grid, aspect="auto", origin="lower",
                          cmap="RdYlGn",
                          vmin=vmin_color, vmax=vmax_color,
                          extent=[min(v_x), max(v_x), min(v_y), max(v_y)])

                # Stable-region boundary contours
                if not np.all(np.isnan(slope_grid)):
                    try:
                        xs = np.linspace(min(v_x), max(v_x), slope_grid.shape[1])
                        ys = np.linspace(min(v_y), max(v_y), slope_grid.shape[0])
                        ax.contour(xs, ys, slope_grid,
                                   levels=[ref_slope - _STABLE_SIGMA * ref_sigma,
                                           ref_slope + _STABLE_SIGMA * ref_sigma],
                                   colors="white", linestyles="dashed",
                                   linewidths=0.8)
                    except Exception:
                        pass

                # Blue crosshairs at best_cuts
                ax.axvline(best_cuts[pj], color="steelblue", linestyle="--",
                           linewidth=1.0)
                ax.axhline(best_cuts[pi], color="steelblue", linestyle="--",
                           linewidth=1.0)

                # Orange x at rec_cuts where they differ
                if rec_cuts is not None:
                    if rec_cuts[pj] != best_cuts[pj] or rec_cuts[pi] != best_cuts[pi]:
                        ax.plot(rec_cuts[pj], rec_cuts[pi], "x", color="orange",
                                markersize=8, markeredgewidth=1.5)

                ax.set_xlabel(_label(pj), fontsize=8)
                ax.set_ylabel(_label(pi), fontsize=8)
                ax.tick_params(labelsize=7)

    # Shared colorbar
    sm = cm.ScalarMappable(
        cmap="RdYlGn",
        norm=mcolors.Normalize(vmin=vmin_color, vmax=vmax_color))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="fitted TFR slope")

    fig.suptitle("Corner plot: slope phase space", fontsize=12)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def make_plots(df, param_cols, best_cuts, run_dir, true_slope=None, rec_cuts=None):
    """Generate all 1-D and 2-D plots."""
    out_1d = os.path.join(run_dir, "cut_sweep_1d.png")
    plot_1d(df, param_cols, best_cuts, true_slope, out_1d, rec_cuts=rec_cuts)

    for p1, p2 in itertools.combinations(param_cols, 2):
        safe1 = p1.replace("_", "")
        safe2 = p2.replace("_", "")
        out_2d = os.path.join(run_dir, f"cut_sweep_2d_{safe1}_{safe2}.png")
        plot_2d(df, p1, p2, param_cols, best_cuts, out_2d)

    out_corner = os.path.join(run_dir, "cut_sweep_corner.png")
    plot_corner(df, param_cols, best_cuts, true_slope, out_corner, rec_cuts=rec_cuts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: BEST-CUTS SELECTION AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def find_best(df, criterion="score"):
    """Return the row with the best (maximum) value of `criterion`."""
    sub = df.dropna(subset=[criterion])
    if sub.empty:
        return None
    return sub.loc[sub[criterion].idxmax()]


def find_best_stable_max_N(df, vol_threshold_factor=25.0):
    """Return the cut combination in the low-volatility plateau with the most galaxies.

    Algorithm
    ---------
    1. Exclude boundary-hitting fits (slope at _SLOPE_LO / _SLOPE_HI) from the
       working set ('interior').  Those fits have degenerate likelihoods; their
       anomalously near-zero volatility would bias the percentile threshold downward.
    2. Compute a volatility threshold = the vol_threshold_factor-th percentile of
       interior volatility.  All interior points at or below this threshold form the
       'plateau' — the set of least cut-sensitive configurations.
    3. Return the plateau point with the largest N_sel (number of selected galaxies).

    Percentile vs multiplicative threshold
    ----------------------------------------
    The old approach (vol_threshold = vol_min * factor) broke when vol_min was
    near zero: a tiny vol_min made the threshold too tight to span the physically
    meaningful plateau.  A percentile threshold is scale-independent — it always
    selects a fixed fraction of the distribution regardless of absolute values.

    Volatility definition (see also module docstring and compute_volatility)
    -------------------------------------------------------------------------
    volatility(g) = mean over adjacent grid neighbors g' of
                      |slope(g) − slope(g')| / sqrt(σ(g)² + σ(g')²)
    Low volatility → the fitted TFR slope is insensitive to small changes in the
    selection cuts, i.e., the cut combination is in a stable region of parameter space.

    Choosing max-N within the plateau
    ----------------------------------
    The volatility landscape can contain more than one low-volatility plateau.  For
    example, very loose cuts and very tight cuts may both yield a stable slope, but
    at different values.  Maximising N_sel within the plateau selects the solution
    with the most galaxies — typically the physically meaningful one (tight enough
    cuts to exclude contamination, loose enough to retain the full population).

    Parameters
    ----------
    vol_threshold_factor : float, optional
        Percentile (0–100) of the interior volatility distribution used as the
        plateau threshold.  Default 25 (bottom quarter = least-volatile 25% of
        grid points).  Increase to widen the plateau; decrease to restrict it.
    """
    sub = df.dropna(subset=["volatility", "N", "score"])
    if sub.empty:
        return None
    tol = 0.01
    interior = sub[
        (sub["slope"] > _SLOPE_LO + tol) & (sub["slope"] < _SLOPE_HI - tol)
    ]
    if interior.empty:
        interior = sub   # nothing to exclude — use everything
    vol_threshold = float(np.percentile(interior["volatility"], vol_threshold_factor))
    stable = interior[interior["volatility"] <= vol_threshold]
    return stable.loc[stable["N"].idxmax()]


def report_best(best_row, param_cols, true_slope=None):
    print("\n" + "=" * 60)
    print("BEST CUT PARAMETERS  (max-N in plateau)")
    print("=" * 60)
    for p in param_cols:
        print(f"  {p:25s} = {best_row[p]:.4f}")
    print(f"\n  slope            = {best_row['slope']:.4f} ± {best_row['sigma_slope']:.4f}")
    if true_slope is not None:
        bias = (best_row["slope"] - true_slope) / best_row["sigma_slope"]
        print(f"  true slope       = {true_slope:.4f}")
        print(f"  bias (in σ)      = {bias:.2f}")
    print(f"  N_sel            = {int(best_row['N'])}")
    print(f"  loglike          = {best_row['loglike']:.1f}")
    print(f"  volatility       = {best_row['volatility']:.4f}")
    print(f"  score            = {best_row['score']:.1f}")
    print("=" * 60)


def sweetspot_summary(df, param_cols, best_cuts, true_slope=None):
    """Print a human-readable sweet-spot summary after the sweep.

    For each cut parameter:
      - Classifies it as SENSITIVE or INSENSITIVE based on how much the slope
        varies across the 1-D profile (others fixed at best cuts).
      - For sensitive parameters, identifies the stable region and the
        recommended value (stable point with the largest selected sample).
    Also reports whether the log-likelihood is a reliable criterion.
    """
    W = 70
    STABLE_SIGMA = _STABLE_SIGMA  # a point is 'stable' if |slope - ref| < STABLE_SIGMA * σ
    SENS_SIGMA   = 3.0            # parameter is 'sensitive' if range > SENS_SIGMA * median(σ)

    # Reference slope at the best-score cuts
    ref_mask = pd.Series(True, index=df.index)
    for p in param_cols:
        ref_mask &= df[p] == best_cuts[p]
    ref_rows = df[ref_mask].dropna(subset=["slope", "sigma_slope"])
    if ref_rows.empty:
        print("WARNING: no data at best_cuts for reference slope.")
        return {}
    ref_slope = float(ref_rows["slope"].iloc[0])
    ref_sigma = float(ref_rows["sigma_slope"].iloc[0])

    print("\n" + "=" * W)
    print("SWEET SPOT SUMMARY")
    print("=" * W)
    print()
    print("  Criterion: within the stable (plateau) region, the loosest cut")
    print("  is always preferred — it maximises the number of galaxies and")
    print("  thereby minimises statistical uncertainty on the slope.")

    # ── log-likelihood note ────────────────────────────────────────────────
    print(f"\n  Log-likelihood note:")
    print(f"    Log-like is NOT directly comparable across cuts — looser cuts")
    print(f"    include more galaxies and therefore contribute more terms to the")
    print(f"    sum. Use log-like for qualitative guidance only; the volatility")
    print(f"    (slope stability) is the primary criterion.")

    print(f"\n  Reference slope at best cuts: {ref_slope:.4f} ± {ref_sigma:.4f}")
    if true_slope is not None:
        bias = (ref_slope - true_slope) / ref_sigma
        print(f"  True slope:                   {true_slope:.4f}")
        print(f"  Bias at best cuts:             {bias:+.2f} σ")

    # ── per-parameter analysis ─────────────────────────────────────────────
    print(f"\n{'─' * W}")
    recommendations = {}

    for p in param_cols:
        mask = pd.Series(True, index=df.index)
        for q in param_cols:
            if q != p:
                mask &= df[q] == best_cuts[q]
        sub = (df[mask]
               .sort_values(p)
               .dropna(subset=["slope", "sigma_slope", "N"])
               .reset_index(drop=True))
        if len(sub) < 2:
            continue

        slopes  = sub["slope"].values
        sigmas  = sub["sigma_slope"].values
        vals    = sub[p].values
        ns      = sub["N"].values

        tol = 0.01
        interior = (slopes > _SLOPE_LO + tol) & (slopes < _SLOPE_HI - tol)
        n_boundary = int((~interior).sum())

        i_slopes = slopes[interior]
        i_sigmas = sigmas[interior]
        i_vals   = vals[interior]
        i_ns     = ns[interior]

        print(f"\n  {_label(p)}  [{p}]")
        print(f"    Tested range: [{vals.min():.2f}, {vals.max():.2f}]  "
              f"({len(vals)} points)")

        if i_slopes.size == 0:
            print(f"    WARNING: all tested values have slope at optimizer boundary — "
                  f"widen the grid or tighten the cut range.")
            recommendations[p] = best_cuts[p]
            continue
        if n_boundary:
            print(f"    ({n_boundary} point(s) at optimizer slope boundary excluded)")

        slope_range   = float(i_slopes.max() - i_slopes.min())
        median_sigma  = float(np.median(i_sigmas))
        sensitive     = slope_range > SENS_SIGMA * median_sigma

        # Stable: slope within STABLE_SIGMA * σ of the reference
        combined_sigma = np.maximum(i_sigmas, ref_sigma)
        stable = np.abs(i_slopes - ref_slope) < STABLE_SIGMA * combined_sigma

        if not sensitive:
            print(f"    INSENSITIVE — slope varies by only {slope_range:.4f} "
                  f"({slope_range/median_sigma:.1f}σ) across the full range.")
            print(f"    All tested values are in the stable region.")
            # prefer the loosest cut (most galaxies) even when insensitive
            best_idx = int(np.argmax(i_ns))
            rec_val  = float(i_vals[best_idx])
            rec_note = (f"loosest value — maximises N_sel = {int(i_ns[best_idx])}")
        else:
            stable_vals = i_vals[stable]
            stable_ns   = i_ns[stable]
            unstable    = i_vals[~stable]
            print(f"    SENSITIVE — slope varies by {slope_range:.4f} "
                  f"({slope_range/median_sigma:.1f}σ).")
            print(f"    Unstable values (slope deviates > {STABLE_SIGMA}σ): "
                  f"{[f'{v:.2f}' for v in unstable]}")
            if len(stable_vals) > 0:
                print(f"    Stable region: [{stable_vals.min():.2f}, "
                      f"{stable_vals.max():.2f}]  "
                      f"(slope ≈ {i_slopes[stable].mean():.4f})")
                # prefer the loosest stable cut (most galaxies)
                best_idx = int(np.argmax(stable_ns))
                rec_val  = float(stable_vals[best_idx])
                rec_note = (f"loosest stable value — maximises N_sel = "
                            f"{int(stable_ns[best_idx])}")
            else:
                rec_val  = best_cuts[p]
                rec_note = "no clearly stable value found — widen the grid"

        print(f"    → Recommended: {p} = {rec_val:.4f}  ({rec_note})")
        recommendations[p] = rec_val

    # ── compact recommendation block ──────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  RECOMMENDED CUT VALUES")
    print(f"{'─' * W}")
    for p in param_cols:
        flag = "--" + p.replace("_", "_")
        print(f"    {flag:<28s} {recommendations.get(p, best_cuts[p]):.4f}")
    print("=" * W)
    return recommendations


def write_best_config(rec_cuts, meta_row, fixed_cuts, out_file):
    config = {p: float(rec_cuts[p]) for p in rec_cuts}
    config.update({k: v for k, v in fixed_cuts.items() if v is not None})
    config["_slope_mle"]   = float(meta_row["slope"])
    config["_slope_lo"]    = _SLOPE_LO
    config["_slope_hi"]    = _SLOPE_HI
    config["_sigma_slope"] = float(meta_row["sigma_slope"])
    config["_N_sel"]       = int(meta_row["N"])
    config["_loglike"]     = float(meta_row["loglike"])
    config["_volatility"]  = float(meta_row["volatility"])
    with open(out_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Best config written to: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: CLI
# ─────────────────────────────────────────────────────────────────────────────

# Grid parameter names — used by `recommend` to identify grid columns in CSV.
_GRID_PARAM_NAMES = [
    "haty_max", "haty_min", "slope_plane", "intercept_plane", "intercept_plane2"
]


def _add_grid_args(parser):
    """Add --<param>_range / --<param>_n arguments to a (sub)parser."""
    def _grid_arg(name, lo, hi, n):
        parser.add_argument(f"--{name}_range", type=float, nargs=2,
                            default=[lo, hi], metavar=("LO", "HI"),
                            help=f"{name} grid range (default: [{lo}, {hi}])")
        parser.add_argument(f"--{name}_n", type=int, default=n,
                            help=f"Number of {name} grid points (default: {n})")

    _grid_arg("haty_max",         -20.0, -19.0, 5)
    _grid_arg("haty_min",         -22.2, -21.3, 5)
    _grid_arg("slope_plane",       -7.5,  -5.5, 5)
    _grid_arg("intercept_plane",  -21.0, -19.8, 5)
    _grid_arg("intercept_plane2", -19.2, -18.0, 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep selection cuts and find the stability plateau.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ── sweep subcommand ───────────────────────────────────────────────────────
    sp_sweep = subparsers.add_parser(
        "sweep",
        help="Run MLE at every grid point and save cut_sweep.csv (no plots).")
    sp_sweep.add_argument("--source", choices=["fullmocks", "DESI", "ariel"],
                          default="fullmocks")
    sp_sweep.add_argument("--fits_file", default=None,
                          help="Path to a single FITS file (fullmocks or DESI)")
    sp_sweep.add_argument("--dir",
                          default="/Users/akim/Projects/TFPV/ariel/data",
                          help="Directory of FITS files (fullmocks, picks first match)")
    sp_sweep.add_argument("--run", required=True,
                          help="Run name; outputs go to output/<run>/")
    _add_grid_args(sp_sweep)
    sp_sweep.add_argument("--z_obs_min", type=float, default=0.03)
    sp_sweep.add_argument("--z_obs_max", type=float, default=0.10)
    sp_sweep.add_argument("--n_workers", type=int, default=None,
                          help="Number of parallel workers (default: all CPUs)")
    sp_sweep.add_argument("--debug", action="store_true",
                          help="Use a 3-point grid (~243 evaluations, ~50× faster)")
    sp_sweep.add_argument("--n_sweep_objects", type=int, default=10000,
                          help="Maximum number of objects passed to MLE fits; "
                               "raw data is randomly subsampled to this size when "
                               "larger (default: 10000; 0 = use all)")

    # ── recommend subcommand ───────────────────────────────────────────────────
    sp_rec = subparsers.add_parser(
        "recommend",
        help="Load cut_sweep.csv and generate plots/recommendations.")
    sp_rec.add_argument("--run", required=True,
                        help="Run name; reads output/<run>/cut_sweep.csv")
    sp_rec.add_argument("--true_slope", type=float, default=None,
                        help="True TFR slope (fullmocks) for bias reporting")
    sp_rec.add_argument("--vol_threshold_factor", type=float, default=25.0,
                        help="Plateau threshold: Pth percentile of the interior-slope "
                             "volatility distribution (0–100, default 25).  "
                             "Grid points with volatility <= this percentile value "
                             "form the 'stable plateau'; the cut combination with the "
                             "most galaxies is selected from within it.  "
                             "Increase P to widen the plateau; decrease to restrict it.")
    sp_rec.add_argument("--write_best", action="store_true",
                        help="Write cut_sweep_best_config.json at the best grid point")

    args = parser.parse_args()

    run_dir  = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "cut_sweep.csv")

    # ── sweep ──────────────────────────────────────────────────────────────────
    if args.subcommand == "sweep":
        def _lin(rng, n):
            return np.linspace(rng[0], rng[1], n)

        grid_params = {
            "haty_max":         _lin(args.haty_max_range,         args.haty_max_n),
            "haty_min":         _lin(args.haty_min_range,         args.haty_min_n),
            "slope_plane":      _lin(args.slope_plane_range,      args.slope_plane_n),
            "intercept_plane":  _lin(args.intercept_plane_range,  args.intercept_plane_n),
            "intercept_plane2": _lin(args.intercept_plane2_range, args.intercept_plane2_n),
        }
        if args.debug:
            grid_params = {k: np.linspace(v[0], v[-1], 3)
                           for k, v in grid_params.items()}
            print(f"DEBUG: grid reduced to 3 points per parameter "
                  f"({3**len(grid_params)} evaluations)")
        fixed_cuts = {
            "z_obs_min": args.z_obs_min,
            "z_obs_max": args.z_obs_max,
        }

        # Load raw data
        if args.source == "fullmocks":
            fits_file = args.fits_file
            if fits_file is None:
                pattern = os.path.join(args.dir, "TF_extended_AbacusSummit_*.fits")
                matches = sorted(glob.glob(pattern))
                if not matches:
                    raise FileNotFoundError(f"No FITS files found: {pattern}")
                fits_file = matches[0]
                print(f"Auto-selected: {fits_file}")
            raw_data = load_fullmocks(fits_file)
        elif args.source == "DESI":
            fits_file = args.fits_file or "data/DESI-DR1_TF_pv_cat_v15.fits"
            raw_data = load_desi(fits_file)
        else:
            raise NotImplementedError("--source ariel: provide --fits_file for raw data")

        n_cap = args.n_sweep_objects
        if n_cap and len(raw_data["x"]) > n_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(raw_data["x"]), size=n_cap, replace=False)
            raw_data = {k: (v[idx] if isinstance(v, np.ndarray) else v)
                        for k, v in raw_data.items()}
            print(f"Subsampled raw data to {n_cap} objects")

        df, param_cols = run_sweep(raw_data, grid_params, fixed_cuts,
                                   n_workers=args.n_workers)
        df["volatility"] = compute_volatility(df, grid_params, param_cols)
        df = compute_scores(df)

        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    # ── recommend ──────────────────────────────────────────────────────────────
    elif args.subcommand == "recommend":
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found — run 'sweep' first.")

        print(f"Loading sweep results from {csv_path}")
        df = pd.read_csv(csv_path)

        # Infer param_cols and grid_params from CSV columns
        param_cols  = [p for p in _GRID_PARAM_NAMES if p in df.columns]
        grid_params = {p: sorted(df[p].dropna().unique()) for p in param_cols}

        # Reconstruct fixed_cuts from the first data row
        fixed_cuts = {}
        for key in ("z_obs_min", "z_obs_max"):
            if key in df.columns:
                val = df[key].dropna().iloc[0] if not df[key].dropna().empty else None
                if val is not None:
                    fixed_cuts[key] = float(val)

        best_row = find_best_stable_max_N(df, args.vol_threshold_factor)
        if best_row is None:
            print("WARNING: no valid grid points found — check cut ranges and data.")
        else:
            best_cuts = {p: float(best_row[p]) for p in param_cols}
            recs = sweetspot_summary(df, param_cols, best_cuts, args.true_slope)
            rec_cuts = {p: recs.get(p, best_cuts[p]) for p in param_cols}

            if args.write_best:
                cfg_path = os.path.join(run_dir, "cut_sweep_best_config.json")
                write_best_config(rec_cuts, best_row, fixed_cuts, cfg_path)

            make_plots(df, param_cols, best_cuts, run_dir, args.true_slope, rec_cuts=rec_cuts)
