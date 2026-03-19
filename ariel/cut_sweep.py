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
  # fullmocks (reads raw FITS):
  python cut_sweep.py --source fullmocks \\
      --fits_file data/TF_extended_AbacusSummit_base_c000_ph000_r001_z0.11.fits \\
      --run c000_ph000_r001

  # DESI (reads raw FITS):
  python cut_sweep.py --source DESI --run DESI

  # Restrict the grid (override range and resolution for any parameter):
  python cut_sweep.py --source fullmocks --fits_file ... --run test \\
      --haty_max_range -21.5 -19.0 --haty_max_n 4 \\
      --slope_plane_range -8.0 -5.0 --slope_plane_n 4

  # Use a pre-saved sweep CSV to regenerate plots only:
  python cut_sweep.py --run c000_ph000_r001 --plots_only

  # After identifying the best point, write config.json:
  python cut_sweep.py --source fullmocks --fits_file ... --run c000_ph000_r001 \\
      --write_best

Output (all in output/<run>/):
  cut_sweep.csv                   — full grid results
  cut_sweep_1d.png                — 1-D slope profiles (one panel per parameter)
  cut_sweep_2d_<p1>_<p2>.png     — 2-D slices (slope / volatility / loglike)
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


def tophat_loglik(x_std, sigma_x_std, y, sigma_y,
                  slope_std, intercept_std, sigma_int_x, sigma_int_y,
                  y_min, y_max, haty_min, haty_max):
    """Total tophat model log-likelihood, vectorised over N galaxies.

    Implements the Stan tophat.stan likelihood in Python/numpy:

      (1) y ~ N(intercept_std + slope_std * x_std,  sqrt(sigmasq_tot))
          [marginal likelihood with y_TF integrated out over flat prior]
      (2) + log|slope_std| * N                       [Jacobian]
      (3) + log[Φ((y_max−μ*)/σ*) − Φ((y_min−μ*)/σ*)]  [y_TF limits]
      (4) − log P_sel_i                              [selection correction]

    where P_sel ≈ Φ((haty_max−yfromx)/√sigmasq_tot) − Φ((haty_min−yfromx)/√sigmasq_tot)
    (magnitude-limits-only approximation; sufficient for locating the plateau).
    """
    sigmasq1   = sigma_int_x**2 + sigma_x_std**2    # variance in x_std
    sigmasq2   = sigma_int_y**2 + sigma_y**2         # variance in y
    s2         = slope_std**2
    sigmasq_tot = s2 * sigmasq1 + sigmasq2
    sqrt_tot   = np.sqrt(sigmasq_tot)

    yfromx = intercept_std + slope_std * x_std

    # (1) marginal likelihood in y
    ll = norm.logpdf(y, yfromx, sqrt_tot)

    # (2) Jacobian (scalar, broadcast)
    ll = ll + np.log(np.abs(slope_std))

    # (3) y_TF limits correction
    mu_star        = (yfromx * sigmasq2 + y * s2 * sigmasq1) / sigmasq_tot
    sigmasq_star   = s2 * sigmasq1 * sigmasq2 / sigmasq_tot
    sqrt_star      = np.sqrt(np.maximum(sigmasq_star, 1e-16))
    log_Phi_max    = norm.logcdf((y_max - mu_star) / sqrt_star)
    log_Phi_min    = norm.logcdf((y_min - mu_star) / sqrt_star)
    ll            += _log_diff_exp(log_Phi_max, log_Phi_min)

    # (4) Selection correction: magnitude-limits-only approximation
    log_Psel_max   = norm.logcdf((haty_max - yfromx) / sqrt_tot)
    log_Psel_min   = norm.logcdf((haty_min - yfromx) / sqrt_tot)
    log_Psel       = _log_diff_exp(log_Psel_max, log_Psel_min)
    ll            -= log_Psel

    return float(np.sum(ll))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MLE FITTING
# ─────────────────────────────────────────────────────────────────────────────

_N_MIN = 30   # minimum galaxies to attempt a fit


def fit_mle(x, sigma_x, y, sigma_y, haty_min, haty_max):
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
                                 y_min, y_max, float(haty_min), float(haty_max))
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
    result = fit_mle(x, sx, y, sy, cuts["haty_min"], cuts["haty_max"])
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


def plot_1d(df, param_cols, best_cuts, true_slope, out_file):
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
        ax.axvline(best_cuts[p], color="orange", linestyle=":", linewidth=1.5,
                   label="best")
        ax.set_xlabel(_label(p), fontsize=11)
        ax.set_ylabel("slope", fontsize=11)
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


def make_plots(df, param_cols, best_cuts, run_dir, true_slope=None):
    """Generate all 1-D and 2-D plots."""
    out_1d = os.path.join(run_dir, "cut_sweep_1d.png")
    plot_1d(df, param_cols, best_cuts, true_slope, out_1d)

    for p1, p2 in itertools.combinations(param_cols, 2):
        safe1 = p1.replace("_", "")
        safe2 = p2.replace("_", "")
        out_2d = os.path.join(run_dir, f"cut_sweep_2d_{safe1}_{safe2}.png")
        plot_2d(df, p1, p2, param_cols, best_cuts, out_2d)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: BEST-CUTS SELECTION AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def find_best(df, criterion="score"):
    """Return the row with the best (maximum) value of `criterion`."""
    sub = df.dropna(subset=[criterion])
    if sub.empty:
        return None
    return sub.loc[sub[criterion].idxmax()]


def find_best_stable_max_N(df, vol_threshold_factor=3.0):
    """Return the plateau point with the most galaxies.

    'Plateau' = volatility <= vol_min * vol_threshold_factor.
    Falls back to the score-best point if no plateau is found.
    """
    sub = df.dropna(subset=["volatility", "N", "score"])
    if sub.empty:
        return None
    vol_min = float(sub["volatility"].min())
    vol_threshold = max(vol_min * vol_threshold_factor, vol_min + 1e-6)
    stable = sub[sub["volatility"] <= vol_threshold]
    if stable.empty:
        return sub.loc[sub["score"].idxmax()]   # fallback
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
    STABLE_SIGMA = 2.0   # a point is 'stable' if |slope - ref| < STABLE_SIGMA * σ
    SENS_SIGMA   = 3.0   # parameter is 'sensitive' if range > SENS_SIGMA * median(σ)

    # Reference slope at the best-score cuts
    ref_mask = pd.Series(True, index=df.index)
    for p in param_cols:
        ref_mask &= df[p] == best_cuts[p]
    ref_rows = df[ref_mask].dropna(subset=["slope", "sigma_slope"])
    if ref_rows.empty:
        print("WARNING: no data at best_cuts for reference slope.")
        return
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

        slope_range   = float(slopes.max() - slopes.min())
        median_sigma  = float(np.median(sigmas))
        sensitive     = slope_range > SENS_SIGMA * median_sigma

        # Stable: slope within STABLE_SIGMA * σ of the reference
        combined_sigma = np.maximum(sigmas, ref_sigma)
        stable = np.abs(slopes - ref_slope) < STABLE_SIGMA * combined_sigma

        print(f"\n  {_label(p)}  [{p}]")
        print(f"    Tested range: [{vals.min():.2f}, {vals.max():.2f}]  "
              f"({len(vals)} points)")

        if not sensitive:
            print(f"    INSENSITIVE — slope varies by only {slope_range:.4f} "
                  f"({slope_range/median_sigma:.1f}σ) across the full range.")
            print(f"    All tested values are in the stable region.")
            # prefer the loosest cut (most galaxies) even when insensitive
            best_idx = int(np.argmax(ns))
            rec_val  = float(vals[best_idx])
            rec_note = (f"loosest value — maximises N_sel = {int(ns[best_idx])}")
        else:
            stable_vals = vals[stable]
            stable_ns   = ns[stable]
            unstable    = vals[~stable]
            print(f"    SENSITIVE — slope varies by {slope_range:.4f} "
                  f"({slope_range/median_sigma:.1f}σ).")
            print(f"    Unstable values (slope deviates > {STABLE_SIGMA}σ): "
                  f"{[f'{v:.2f}' for v in unstable]}")
            if len(stable_vals) > 0:
                print(f"    Stable region: [{stable_vals.min():.2f}, "
                      f"{stable_vals.max():.2f}]  "
                      f"(slope ≈ {slopes[stable].mean():.4f})")
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


def write_best_config(best_row, param_cols, fixed_cuts, out_file):
    config = {p: float(best_row[p]) for p in param_cols}
    config.update({k: v for k, v in fixed_cuts.items() if v is not None})
    config["_slope_mle"]  = float(best_row["slope"])
    config["_sigma_slope"] = float(best_row["sigma_slope"])
    config["_N_sel"]      = int(best_row["N"])
    config["_loglike"]    = float(best_row["loglike"])
    config["_volatility"] = float(best_row["volatility"])
    with open(out_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Best config written to: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: CLI
# ─────────────────────────────────────────────────────────────────────────────

def _grid_arg(parser, name, lo, hi, n, label):
    parser.add_argument(f"--{name}_range", type=float, nargs=2,
                        default=[lo, hi], metavar=("LO", "HI"),
                        help=f"{label} grid range (default: [{lo}, {hi}])")
    parser.add_argument(f"--{name}_n", type=int, default=n,
                        help=f"Number of {label} grid points (default: {n})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep selection cuts and find the stability plateau.")

    # --- source / input ---
    parser.add_argument("--source", choices=["fullmocks", "DESI", "ariel"],
                        default="fullmocks")
    parser.add_argument("--fits_file", default=None,
                        help="Path to a single FITS file (fullmocks or DESI)")
    parser.add_argument("--dir",
                        default="/Users/akim/Projects/TFPV/ariel/data",
                        help="Directory of FITS files (fullmocks, picks first match)")
    parser.add_argument("--run", default=None,
                        help="Run name; outputs go to output/<run>/")

    # --- grid parameters ---
    # parser.add_argument("--haty_max",  type=float, default=-20.0)
    # parser.add_argument("--haty_min",  type=float, default=-21.8)
    # parser.add_argument("--slope_plane",      type=float, default=-6.5)
    # parser.add_argument("--intercept_plane",  type=float, default=-20.)
    # parser.add_argument("--intercept_plane2", type=float, default=-19.)
    _grid_arg(parser, "haty_max",          -20.0, -19.0, 5, "haty_max")
    _grid_arg(parser, "haty_min",          -22.2, -21.3, 5, "haty_min")
    _grid_arg(parser, "slope_plane",       -7.5,  -5.5, 5, "slope_plane")
    _grid_arg(parser, "intercept_plane",   -21, -19.8, 5, "intercept_plane")
    _grid_arg(parser, "intercept_plane2",  -19.2, -18.0, 5, "intercept_plane2")

    # --- fixed cuts (redshift window) ---
    parser.add_argument("--z_obs_min", type=float, default=0.03)
    parser.add_argument("--z_obs_max", type=float, default=0.10)

    # --- other options ---
    parser.add_argument("--true_slope", type=float, default=None,
                        help="True TFR slope (fullmocks only) for bias reporting")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPUs)")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip the sweep; reload cut_sweep.csv and regenerate plots")
    parser.add_argument("--write_best", action="store_true",
                        help="Write cut_sweep_best_config.json at the best grid point")
    parser.add_argument("--vol_threshold_factor", type=float, default=3.0,
                        help="Plateau threshold: volatility <= vol_min * factor "
                             "(default 3.0)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: subsample raw data to 1/4 and use 3-point grid")

    args = parser.parse_args()

    # ── run directory ──────────────────────────────────────────────────────────
    run_dir = os.path.join("output", args.run) if args.run else "output/cut_sweep"
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "cut_sweep.csv")

    # ── grid definition ────────────────────────────────────────────────────────
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
        grid_params = {k: np.linspace(v[0], v[-1], 3) for k, v in grid_params.items()}
        print(f"DEBUG: grid reduced to 3 points per parameter "
              f"({3**len(grid_params)} evaluations)")
    fixed_cuts = {
        "z_obs_min": args.z_obs_min,
        "z_obs_max": args.z_obs_max,
    }

    # ── sweep or reload ────────────────────────────────────────────────────────
    if args.plots_only and os.path.exists(csv_path):
        print(f"Loading existing sweep results from {csv_path}")
        df = pd.read_csv(csv_path)
        param_cols = [p for p in grid_params if p in df.columns]
        # Reconstruct grid_params from CSV (values actually present)
        grid_params = {p: sorted(df[p].dropna().unique()) for p in param_cols}
    else:
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

        if args.debug:
            rng = np.random.default_rng(0)
            n_debug = max(100, len(raw_data["x"]) // 4)
            idx = rng.choice(len(raw_data["x"]), size=n_debug, replace=False)
            raw_data = {k: (v[idx] if isinstance(v, np.ndarray) else v)
                        for k, v in raw_data.items()}
            print(f"DEBUG: subsampled raw data to {n_debug} objects")

        df, param_cols = run_sweep(raw_data, grid_params, fixed_cuts,
                                   n_workers=args.n_workers)
        df["volatility"] = compute_volatility(df, grid_params, param_cols)
        df = compute_scores(df)

        if args.true_slope is not None:
            df["bias"] = (df["slope"] - args.true_slope) / df["sigma_slope"]

        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    # ── best cuts ──────────────────────────────────────────────────────────────
    best_row = find_best_stable_max_N(df, args.vol_threshold_factor)
    if best_row is None:
        print("WARNING: no valid grid points found — check cut ranges and data.")
    else:
        best_cuts = {p: float(best_row[p]) for p in param_cols}
        report_best(best_row, param_cols, args.true_slope)

        if args.write_best:
            cfg_path = os.path.join(run_dir, "cut_sweep_best_config.json")
            write_best_config(best_row, param_cols, fixed_cuts, cfg_path)

        # ── plots ──────────────────────────────────────────────────────────────
        make_plots(df, param_cols, best_cuts, run_dir, args.true_slope)

        # ── sweet spot summary ─────────────────────────────────────────────
        sweetspot_summary(df, param_cols, best_cuts, args.true_slope)
