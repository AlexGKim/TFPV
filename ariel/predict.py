import argparse
import os
import numpy as np
import glob
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt 
from astropy.io import fits
import json
from scipy.special import erf
import matplotlib.colors as mcolors
from scipy.stats import norm


def create_average_grid_image(x_coords, y_coords, values, grid_resolution_x, grid_resolution_y, v_lim=None, clip_percentile=(1, 99)):
    x = np.array(x_coords)
    y = np.array(y_coords)
    z = np.array(values)

    if clip_percentile is not None:
        lo, hi = np.nanpercentile(z, clip_percentile)
        keep = (z >= lo) & (z <= hi)
        x, y, z = x[keep], y[keep], z[keep]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # 1. Compute Grid
    sum_heatmap, xedges, yedges = np.histogram2d(x, y, bins=[grid_resolution_x, grid_resolution_y], weights=z)
    count_heatmap, _, _ = np.histogram2d(x, y, bins=[grid_resolution_x, grid_resolution_y])

    # Calculate average
    average_heatmap = np.divide(sum_heatmap, count_heatmap, where=count_heatmap!=0, out=np.nan * np.ones_like(sum_heatmap))
    
    # 2. Handle Color Limits
    if v_lim is None:
        # Fallback to 95th percentile if no manual limit is provided
        v_lim = np.nanpercentile(np.abs(average_heatmap), 95)
    
    # Use CenteredNorm to make 0 the neutral color
    norm = mcolors.CenteredNorm(vcenter=0.0, halfrange=v_lim)

    cmap = plt.colormaps['seismic'].copy()
    cmap.set_bad(color='lightgrey')

    # 3. Create Figure
    fig, ax = plt.subplots(figsize=(8, 6)) # Wide aspect ratio

    img = ax.imshow(
        average_heatmap.T,
        extent=[xmin, xmax, ymin, ymax],
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        aspect='auto',
        norm=norm
    )
    
    ax.yaxis.set_inverted(True) # Your requested Y flip
    
    return fig, ax, img

# def create_average_grid_image(x_coords, y_coords, values, grid_resolution_x, grid_resolution_y):
#     # Ensure inputs are numpy arrays
#     x = np.array(x_coords)
#     y = np.array(y_coords)
#     z = np.array(values)

#     # Define the grid boundaries
#     xmin, xmax = x.min(), x.max()
#     ymin, ymax = y.min(), y.max()

#     # Calculate the sum of values and the count of points in each bin
#     # 'weights=z' makes the function sum the 'z' values instead of counting points
#     sum_heatmap, xedges, yedges = np.histogram2d(x, y, bins=[grid_resolution_x, grid_resolution_y], weights=z)
#     # Count the number of points in each bin
#     count_heatmap, _, _ = np.histogram2d(x, y, bins=[grid_resolution_x, grid_resolution_y])

#     # Calculate the average: sum / count. Handle empty bins (count=0) with np.nan or 0.
#     # We use np.divide and a mask to avoid ZeroDivisionError
#     average_heatmap = np.divide(sum_heatmap, count_heatmap, where=count_heatmap!=0, out=np.zeros_like(sum_heatmap))
    
#     # Plot the result using imshow
#     fig = plt.figure(figsize=(8, 6))
#     # Note: imshow handles origin differently than pcolormesh; 'origin="lower"' matches typical Cartesian coordinates
#     plt.imshow(average_heatmap.T, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='seismic', interpolation='nearest', aspect='auto')
#     plt.colorbar(label='Average Value')
#     plt.xlabel(r'$\log{V/V_0}$')
#     plt.ylabel(r'$M$')
#     plt.title(r'$M_\text{predicted} - M$')
#     plt.gca().yaxis.set_inverted(True)
#     plt.show()

def draw_ystar_posterior_predictive_normal(
    N,
    xhat_star,
    sigma_x_star,
    s,
    c,
    sigma_int_x,
    mu_y_TF,
    tau,
    sigma_int_y,
    rng=None,
):
    """
    Draw N samples from the (single-theta) posterior predictive for the *latent*
    magnitude y_* (no y-measurement error), in the Normal y_TF prior case:

        y_TF ~ Normal(mu_y_TF, tau^2)
        x    | y_TF ~ Normal((y_TF - c)/s, sigma_int_x^2)
        xhat | x    ~ Normal(x,           sigma_x_star^2)
        y_*  | y_TF ~ Normal(y_TF,        sigma_int_y^2)

    Returns draws from:
        p(y_* | xhat_*, sigma_x_star, theta)

    Notes
    -----
    tau is an SD (variance = tau**2). Requires s != 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    if s == 0:
        raise ValueError("s must be nonzero.")
    for name, val in [
        ("sigma_x_star", sigma_x_star),
        ("sigma_int_x", sigma_int_x),
        ("tau", tau),
        ("sigma_int_y", sigma_int_y),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0.")

    # Likelihood implied for y_TF from xhat:
    # xhat | y_TF ~ Normal((y_TF - c)/s, sigma_x_tot^2)
    sigma_x_tot2 = sigma_x_star**2 + sigma_int_x**2
    mu_L = c + s * xhat_star
    V_L = (s**2) * sigma_x_tot2          # variance of implied y_TF from xhat

    # Prior on y_TF
    mu0 = mu_y_TF
    V0 = tau**2

    # Posterior for y_TF | xhat (Gaussian conjugacy, with degeneracy handling)
    if V0 == 0.0 and V_L == 0.0:
        if not np.isclose(mu0, mu_L):
            raise ValueError(
                "Degenerate case inconsistent: tau=0 and s^2*(sigma_x_star^2+sigma_int_x^2)=0 "
                "but mu_y_TF != c + s*xhat_star."
            )
        mu_post, V_post = mu0, 0.0
    elif V0 == 0.0:
        mu_post, V_post = mu0, 0.0
    elif V_L == 0.0:
        mu_post, V_post = mu_L, 0.0
    else:
        V_post = 1.0 / (1.0 / V0 + 1.0 / V_L)
        mu_post = V_post * (mu0 / V0 + mu_L / V_L)

    # Marginalize y_TF to get y_* | xhat:
    # y_* = y_TF + Normal(0, sigma_int_y^2)
    V_ystar = V_post + sigma_int_y**2

    return rng.normal(loc=mu_post, scale=np.sqrt(V_ystar), size=N)

def load_xy_and_uncertainties_from_stan_json(
    json_path,
    row=None,
    sort_by_zobs=False,
    *,
    x_key="x",
    sigma_x_key="sigma_x",
    y_key="y",              # per your note: y_star is y
    sigma_y_key="sigma_y",
    z_key="z_obs",
    N_key="N_total",
    apply_valid_mask=True,
):
    """
    JSON analogue of load_xy_and_uncertainties_from_desi, for a Stan-style
    data dictionary saved to JSON, e.g.

        stan_data = {
            'N_total': N_total,
            'x': x_data,
            'sigma_x': sigma_x_data,
            'y': y_data,
            'sigma_y': sigma_y_data,
            'z_obs': z_obs,
            ...
        }

    Returns xhat, sigma_x, yhat, sigma_y, zobs from the JSON payload.

    Parameters
    ----------
    json_path : str or path-like
        Path to the JSON file containing the stan_data dict.
    row : int or None
        If None, return arrays for all rows (after optional filtering/sorting).
        If an int, return a single 5-tuple (floats) for that row (0-based index
        in the filtered/sorted arrays).
    sort_by_zobs : bool
        If True, sort all returned arrays by increasing zobs.
    x_key, sigma_x_key, y_key, sigma_y_key, z_key : str
        Keys to use in the JSON dictionary.
    N_key : str
        Optional key giving the expected length; if present, lengths are checked.
    apply_valid_mask : bool
        If True, apply a validity mask:
            finite x, sigma_x, y, sigma_y, zobs;
            sigma_x > 0;
            sigma_y >= 0.

    Returns
    -------
    If row is None:
        xhat, sigma_x, yhat, sigma_y, zobs : np.ndarray
    Else:
        xhat, sigma_x, yhat, sigma_y, zobs : float
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        stan_data = json.load(f)

    # --- required keys
    required = [x_key, sigma_x_key, y_key, sigma_y_key, z_key]
    missing = [k for k in required if k not in stan_data]
    if missing:
        raise ValueError(
            f"Missing required key(s) {missing} in {json_path}. "
            f"Available keys: {sorted(list(stan_data.keys()))}"
        )

    # --- load as arrays
    xhat = np.asarray(stan_data[x_key], dtype=float)
    sigma_x = np.asarray(stan_data[sigma_x_key], dtype=float)
    yhat = np.asarray(stan_data[y_key], dtype=float)
    sigma_y = np.asarray(stan_data[sigma_y_key], dtype=float)
    zobs = np.asarray(stan_data[z_key], dtype=float)

    # --- shape checks
    for name, arr in [("xhat", xhat), ("sigma_x", sigma_x), ("yhat", yhat), ("sigma_y", sigma_y), ("zobs", zobs)]:
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array for '{name}', got shape {arr.shape}.")

    n = xhat.size
    if not (sigma_x.size == yhat.size == sigma_y.size == zobs.size == n):
        raise ValueError(
            "Length mismatch among arrays: "
            f"xhat={xhat.size}, sigma_x={sigma_x.size}, yhat={yhat.size}, "
            f"sigma_y={sigma_y.size}, zobs={zobs.size}."
        )

    # Optional consistency check with N_total
    if N_key in stan_data:
        N_total = int(stan_data[N_key])
        if N_total != n:
            raise ValueError(f"{N_key}={N_total} but loaded arrays have length {n}.")

    # --- validity mask
    if apply_valid_mask:
        mask = (
            np.isfinite(xhat) & np.isfinite(sigma_x) &
            np.isfinite(yhat) & np.isfinite(sigma_y) &
            np.isfinite(zobs) &
            (sigma_x > 0) &
            (sigma_y >= 0)
        )
        xhat = xhat[mask]
        sigma_x = sigma_x[mask]
        yhat = yhat[mask]
        sigma_y = sigma_y[mask]
        zobs = zobs[mask]
    else:
        for name, arr in [("xhat", xhat), ("sigma_x", sigma_x), ("yhat", yhat), ("sigma_y", sigma_y), ("zobs", zobs)]:
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"Non-finite values found in '{name}'.")
        if np.any(sigma_x <= 0):
            raise ValueError("Non-positive sigma_x encountered.")
        if np.any(sigma_y < 0):
            raise ValueError("Negative sigma_y encountered.")

    # --- optional sort
    if row is None and sort_by_zobs:
        idx = np.argsort(zobs)
        xhat = xhat[idx]
        sigma_x = sigma_x[idx]
        yhat = yhat[idx]
        sigma_y = sigma_y[idx]
        zobs = zobs[idx]

    # --- row selection
    if row is not None:
        return (float(xhat[row]), float(sigma_x[row]),
                float(yhat[row]), float(sigma_y[row]),
                float(zobs[row]))

    return xhat, sigma_x, yhat, sigma_y, zobs

def load_xy_and_uncertainties_from_desi(
    fits_path,
    row=None,
    sort_by_zobs=False,
    *,
    V0=100.0,
    vel_col="V_0p4R26",
    vel_err_col="V_0p4R26_ERR",
    mag_col="R_ABSMAG_SB26",
    mag_err_col="R_ABSMAG_SB26_ERR",
    z_col="Z_DESI",
    z_col_candidates=("zobs", "ZOBS", "Z", "ZHELIO", "Z_CMB", "ZDESI", "ZTRUE"),
    apply_valid_mask=True,
):
    """
    DESI FITS analogue of load_xy_and_uncertainties_from_csv.

    Reads a FITS table (assumes data in HDU 1) and returns:
        xhat      = log10(V / V0)
        sigma_x   = V_err / (V * ln(10))
        yhat      = absolute magnitude column
        sigma_y   = magnitude uncertainty column
        zobs      = redshift column

    Parameters
    ----------
    fits_path : str or path-like
        Path to the FITS file.
    row : int or None
        If None, return arrays for all rows (after optional filtering/sorting).
        If an int, return a single 5-tuple (floats) for that row (0-based index
        in the filtered/sorted arrays).
    sort_by_zobs : bool
        If True, sort all returned arrays by increasing zobs.
    V0 : float
        Reference velocity for xhat = log10(V/V0).
    vel_col, vel_err_col, mag_col, mag_err_col : str
        Column names in the FITS table.
    z_col : str
        Preferred redshift column name. If not present, will try z_col_candidates.
    z_col_candidates : tuple[str,...]
        Fallback names for redshift if z_col is missing.
    apply_valid_mask : bool
        If True, apply a validity mask similar to your processing code:
        finite values and positive V and V_err, plus non-negative mag_err.

    Returns
    -------
    If row is None:
        xhat, sigma_x, yhat, sigma_y, zobs : np.ndarray
    Else:
        xhat, sigma_x, yhat, sigma_y, zobs : float
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        names = set(data.dtype.names or ())

        # Resolve z column name
        if z_col not in names:
            found = None
            for cand in z_col_candidates:
                if cand in names:
                    found = cand
                    break
            if found is None:
                raise ValueError(
                    f"Could not find redshift column. Tried z_col={z_col!r} and candidates "
                    f"{z_col_candidates}. Available columns include: {sorted(list(names))[:30]} ..."
                )
            z_col_use = found
        else:
            z_col_use = z_col

        # Resolve mag_err_col with fallback
        if mag_err_col not in names:
            fallback = "R_MAG_SB26_ERR"
            if fallback in names:
                print(f"Warning: {mag_err_col!r} absent; falling back to {fallback!r}")
                mag_err_col = fallback
            else:
                raise ValueError(
                    f"Missing magnitude error column {mag_err_col!r} and fallback "
                    f"{fallback!r}. Available: {sorted(list(names))[:30]} ..."
                )

        # Required columns
        required = [vel_col, vel_err_col, mag_col, mag_err_col, z_col_use]
        missing = [c for c in required if c not in names]
        if missing:
            raise ValueError(f"Missing required column(s) {missing}. Available: {sorted(list(names))[:30]} ...")

        V = np.asarray(data[vel_col], dtype=float)
        V_err = np.asarray(data[vel_err_col], dtype=float)
        yhat = np.asarray(data[mag_col], dtype=float)
        sigma_y = np.asarray(data[mag_err_col], dtype=float)
        zobs = np.asarray(data[z_col_use], dtype=float)

    # Convert to xhat and sigma_x
    if V0 <= 0:
        raise ValueError("V0 must be > 0.")
    xhat = np.log10(V / V0)
    sigma_x = V_err / (V * np.log(10))

    # Optional validity mask (mirrors your process_desi_tf_data logic)
    if apply_valid_mask:
        mask = (
            np.isfinite(V) & np.isfinite(V_err) &
            np.isfinite(yhat) & np.isfinite(sigma_y) &
            np.isfinite(zobs) &
            (V > 0) & (V_err > 0) &
            (sigma_y >= 0)
        )
        xhat = xhat[mask]
        sigma_x = sigma_x[mask]
        yhat = yhat[mask]
        sigma_y = sigma_y[mask]
        zobs = zobs[mask]
    else:
        # still basic checks
        for name, arr in [("xhat", xhat), ("sigma_x", sigma_x), ("yhat", yhat), ("sigma_y", sigma_y), ("zobs", zobs)]:
            if np.any(~np.isfinite(arr)):
                raise ValueError(f"Non-finite values found in derived/loaded array '{name}'.")
        if np.any(sigma_x < 0):
            raise ValueError("Negative sigma_x encountered (check V and V_err).")
        if np.any(sigma_y < 0):
            raise ValueError("Negative sigma_y encountered (magnitude uncertainty).")

    # Optional sort
    if row is None and sort_by_zobs:
        idx = np.argsort(zobs)
        xhat = xhat[idx]
        sigma_x = sigma_x[idx]
        yhat = yhat[idx]
        sigma_y = sigma_y[idx]
        zobs = zobs[idx]

    # Row selection
    if row is not None:
        return (float(xhat[row]), float(sigma_x[row]),
                float(yhat[row]), float(sigma_y[row]),
                float(zobs[row]))

    return xhat, sigma_x, yhat, sigma_y, zobs

def load_xy_and_uncertainties_from_csv(csv_path, row=None, sort_by_zobs=False):
    """
    Read a CSV with columns:
      log_V_V0, log_V_V0_unc, M_abs, M_abs_unc, zobs

    Returns (xhat, sigma_x, yhat, sigma_y, zobs).
    Optionally sorts all returned arrays by increasing zobs.

    If row is not None, returns a single 5-tuple (floats) and sorting is not applied.
    """
    df = pd.read_csv(csv_path)

    required = ["log_V_V0", "log_V_V0_unc", "M_abs", "M_abs_unc", "zobs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing}. Found columns: {list(df.columns)}")

    xhat = df["log_V_V0"].to_numpy(dtype=float)
    sigx = df["log_V_V0_unc"].to_numpy(dtype=float)
    yhat = df["M_abs"].to_numpy(dtype=float)
    sigy = df["M_abs_unc"].to_numpy(dtype=float)
    zobs = df["zobs"].to_numpy(dtype=float)

    for name, arr in [("log_V_V0", xhat), ("log_V_V0_unc", sigx),
                      ("M_abs", yhat), ("M_abs_unc", sigy), ("zobs", zobs)]:
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"Non-finite values found in column '{name}'.")
    if np.any(sigx < 0):
        raise ValueError("Negative uncertainties found in column 'log_V_V0_unc'.")
    if np.any(sigy < 0):
        raise ValueError("Negative uncertainties found in column 'M_abs_unc'.")

    if row is not None:
        return (float(xhat[row]), float(sigx[row]),
                float(yhat[row]), float(sigy[row]),
                float(zobs[row]))

    if sort_by_zobs:
        idx = np.argsort(zobs)
        xhat = xhat[idx]
        sigx = sigx[idx]
        yhat = yhat[idx]
        sigy = sigy[idx]
        zobs = zobs[idx]

    return xhat, sigx, yhat, sigy, zobs

def load_xy_and_uncertainties_from_fullmocks(
    fits_path,
    n_objects=None,
    random_seed=None,
    apply_valid_mask=True,
):
    """
    Load data from a fullmocks FITS file (TF_extended_AbacusSummit_*.fits).

    Applies MAIN=True filter and validity mask, then optionally subsamples.

    Returns
    -------
    xhat, sigma_x, yhat, sigma_y, zobs, y_true : np.ndarray
        xhat    = LOGVROT - 2.0  (log10(V_rot / 100 km/s))
        sigma_x = LOGVROT_ERR
        yhat    = R_ABSMAG_SB26  (observed magnitude)
        sigma_y = R_ABSMAG_SB26_ERR
        zobs    = ZOBS
        y_true  = R_ABSMAG_SB26_TRUE  (true magnitude from simulation)
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        main_mask = np.asarray(data["MAIN"], dtype=bool)
        data = data[main_mask]

        logvrot   = np.asarray(data["LOGVROT"],             dtype=float)
        logvrot_e = np.asarray(data["LOGVROT_ERR"],         dtype=float)
        absmag    = np.asarray(data["R_ABSMAG_SB26"],       dtype=float)
        absmag_e  = np.asarray(data["R_ABSMAG_SB26_ERR"],   dtype=float)
        zobs      = np.asarray(data["ZOBS"],                dtype=float)
        y_true    = np.asarray(data["R_ABSMAG_SB26_TRUE"],    dtype=float)

    xhat    = logvrot - 2.0
    sigma_x = logvrot_e

    if apply_valid_mask:
        mask = (
            np.isfinite(xhat) & np.isfinite(sigma_x) &
            np.isfinite(absmag) & np.isfinite(absmag_e) &
            np.isfinite(zobs) & np.isfinite(y_true) &
            (logvrot > 0) & (sigma_x > 0) & (absmag_e >= 0)
        )
        xhat    = xhat[mask]
        sigma_x = sigma_x[mask]
        absmag  = absmag[mask]
        absmag_e = absmag_e[mask]
        zobs    = zobs[mask]
        y_true  = y_true[mask]

    if n_objects is not None and n_objects < len(xhat):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(xhat), size=n_objects, replace=False)
        xhat     = xhat[idx]
        sigma_x  = sigma_x[idx]
        absmag   = absmag[idx]
        absmag_e = absmag_e[idx]
        zobs     = zobs[idx]
        y_true   = y_true[idx]

    return xhat, sigma_x, absmag, absmag_e, zobs, y_true


def read_cmdstan_posterior(
    pattern: Union[str, Path],
    *,
    keep: Optional[Iterable[str]] = None,
    drop_diagnostics: bool = False,
    sort_files: bool = True,
) -> pd.DataFrame:
    """
    Read CmdStan CSV posterior draws from one or many chain files.

    Parameters
    ----------
    pattern
        Filename or glob pattern, e.g. "ariel_normal_*.csv" or "ariel_normal_?.csv".
    keep
        Optional iterable of column names to keep (must match header names exactly),
        e.g. ["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"].
        If None, keep all columns.
    drop_diagnostics
        If True, drop typical sampler diagnostic columns like lp__, stepsize__, etc.
    sort_files
        If True, sort matched files for stable ordering.

    Returns
    -------
    pandas.DataFrame
        Concatenated draws across all matched files (rows = draws, cols = parameters).

    Notes
    -----
    CmdStan CSVs have comment/metadata lines starting with '#'; these are ignored.
    """
    pattern = str(pattern)
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    if sort_files:
        files = sorted(files)

    dfs = []
    for f in files:
        df = pd.read_csv(f, comment="#")  # ignore CmdStan metadata lines
        df["chain_file"] = Path(f).name   # optional provenance
        dfs.append(df)

    draws = pd.concat(dfs, axis=0, ignore_index=True)

    if drop_diagnostics:
        diag_prefixes = ("lp__", "accept_stat__", "stepsize__", "treedepth__",
                         "n_leapfrog__", "divergent__", "energy__")
        cols = [c for c in draws.columns if not c.startswith(diag_prefixes)]
        draws = draws[cols]

    if keep is not None:
        keep = list(keep)
        missing = [c for c in keep if c not in draws.columns]
        if missing:
            raise KeyError(f"Requested columns not found: {missing}\n"
                           f"Available columns include: {list(draws.columns)[:20]} ...")
        # Always keep provenance if present
        if "chain_file" in draws.columns and "chain_file" not in keep:
            keep = keep + ["chain_file"]
        draws = draws[keep]

    return draws

def ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star):
    """
    Compute posterior predictive mean and SD of y_* for all galaxies (Normal case),
    marginalizing over posterior draws by Monte Carlo *over draws only* (no inner sampling).

    Returns
    -------
    mean_y : (G,) array
    sd_y   : (G,) array
    """
    xhat_star = np.asarray(xhat_star, dtype=float)          # (G,)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)    # (G,)
    G = xhat_star.size
    M = len(draws)

    # Extract draws as NumPy arrays (fast)
    s  = draws["slope"].to_numpy(float)          # (M,)
    c  = draws["intercept.1"].to_numpy(float)    # (M,)
    six = draws["sigma_int_x"].to_numpy(float)   # (M,)
    siy = draws["sigma_int_y"].to_numpy(float)   # (M,)
    mu0 = draws["mu_y_TF"].to_numpy(float)       # (M,)
    tau = draws["tau"].to_numpy(float)           # (M,)

    if np.any(s == 0):
        raise ValueError("Found slope == 0 in draws; model requires s != 0.")
    if np.any(sigma_x_star < 0) or np.any(six < 0) or np.any(siy < 0) or np.any(tau < 0):
        raise ValueError("Negative SD encountered.")

    # Broadcast to (M,G)
    sMG   = s[:, None]
    cMG   = c[:, None]
    mu0MG = mu0[:, None]
    V0MG  = (tau[:, None] ** 2)

    sigma_x_tot2 = (six[:, None] ** 2) + (sigma_x_star[None, :] ** 2)   # (M,G)
    mu_L = cMG + sMG * xhat_star[None, :]                               # (M,G)
    V_L  = (sMG ** 2) * sigma_x_tot2                                    # (M,G)

    # Conjugate posterior for y_TF | xhat (assume non-degenerate variances)
    V_post  = 1.0 / (1.0 / V0MG + 1.0 / V_L)
    mu_post = V_post * (mu0MG / V0MG + mu_L / V_L)

    # y_* | xhat, theta : Normal(mu_post, V_post + sigma_int_y^2)
    V_ystar = V_post + (siy[:, None] ** 2)

    # Mixture moments over theta-draws:
    mean_y = mu_post.mean(axis=0)  # (G,)

    # Var(Y) = E[Var(Y|theta)] + Var(E[Y|theta])
    #        = E[V_ystar] + (E[mu_post^2] - (E[mu_post])^2)
    var_y = V_ystar.mean(axis=0) + (mu_post**2).mean(axis=0) - mean_y**2
    sd_y = np.sqrt(var_y)

    return mean_y, sd_y



###############################################################################
import json
from pathlib import Path

import numpy as np

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(z):
    """Standard normal PDF, vectorized."""
    return np.exp(-0.5 * z**2) / _SQRT2PI


def _Phi(z):
    """Standard normal CDF, vectorized (no SciPy)."""
    return 0.5 * (1.0 + erf(z / _SQRT2))


def ystar_pp_mean_sd_tophat_vectorized(
    draws,
    xhat_star,
    sigma_x_star,
    *,
    bounds_json="DESI_input.json",
    y_min_key="y_min",
    y_max_key="y_max",
    y_min=None,
    y_max=None,
    on_bad_Z="raise",   # "raise" or "floor"
    Z_floor=1e-300,
):
    """
    Compute posterior predictive mean and SD of latent y_* for all galaxies (Top-Hat y_TF prior),
    marginalizing over posterior draws by Monte Carlo over draws only (no inner sampling).

    Top-Hat model (single draw theta):
        y_TF ~ Unif(y_min, y_max)      [SCALARS]
        x    | y_TF ~ Normal((y_TF - c)/s, sigma_int_x^2)
        xhat | x    ~ Normal(x,           sigma_x_star^2)
        y_*  | y_TF ~ Normal(y_TF,        sigma_int_y^2)

    This returns mixture moments over theta-draws:
        mean_y[g] = E_theta[ E(y_* | theta) ]
        sd_y[g]   = sqrt( E_theta[Var(y_*|theta)] + Var_theta(E(y_*|theta)) )

    Bounds behavior
    --------------
    - If y_min/y_max are not provided, they are loaded as scalars from `bounds_json`
      (e.g. DESI_input.json) using keys y_min_key/y_max_key.
    - If y_min and y_max are provided explicitly, the JSON is not read.

    Parameters
    ----------
    draws : pandas.DataFrame
        Must contain columns: "slope", "intercept.1", "sigma_int_x", "sigma_int_y".
    xhat_star, sigma_x_star : array-like, shape (G,)
    bounds_json : str/path-like
        JSON file containing scalar y_min and y_max.
    y_min_key, y_max_key : str
        Keys in JSON for the scalar bounds.
    y_min, y_max : float or None
        Optional scalar overrides.
    on_bad_Z : {"raise","floor"}
        Handling when Z = Phi(beta)-Phi(alpha) is non-positive/non-finite.
    Z_floor : float
        Floor used when on_bad_Z="floor".

    Returns
    -------
    mean_y : (G,) np.ndarray
    sd_y   : (G,) np.ndarray
    """
    xhat_star = np.asarray(xhat_star, dtype=float)          # (G,)
    sigma_x_star = np.asarray(sigma_x_star, dtype=float)    # (G,)
    G = xhat_star.size

    # ---- Load scalar bounds if not provided
    if y_min is None or y_max is None:
        if bounds_json is None:
            raise ValueError("Provide scalar y_min and y_max, or set bounds_json to a JSON file path.")
        bounds_json = Path(bounds_json)
        with bounds_json.open("r") as f:
            stan_data = json.load(f)
        if y_min_key not in stan_data or y_max_key not in stan_data:
            raise KeyError(
                f"Missing {y_min_key!r} or {y_max_key!r} in {bounds_json}. "
                f"Available keys: {sorted(list(stan_data.keys()))}"
            )
        y_min = stan_data[y_min_key]
        y_max = stan_data[y_max_key]

    a = float(y_min)
    b = float(y_max)
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("y_min and y_max must be finite scalars.")
    if not (a < b):
        raise ValueError(f"Require y_min < y_max; got y_min={a}, y_max={b}.")

    # ---- Extract draws (M,)
    s   = draws["slope"].to_numpy(float)          # (M,)
    c   = draws["intercept.1"].to_numpy(float)    # (M,)
    six = draws["sigma_int_x"].to_numpy(float)    # (M,)
    siy = draws["sigma_int_y"].to_numpy(float)    # (M,)

    if np.any(s == 0):
        raise ValueError("Found slope == 0 in draws; model requires s != 0.")
    if np.any(sigma_x_star < 0) or np.any(six < 0) or np.any(siy < 0):
        raise ValueError("Negative SD encountered.")

    # ---- Broadcast to (M,G)
    sMG = s[:, None]
    cMG = c[:, None]

    sigma_x_tot2 = (six[:, None] ** 2) + (sigma_x_star[None, :] ** 2)   # (M,G)
    mu_L = cMG + sMG * xhat_star[None, :]                               # (M,G)
    sigma_L2 = (sMG ** 2) * sigma_x_tot2                                # (M,G)
    sigma_L = np.sqrt(sigma_L2)                                         # (M,G)

    # ---- Truncated normal moments for y_TF | xhat, theta
    mean_yTF = np.empty_like(mu_L)
    var_yTF  = np.empty_like(mu_L)

    deg = (sigma_L == 0.0)
    if np.any(deg):
        mu_deg = mu_L[deg]  # flat
        ok = (mu_deg >= a) & (mu_deg <= b)
        if not np.all(ok):
            raise ValueError(
                "Encountered sigma_L == 0 with mu_L outside [y_min,y_max] for at least one (draw, galaxy). "
                "This implies zero posterior mass under the Top-Hat prior."
            )
        mean_yTF[deg] = mu_deg
        var_yTF[deg] = 0.0

    nd = ~deg
    if np.any(nd):
        mu  = mu_L[nd]
        sig = sigma_L[nd]

        alpha = (a - mu) / sig
        beta  = (b - mu) / sig

        # Compute log(Z) = log(Φ(β) − Φ(α)) in a numerically stable way.
        # When α ≥ 0 (both tails in the right half), Z = SF(α) − SF(β) and
        # norm.logsf is accurate.  Otherwise use norm.logcdf which is accurate
        # in the left tail.  The naive Z = Φ(β)−Φ(α) underflows to 0 for
        # large positive α, causing φ/Z to overflow and the variance to become
        # −inf / nan.
        use_sf    = alpha >= 0.0
        log_sf_a  = norm.logsf(alpha)
        log_sf_b  = norm.logsf(beta)
        log_cdf_a = norm.logcdf(alpha)
        log_cdf_b = norm.logcdf(beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_Z_sf  = log_sf_a  + np.log1p(-np.exp(np.clip(log_sf_b  - log_sf_a,  -np.inf, 0.0)))
            log_Z_cdf = log_cdf_b + np.log1p(-np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0)))
        log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)

        if on_bad_Z == "raise":
            if np.any(~np.isfinite(log_Z)):
                raise ValueError("log(Z) is non-finite for some (draw, galaxy).")
        elif on_bad_Z == "floor":
            log_Z = np.maximum(log_Z, np.log(Z_floor))
        else:
            raise ValueError("on_bad_Z must be 'raise' or 'floor'.")

        # Compute la = φ(α)/Z and lb = φ(β)/Z in log-space so the ratio stays
        # O(α)-scale even when both φ and Z are tiny.
        log_phi_a = norm.logpdf(alpha)
        log_phi_b = norm.logpdf(beta)
        la = np.exp(log_phi_a - log_Z)   # ≈ α for large α
        lb = np.exp(log_phi_b - log_Z)   # ≈ β for large |β|; ≈ 0 when β >> α

        t = la - lb
        m = mu + sig * t

        u = alpha * la - beta * lb
        v = (sig ** 2) * (1.0 + u - t**2)
        v = np.maximum(v, 0.0)           # guard against residual floating-point negative values

        mean_yTF[nd] = m
        var_yTF[nd]  = v

    # ---- y_* adds intrinsic y-scatter
    mean_ystar = mean_yTF
    var_ystar  = var_yTF + (siy[:, None] ** 2)

    # ---- Mixture moments over draws
    mean_y = mean_ystar.mean(axis=0)  # (G,)
    var_y  = var_ystar.mean(axis=0) + (mean_ystar**2).mean(axis=0) - mean_y**2
    sd_y   = np.sqrt(var_y)

    return mean_y, sd_y

#############################################################################


# def ariel_main():
#     draws = read_cmdstan_posterior(
#         "ariel_normal_?.csv",
#         keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
#         drop_diagnostics=True,
#     )

#     galaxy_csv = "data/TF_mock_tophat-mag_input.csv"
#     xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_csv(
#         galaxy_csv, row=None, sort_by_zobs=True
#     )

#     mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)

#     # Your plot uses (predicted mean - observed yhat)
#     mean_y = mean_pred - yhat_star
#     sigma_y = sd_pred

#     plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.01)
#     plt.show()

def DESI_normal():
    draws = read_cmdstan_posterior(
        "DESI_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )

    # galaxy_fits = "data/DESI-DR1_TF_pv_cat_v15.fits"
    # xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_desi(
    #     galaxy_fits, row=None, sort_by_zobs=False
    # )

    # mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)

    # # Your plot uses (predicted mean - observed yhat)
    # mean_y = mean_pred - yhat_star
    # sigma_y = sd_pred

    # # plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1)

    galaxy_json = "DESI_input.json"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_stan_json(
        galaxy_json)
    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)
    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred


    fig, ax, img = create_average_grid_image(xhat_star, yhat_star, mean_y, grid_resolution_x=50, grid_resolution_y=50)
    ax.set_xlabel(r'$\log{V/V_0}$')
    ax.set_ylabel(r'$M$')
    # ax.set_title(r'$M_{\text{predicted}} - M$ (Filtered Subset)')

    # 4. Add colorbar using the returned 'img' object
    fig.colorbar(img, ax=ax, label='Average Magnitude Difference')

    plt.savefig('DESI_normal_grid.png', dpi=300)
    plt.clf()


    plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1,label="Normal")
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.legend()
    plt.savefig("DESI_redshift_normal.png", dpi=300)
    plt.clf()

def DESI_tophat():

    draws = read_cmdstan_posterior(
        "DESI_tophat_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
        drop_diagnostics=True,
    )

    # galaxy_fits = "data/DESI-DR1_TF_pv_cat_v15.fits"
    # xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_desi(
    #     galaxy_fits, row=None, sort_by_zobs=False
    # )
    # y_threshold = -18  # Replace with your specific value

    # # 2. Create a boolean mask
    # # This creates an array of True/False values the same length as your data
    # mask = yhat_star < y_threshold

    # # 3. Apply the mask to all related arrays
    # xhat_star = xhat_star[mask]
    # yhat_star = yhat_star[mask]
    # zobs_star = zobs_star[mask]

    # # If you also need the uncertainties for the subset:
    # sigma_x_star = sigma_x_star[mask]
    # sigma_y_star = sigma_y_star[mask]

    # mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(draws, xhat_star, sigma_x_star, bounds_json="DESI_input.json")

    # # Your plot uses (predicted mean - observed yhat)
    # mean_y = mean_pred - yhat_star
    # sigma_y = sd_pred
    # create_average_grid_image(xhat_star, yhat_star, mean_y, grid_resolution_x=50, grid_resolution_y=50)
    # plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1)

    galaxy_json = "DESI_input.json"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_stan_json(
        galaxy_json)
    

    mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(draws, xhat_star, sigma_x_star, y_min=-22.5-0.1, y_max=-18.5+0.1)
    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    fig, ax, img = create_average_grid_image(xhat_star, yhat_star, mean_y, grid_resolution_x=50, grid_resolution_y=50)
    ax.set_xlabel(r'$\log{V/V_0}$')
    ax.set_ylabel(r'$M$')
    # ax.set_title(r'$M_{\text{predicted}} - M$ (Filtered Subset)')

    # 4. Add colorbar using the returned 'img' object
    fig.colorbar(img, ax=ax, label='Average Magnitude Difference')

    plt.savefig('DESI_tophat_grid.png', dpi=300)
    plt.clf()

    # redshift plot
    fig, ax, img = create_average_grid_image(xhat_star, yhat_star, zobs_star, grid_resolution_x=50, grid_resolution_y=50)
    ax.set_xlabel(r'$\log{V/V_0}$')
    ax.set_ylabel(r'$M$')
    # ax.set_title(r'Redshift')

    # 2. Add the colorbar
    cbar = fig.colorbar(img, ax=ax, label='Average Redshift')

    # 3. Get the symmetric limit (e.g., if data is ±0.5, this is 0.5)
    # CenteredNorm stores this in 'halfrange'
    current_vmax = img.norm.halfrange 

    # 4. CRITICAL: Only change the VIEW of the colorbar, not the data mapping
    # This crops the physical bar so it starts at White (0) and ends at Red (vmax)
    cbar.ax.set_ylim(0, current_vmax)
    cbar.set_ticks(np.linspace(0, current_vmax, 5))
    plt.savefig('DESI_redshift_grid.png', dpi=300)
    plt.clf()


    plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1,label="Top-Hat")
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.legend()
    plt.ylim((-9,6))
    plt.savefig("DESI_redshift_tophat.png", dpi=300)
    plt.clf()

    draws = read_cmdstan_posterior(
    "DESI_normal_?.csv",
    keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
    drop_diagnostics=True,)
    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)
    mean_y2 = mean_pred - yhat_star
    sigma_y2 = sd_pred
    # plt.errorbar(zobs_star, mean_y2, yerr=sigma_y2, fmt="o", alpha=0.1,label="Normal")



    plt.scatter(mean_y, mean_y2, alpha=0.1)
    plt.savefig("DESI_tophat_vs_normal.png", dpi=300)
    plt.clf()


def ariel_normal():
    # Posterior draws from the Normal (Gaussian) TF model
    draws = read_cmdstan_posterior(
        "ariel_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )

    # Load mock data (Stan JSON)
    galaxy_json = "ariel_n10000_input.json"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = (
        load_xy_and_uncertainties_from_stan_json(galaxy_json)
    )

    # Posterior predictive for y*
    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)

    # Plot (predicted mean - observed yhat)
    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    plt.errorbar(
        zobs_star, mean_y, yerr=sigma_y,
        fmt="o", alpha=0.1, label="Normal"
    )
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.legend()
    plt.savefig("ariel_redshift_normal.png", dpi=300)
    plt.clf()


def ariel_tophat():
    # Load mock data (Stan JSON)
    galaxy_json = "ariel_n10000_input.json"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = (
        load_xy_and_uncertainties_from_stan_json(galaxy_json)
    )

    # Posterior draws from the Top-Hat (truncated/uniform) TF model
    draws = read_cmdstan_posterior(
        "ariel_tophat_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
        drop_diagnostics=True,
    )

    # Posterior predictive for y* under top-hat bounds
    # (Adjust y_min/y_max to match your mock selection function.)
    mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
        draws,
        xhat_star,
        sigma_x_star,
        y_min=-22.5 - 0.1,
        y_max=-18.5 + 0.1,
        # alternatively (if your function supports it):
        # bounds_json=galaxy_json,
    )

    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    plt.errorbar(
        zobs_star, mean_y, yerr=sigma_y,
        fmt="o", alpha=0.1, label="Top-Hat"
    )
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.legend()
    plt.savefig("ariel_redshift_tophat.png", dpi=300)
    plt.clf()

    # Compare Top-Hat vs Normal posterior predictive mean residuals
    draws_normal = read_cmdstan_posterior(
        "ariel_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )
    mean_pred_n, sd_pred_n = ystar_pp_mean_sd_normal_vectorized(
        draws_normal, xhat_star, sigma_x_star
    )
    mean_y_normal = mean_pred_n - yhat_star

    plt.scatter(mean_y, mean_y_normal, alpha=0.1)
    plt.xlabel("Top-Hat: mean_pred - y_obs (mag)")
    plt.ylabel("Normal: mean_pred - y_obs (mag)")
    plt.savefig("ariel_tophat_vs_normal.png", dpi=300)
    plt.clf()

import numpy as np
import matplotlib.pyplot as plt

def DESI(kind="normal",
         grid_resolution_x=50,
         grid_resolution_y=50,
         # tophat-only bounds:
         y_min=-22.5 - 0.1,
         y_max=-18.5 + 0.1,
         # grids:
         make_residual_grid=True,
         make_redshift_grid=True,
         # tophat-only comparison plot:
         compare_tophat_vs_normal=True,
         run_dir=None):
    kind = kind.lower()
    if kind not in {"normal", "tophat"}:
        raise ValueError("kind must be 'normal' or 'tophat'")
    _p = lambda name: os.path.join(run_dir, name) if run_dir else f"DESI_{name}"

    # --- load config and derive input FITS path ---
    with open(_p("config.json"), "r") as f:
        cfg = json.load(f)
    galaxy_fits = cfg["source"]

    # --- load data ---
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_desi(
        galaxy_fits, row=None, sort_by_zobs=False
    )

    # --- posterior + predictive ---
    if kind == "normal":
        draws = read_cmdstan_posterior(
            _p("normal_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)
        label = "Normal"
    else:
        draws = read_cmdstan_posterior(
            _p("tophat_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
            draws, xhat_star, sigma_x_star, y_min=y_min, y_max=y_max, on_bad_Z="floor", Z_floor=1e-300
        )
        label = "Top-Hat"

    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    main_mask = _apply_main_cuts(cfg, xhat_star, yhat_star)

    xhat_star2    = xhat_star[main_mask]
    sigma_x_star2 = sigma_x_star[main_mask]
    yhat_star2    = yhat_star[main_mask]
    zobs_star2    = zobs_star[main_mask]

    # --- posterior + predictive ---
    if kind == "normal":
        mean_pred2, sd_pred2 = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star2, sigma_x_star2)
    else:
        mean_pred2, sd_pred2 = ystar_pp_mean_sd_tophat_vectorized(
            draws, xhat_star2, sigma_x_star2, y_min=y_min, y_max=y_max, on_bad_Z="floor", Z_floor=1e-300
        )

    mean_y2 = mean_pred2 - yhat_star2

    # --- GRID: residuals on (xhat, yhat) — selection sample only ---
    if make_residual_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star2, yhat_star2, mean_y2,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        # ax.set_title(r'$M_{\text{predicted}} - M$')
        fig.colorbar(img, ax=ax, label='Average Magnitude Difference')
        fig.savefig(_p(f'{kind}_grid.png'), dpi=300)
        plt.close(fig)

        # --- GRID: residuals on (xhat, yhat) — full input sample ---
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, mean_y,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        fig.colorbar(img, ax=ax, label='Average Magnitude Difference')
        fig.savefig(_p(f'{kind}_grid_full.png'), dpi=300)
        plt.close(fig)

    # --- GRID: redshift on (xhat, yhat) ---
    if make_redshift_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, zobs_star,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        # ax.set_title(r'Redshift')

        cbar = fig.colorbar(img, ax=ax, label='Average Redshift')

        # Keep your DESI_tophat behavior: crop the *visible* colorbar range if CenteredNorm-like
        if kind == "tophat" and hasattr(img, "norm") and hasattr(img.norm, "halfrange"):
            current_vmax = img.norm.halfrange
            cbar.ax.set_ylim(0, current_vmax)
            cbar.set_ticks(np.linspace(0, current_vmax, 5))

        fig.savefig(_p(f'redshift_grid_{kind}.png'), dpi=300)
        plt.close(fig)

    # --- redshift residual errorbar plot ---
    plt.scatter(zobs_star, mean_y, marker=".", alpha=0.2, label="DR2 PV Spirals")
    plt.scatter(zobs_star2, mean_y2, marker=".", alpha=0.2, label="Main Sample")

    
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.axhline(y=0, color='gray', linestyle='dashed', linewidth=1.5, label='y=0 line')
    plt.legend()
    plt.ylim((-8,4))
    plt.savefig(_p(f"redshift_{kind}.png"), dpi=300)
    plt.clf()

    # --- optional: tophat vs normal scatter comparison ---
    if kind == "tophat" and compare_tophat_vs_normal and glob.glob(_p("normal_?.csv")):
        draws_n = read_cmdstan_posterior(
            _p("normal_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
        mean_pred_n, sd_pred_n = ystar_pp_mean_sd_normal_vectorized(draws_n, xhat_star, sigma_x_star)
        mean_y_normal = mean_pred_n - yhat_star

        plt.scatter(mean_y, mean_y_normal, alpha=0.1)
        plt.xlabel("Top-Hat: mean_pred - y_obs (mag)")
        plt.ylabel("Normal: mean_pred - y_obs (mag)")
        plt.savefig(_p("tophat_vs_normal.png"), dpi=300)
        plt.clf()

    return mean_y, sigma_y, zobs_star


from matplotlib.colors import TwoSlopeNorm, SymLogNorm
from matplotlib.ticker import FixedLocator, StrMethodFormatter, ScalarFormatter

# ----------------------------------------------------------------------
# Project‑specific helper imports (keep yours)
# ----------------------------------------------------------------------
# from your_module import (load_xy_and_uncertainties_from_desi,
#                         read_cmdstan_posterior,
#                         ystar_pp_mean_sd_normal_vectorized,
#                         ystar_pp_mean_sd_tophat_vectorized,
#                         create_average_grid_image)

def _symmetric_log_ticks(vmin, vmax, linthresh=0.01, n_ticks=7):
    if vmax <= linthresh:
        return np.linspace(vmin, vmax, n_ticks)
    n_pos = (n_ticks - 1) // 2
    pos = np.logspace(np.log10(linthresh), np.log10(vmax), n_pos)
    pos[0] = linthresh
    pos[-1] = vmax
    return np.concatenate((-pos[::-1], [0.0], pos))

def DESI_compare(
    galaxy_fits="data/DESI-DR1_TF_pv_cat_v15.fits",
    grid_resolution_x=50,
    grid_resolution_y=50,
    y_min=-22.5 - 0.1,
    y_max=-18.5 + 0.1,
    make_residual_grid=True,
    make_redshift_grid=True,
    make_scatter=True,
    output_prefix="DESI",
    cmap_diff="RdBu_r",
    fixed_range=1.0,
    log_scale=False,
    log_base=10.0,
    linthresh=0.01,
):
    """
    Plot the magnitude‑prediction difference (top‑hat – normal) and optionally
    show the colour‑bars on a symmetric‑log scale.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Load data
    # ------------------------------------------------------------------
    (xhat_star, sigma_x_star, yhat_star, sigma_y_star,
     zobs_star) = load_xy_and_uncertainties_from_desi(
        galaxy_fits, row=None, sort_by_zobs=False)

    # ------------------------------------------------------------------
    # 2️⃣ Read posteriors
    # ------------------------------------------------------------------
    draws_normal = read_cmdstan_posterior(
        "DESI_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x",
              "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )
    draws_tophat = read_cmdstan_posterior(
        "DESI_tophat_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
        drop_diagnostics=True,
    )

    # ------------------------------------------------------------------
    # 3️⃣ Predict magnitudes
    # ------------------------------------------------------------------
    mean_pred_norm, sd_pred_norm = ystar_pp_mean_sd_normal_vectorized(
        draws_normal, xhat_star, sigma_x_star)
    mean_pred_top, sd_pred_top = ystar_pp_mean_sd_tophat_vectorized(
        draws_tophat,
        xhat_star,
        sigma_x_star,
        y_min=y_min,
        y_max=y_max,
        on_bad_Z="floor",
        Z_floor=1e-300,
    )

    # ------------------------------------------------------------------
    # 4️⃣ Residuals & difference
    # ------------------------------------------------------------------
    resid_norm = mean_pred_norm - yhat_star
    resid_top  = mean_pred_top  - yhat_star
    diff = resid_top - resid_norm
    diff_sd = np.sqrt(sd_pred_top**2 + sd_pred_norm**2)

    diff    = np.nan_to_num(diff,    nan=0.0)
    diff_sd = np.nan_to_num(diff_sd, nan=0.0)

    # ------------------------------------------------------------------
    # 5️⃣ Normaliser + colour‑bar tick objects
    # ------------------------------------------------------------------
    vmin = -fixed_range
    vmax =  fixed_range

    if log_scale:
        norm = SymLogNorm(linthresh=linthresh, linscale=1.0,
                          vmin=vmin, vmax=vmax, base=log_base)

        ticks = _symmetric_log_ticks(vmin, vmax,
                                     linthresh=linthresh, n_ticks=7)
        _cbar_locator   = FixedLocator(ticks)
        _cbar_formatter = StrMethodFormatter("{x:.2f}")
    else:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

        linear_ticks = np.linspace(vmin, vmax, 7)
        _cbar_locator   = FixedLocator(linear_ticks)
        _cbar_formatter = StrMethodFormatter("{x:.2f}")

    # ------------------------------------------------------------------
    # 6️⃣ Residual‑difference grid
    # ------------------------------------------------------------------
    if make_residual_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, diff,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r"$\log{V/V_0}$")
        ax.set_ylabel(r"$M$")
        # ax.set_title(r"$(M_{\rm pred}^{\rm tophat} - M_{\rm pred}^{\rm normal})$")

        img.set_cmap(cmap_diff)
        img.set_norm(norm)

        cbar = fig.colorbar(img, ax=ax, label="Magnitude Difference")
        cbar.ax.yaxis.set_major_locator(_cbar_locator)
        cbar.ax.yaxis.set_major_formatter(_cbar_formatter)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=9)

        fig.savefig(f"{output_prefix}_diff_grid.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 7️⃣ Redshift‑averaged grid
    # ------------------------------------------------------------------
    if make_redshift_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, diff,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r"$\log{V/V_0}$")
        ax.set_ylabel(r"$M$")
        # ax.set_title(
        #     r"Redshift‑averaged $(M_{\rm pred}^{\rm tophat} - M_{\rm pred}^{\rm normal})$"
        # )

        img.set_cmap(cmap_diff)
        img.set_norm(norm)

        cbar = fig.colorbar(img, ax=ax, label="Magnitude Difference")
        cbar.ax.yaxis.set_major_locator(_cbar_locator)
        cbar.ax.yaxis.set_major_formatter(_cbar_formatter)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=9)

        fig.savefig(f"{output_prefix}_redshift_grid_diff.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 8️⃣ Scatter vs redshift (unchanged)
    # ------------------------------------------------------------------
    if make_scatter:
        plt.figure(figsize=(6, 4))
        plt.scatter(zobs_star, diff, s=3, alpha=0.05,
                    label=r"Top‑Hat – Normal")
        plt.axhline(0, color="k", ls="--", linewidth=0.8)

        plt.xscale("log")
        plt.xlabel(r"$z_{\rm obs}$")
        plt.ylabel(r"$\Delta M_{\rm pred}$ (mag)")
        # plt.title(r"Prediction difference vs. redshift")
        plt.ylim(vmin, vmax)            # respects the fixed range
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_redshift_diff.png",
                    dpi=300, bbox_inches="tight")
        plt.close()

    return diff, diff_sd, zobs_star

def ariel(kind="normal",
         galaxy_json=None,
         grid_resolution_x=50,
         grid_resolution_y=50,
         # grids:
         make_residual_grid=True,
         make_redshift_grid=True,
         # tophat-only comparison plot:
         compare_tophat_vs_normal=True,
         run_dir=None):
    kind = kind.lower()
    if kind not in {"normal", "tophat"}:
        raise ValueError("kind must be 'normal' or 'tophat'")
    _p = lambda name: os.path.join(run_dir, name) if run_dir else f"ariel_{name}"
    if galaxy_json is None:
        galaxy_json = _p("input.json") if run_dir else "ariel_n10000_input.json"

    # --- load mock data ---
    # xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = (
    #     load_xy_and_uncertainties_from_stan_json(galaxy_json)
    # )



    galaxy_csv = "data/TF_mock_tophat-mag_input.csv"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_csv(
        galaxy_csv, row=None, sort_by_zobs=True)
    n = zobs_star.shape[0]
    N =10000
    idx = np.random.choice(n, size=N, replace=False)
    xhat_star = xhat_star[idx]
    sigma_x_star = sigma_x_star[idx]
    yhat_star = yhat_star[idx]
    sigma_y_star = sigma_y_star[idx]
    zobs_star = zobs_star[idx]


    # --- posterior + predictive ---
    if kind == "normal":
        draws = read_cmdstan_posterior(
            _p("normal_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)
        label = "Normal"
    else:
        draws = read_cmdstan_posterior(
            _p("tophat_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
            drop_diagnostics=True,
        )
        with open(galaxy_json, 'r') as f:
            data = json.load(f)
            y_min = data.get('y_min')
            y_max = data.get('y_max')

        mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
            draws, xhat_star, sigma_x_star, y_min=y_min, y_max=y_max
            # alternatively (if supported):
            # , bounds_json=galaxy_json
        )
        label = "Top-Hat"

    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    # --- GRID: residuals on (xhat, yhat) ---
    if make_residual_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, mean_y,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        ax.set_title(r'$M_{\text{predicted}} - M$')
        fig.colorbar(img, ax=ax, label='Average Magnitude Difference')
        fig.savefig(_p(f'{kind}_grid.png'), dpi=300)
        plt.close(fig)

    # --- GRID: redshift on (xhat, yhat) ---
    if make_redshift_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, zobs_star,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        ax.set_title(r'Redshift')
        fig.colorbar(img, ax=ax, label='Average Redshift')
        fig.savefig(_p(f'redshift_grid_{kind}.png'), dpi=300)
        plt.close(fig)

    # --- redshift residual errorbar plot ---
    plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1, label=label)
    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.legend()
    plt.savefig(_p(f"redshift_{kind}.png"), dpi=300)
    plt.clf()

    # --- optional: tophat vs normal scatter comparison ---
    if kind == "tophat" and compare_tophat_vs_normal and glob.glob(_p("normal_?.csv")):
        draws_normal = read_cmdstan_posterior(
            _p("normal_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
        mean_pred_n, sd_pred_n = ystar_pp_mean_sd_normal_vectorized(
            draws_normal, xhat_star, sigma_x_star
        )
        mean_y_normal = mean_pred_n - yhat_star

        plt.scatter(mean_y, mean_y_normal, alpha=0.1)
        plt.xlabel("Top-Hat: mean_pred - y_obs (mag)")
        plt.ylabel("Normal: mean_pred - y_obs (mag)")
        plt.savefig(_p("tophat_vs_normal.png"), dpi=300)
        plt.clf()

    return mean_y, sigma_y, zobs_star

def fullmocks(kind="normal",
              galaxy_fits=None,
              grid_resolution_x=50,
              grid_resolution_y=50,
              y_min=None,
              y_max=None,
              make_residual_grid=True,
              make_truth_diff_grid=True,
              make_redshift_grid=True,
              n_objects=None,
              random_seed=None,
              run_dir=None,
              delta_haty_min=0.0,
              delta_haty_max=0.0,
              delta_z_obs_min=0.0,
              delta_z_obs_max=0.0,
              plane_cut=True,
              delta_intercept_plane=0.0,
              delta_intercept_plane2=0.0):
    """
    Posterior predictions for a fullmocks AbacusSummit FITS source.

    In addition to the standard residual grid (mean_pred - yhat_obs), produces a
    truth-diff grid comparing the model prediction to the simulation truth:
        mean_pred - R_ABSMAG_SB26_TRUE

    Parameters
    ----------
    kind : 'normal' or 'tophat'
    galaxy_fits : path to TF_extended_AbacusSummit_*.fits file
    n_objects : optional int to subsample objects for prediction
    random_seed : optional int for reproducible subsampling
    run_dir : output/<run>/ directory containing MCMC chains
    """
    kind = kind.lower()
    if kind not in {"normal", "tophat"}:
        raise ValueError("kind must be 'normal' or 'tophat'")
    _p = lambda name: os.path.join(run_dir, name) if run_dir else f"fullmocks_{name}"

    # --- load data ---
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star, y_true = (
        load_xy_and_uncertainties_from_fullmocks(
            galaxy_fits, n_objects=n_objects, random_seed=random_seed
        )
    )

    # Read input.json once: provides y_min/y_max (tophat) and training selection cuts
    with open(_p("input.json"), "r") as f:
        _cfg = json.load(f)
    if y_min is None:
        y_min = _cfg.get("y_min", y_min)
    if y_max is None:
        y_max = _cfg.get("y_max", y_max)

    # Apply selection cuts relative to training values from input.json
    sel = np.ones(len(xhat_star), dtype=bool)
    if _cfg.get("haty_min") is not None:
        sel &= (yhat_star >= _cfg["haty_min"] + delta_haty_min)
    if _cfg.get("haty_max") is not None:
        sel &= (yhat_star <= _cfg["haty_max"] + delta_haty_max)
    if _cfg.get("z_obs_min") is not None:
        sel &= (zobs_star > _cfg["z_obs_min"] + delta_z_obs_min)
    if _cfg.get("z_obs_max") is not None:
        sel &= (zobs_star <= _cfg["z_obs_max"] + delta_z_obs_max)
    if plane_cut and _cfg.get("slope_plane") is not None \
                 and _cfg.get("intercept_plane") is not None:
        s = _cfg["slope_plane"]
        sel &= (yhat_star >= s * xhat_star + _cfg["intercept_plane"] + delta_intercept_plane)
        if _cfg.get("intercept_plane2") is not None:
            sel &= (yhat_star <= s * xhat_star + _cfg["intercept_plane2"] + delta_intercept_plane2)
    if sel.sum() < len(sel):
        xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star, y_true = (
            arr[sel] for arr in
            (xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star, y_true)
        )

    # --- posterior + predictive ---
    if kind == "normal":
        draws = read_cmdstan_posterior(
            _p("normal_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)
        label = "Normal"
    else:
        draws = read_cmdstan_posterior(
            _p("tophat_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
            draws, xhat_star, sigma_x_star, y_min=y_min, y_max=y_max,
            on_bad_Z="floor", Z_floor=1e-300,
        )
        label = "Top-Hat"

    mean_y = mean_pred - yhat_star

    # --- GRID: residuals on (xhat, yhat) ---
    if make_residual_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, mean_y,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        ax.set_title(r'$M_{\text{predicted}} - M_{\text{obs}}$')
        fig.colorbar(img, ax=ax, label='Average Magnitude Difference')
        fig.savefig(_p(f'{kind}_grid.png'), dpi=300)
        plt.close(fig)

    # --- GRID: pull (prediction vs simulation truth) / prediction uncertainty ---
    if make_truth_diff_grid:
        truth_diff = (mean_pred - y_true) / sd_pred
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, truth_diff,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        ax.set_title(r'$(M_{\text{predicted}} - M_{\text{true}}) / \sigma_{\text{pred}}$')
        fig.colorbar(img, ax=ax, label='Average Pull')
        fig.savefig(_p(f'{kind}_truth_diff_grid.png'), dpi=300)
        plt.close(fig)

    # --- GRID: redshift on (xhat, yhat) ---
    if make_redshift_grid:
        fig, ax, img = create_average_grid_image(
            xhat_star, yhat_star, zobs_star,
            grid_resolution_x=grid_resolution_x,
            grid_resolution_y=grid_resolution_y,
        )
        ax.set_xlabel(r'$\log{V/V_0}$')
        ax.set_ylabel(r'$M$')
        ax.set_title(r'Redshift')
        fig.colorbar(img, ax=ax, label='Average Redshift')
        fig.savefig(_p(f'redshift_grid_{kind}.png'), dpi=300)
        plt.close(fig)

    # --- redshift pull errorbar plot ---
    pull_y     = mean_y / sd_pred
    pull_yerr  = sigma_y_star / sd_pred

    # --- scatter plot with high-pull galaxies overplotted ---
    high_pull = pull_y > 4
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xhat_star[~high_pull], yhat_star[~high_pull],
               s=2, alpha=0.2, color="gray", label=f"N = {(~high_pull).sum()}")
    ax.scatter(xhat_star[high_pull], yhat_star[high_pull],
               s=8, alpha=0.8, color="red", label=f"pull > 4  (N = {high_pull.sum()})")
    if _cfg.get("haty_max") is not None:
        ax.axhline(_cfg["haty_max"] + delta_haty_max, color="red", linestyle="--",
                   linewidth=1.5, label=f"haty_max = {_cfg['haty_max'] + delta_haty_max:.2f}")
    if _cfg.get("haty_min") is not None:
        ax.axhline(_cfg["haty_min"] + delta_haty_min, color="orange", linestyle="--",
                   linewidth=1.5, label=f"haty_min = {_cfg['haty_min'] + delta_haty_min:.2f}")
    if plane_cut and _cfg.get("slope_plane") is not None \
                 and _cfg.get("intercept_plane") is not None:
        s_p = _cfg["slope_plane"]
        x_range = np.array([xhat_star.min() - 0.1, xhat_star.max() + 0.1])
        ax.plot(x_range, s_p * x_range + _cfg["intercept_plane"] + delta_intercept_plane,
                "g--", linewidth=1.5)
        if _cfg.get("intercept_plane2") is not None:
            ax.plot(x_range, s_p * x_range + _cfg["intercept_plane2"] + delta_intercept_plane2,
                    "g-.", linewidth=1.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\hat{x}$ = log($V_{\rm rot}$/100 km/s)")
    ax.set_ylabel(r"$\hat{y}$ = M")
    ax.set_title(f"{label}: pull > 4 highlighted")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(_p(f"{kind}_highpull.png"), dpi=150)
    plt.close(fig)

    plt.errorbar(zobs_star, pull_y, yerr=pull_yerr, fmt="o", alpha=0.1, label=label)

    # weighted mean in log-spaced redshift bins (weights = 1/pull_yerr^2)
    finite = np.isfinite(pull_y) & np.isfinite(pull_yerr)
    p_lo, p_hi = np.percentile(pull_y[finite], [1, 99])
    inlier = finite & (pull_y >= p_lo) & (pull_y <= p_hi)
    z_edges = np.logspace(np.log10(zobs_star.min()), np.log10(zobs_star.max()), 21)
    w = 1.0 / pull_yerr**2
    bin_idx = np.digitize(zobs_star, z_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(z_edges) - 2)
    bin_centers = np.sqrt(z_edges[:-1] * z_edges[1:])
    bin_wmean = np.full(len(z_edges) - 1, np.nan)
    bin_werr  = np.full(len(z_edges) - 1, np.nan)
    for b in range(len(z_edges) - 1):
        mask_b = (bin_idx == b) & inlier
        if mask_b.sum() > 0:
            w_b = w[mask_b]
            bin_wmean[b] = np.sum(w_b * pull_y[mask_b]) / np.sum(w_b)
            bin_werr[b]  = 1.0 / np.sqrt(np.sum(w_b))
    ok = np.isfinite(bin_wmean)
    plt.errorbar(bin_centers[ok], bin_wmean[ok], yerr=bin_werr[ok],
                 fmt="s-", color="red", linewidth=1.5, markersize=5, capsize=3, label="Weighted mean")

    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel("Pull of difference")
    plt.ylim(np.percentile(pull_y[finite], 0.1), np.percentile(pull_y[finite], 99.9))
    plt.legend()
    plt.savefig(_p(f"redshift_{kind}.png"), dpi=300)
    plt.clf()

    # --- 3x3 pull histogram grid in log-spaced redshift bins ---
    z_hist_edges = np.logspace(np.log10(zobs_star[finite].min()),
                               np.log10(zobs_star[finite].max()), 10)  # 9 bins
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    for b, ax in enumerate(axes):
        mask_b = (np.digitize(zobs_star, z_hist_edges) - 1 == b) & finite
        z_lo, z_hi = z_hist_edges[b], z_hist_edges[b + 1]
        vals = pull_y[mask_b]
        counts, _, _ = ax.hist(vals, bins=50, density=True, alpha=0.7)
        ax.set_ylim(0, counts.max() * 1.3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        mu, sig = (np.mean(vals), np.std(vals)) if mask_b.sum() > 0 else (np.nan, np.nan)
        ax.set_title(f"z ∈ [{z_lo:.3f}, {z_hi:.3f}]  N={mask_b.sum()}\n"
                     f"mean={mu:.3f}  std={sig:.3f}", fontsize=8)
        ax.set_xlabel("Pull")
    fig.suptitle(f"{label} pull distribution by redshift bin")
    fig.tight_layout()
    fig.savefig(_p(f"redshift_hist_{kind}.png"), dpi=150)
    plt.close(fig)

    return mean_y, sd_pred, zobs_star


def _apply_main_cuts(cfg, xhat, yhat):
    """Return boolean mask for MAIN=True using config.json cuts (no z cuts)."""
    mask = np.ones(len(xhat), dtype=bool)
    if cfg.get("haty_min") is not None:
        mask &= (yhat >= cfg["haty_min"])
    if cfg.get("haty_max") is not None:
        mask &= (yhat <= cfg["haty_max"])
    slope_plane      = cfg.get("slope_plane")
    intercept_plane  = cfg.get("intercept_plane")
    intercept_plane2 = cfg.get("intercept_plane2")
    if slope_plane is not None and intercept_plane is not None:
        mask &= (yhat >= slope_plane * xhat + intercept_plane)
        if intercept_plane2 is not None:
            mask &= (yhat <= slope_plane * xhat + intercept_plane2)
    return mask


def write_desi_catalog(model, run_dir, fits_path):
    """
    Augment a DESI FITS catalog with TFR-derived quantities and write to
    output/<run>/<model>_catalog.fits.

    New columns added:
      MU_TF        = R_MAG_SB26_CORR - mean_pred
      MU_ERR       = sqrt(R_MAG_SB26_CORR_ERR^2 + sd_pred^2)
      LOGDIST      = 0.2 * ((R_MAG_SB26 - R_ABSMAG_SB26) - MU_TF)
      LOGDIST_ERR  = 0.2 * MU_ERR
      MAIN         = bool (True if passes selection cuts from config.json)
    """
    _p = lambda name: os.path.join(run_dir, name)

    # 1. Read the full FITS, keeping all rows
    z_col_candidates = ("Z_DESI", "zobs", "ZOBS", "Z", "ZHELIO", "Z_CMB", "ZDESI", "ZTRUE")
    with fits.open(fits_path) as hdul:
        primary_hdu = hdul[0].copy()
        table_hdu = hdul[1].copy()
        data = hdul[1].data
        names = set(data.dtype.names or ())
        n_rows = len(data)

        # Resolve z column
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

        # Resolve corrected magnitude error column with fallback
        corr_err_col = "R_MAG_SB26_CORR_ERR"
        if corr_err_col not in names:
            fallback = "R_MAG_SB26_ERR"
            if fallback in names:
                print(f"Warning: {corr_err_col!r} absent; falling back to {fallback!r}")
                corr_err_col = fallback
            else:
                raise ValueError(
                    f"Missing magnitude error column {corr_err_col!r} and fallback "
                    f"{fallback!r}. Available: {sorted(list(names))[:30]} ..."
                )

        # 2. Extract working arrays for all rows
        V         = np.asarray(data["V_0p4R26"],       dtype=float)
        V_err     = np.asarray(data["V_0p4R26_ERR"],   dtype=float)
        app_corr  = np.asarray(data["R_MAG_SB26_CORR"],dtype=float)
        app_corr_err = np.asarray(data[corr_err_col],  dtype=float)
        app       = np.asarray(data["R_MAG_SB26"],     dtype=float)
        abs_mag_col = "R_ABSMAG_SB26"
        if abs_mag_col not in names:
            fallback = "R_MAG_SB26_ABS"
            if fallback in names:
                print(f"Warning: {abs_mag_col!r} absent; falling back to {fallback!r}")
                abs_mag_col = fallback
            else:
                raise ValueError(
                    f"Missing absolute magnitude column {abs_mag_col!r} and fallback "
                    f"{fallback!r}. Available: {sorted(list(names))[:30]} ..."
                )
        abs_mag   = np.asarray(data[abs_mag_col],      dtype=float)
        zobs      = np.asarray(data[z_col_use],        dtype=float)

    with np.errstate(invalid='ignore', divide='ignore'):
        xhat    = np.where(V > 0, np.log10(V / 100.0),            np.nan)
        sigma_x = np.where(V > 0, V_err / (V * np.log(10.0)),     np.nan)

    # 3. Validity mask for prediction
    valid = (
        np.isfinite(V)   & (V > 0)   &
        np.isfinite(V_err) & (V_err > 0) &
        np.isfinite(xhat) &
        np.isfinite(sigma_x) & (sigma_x > 0)
    )

    # 4. Load posterior draws
    if model == "normal":
        draws = read_cmdstan_posterior(
            _p("normal_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
    else:
        draws = read_cmdstan_posterior(
            _p("tophat_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
            drop_diagnostics=True,
        )

    # 5. Compute mean_pred and sd_pred for valid rows only
    xhat_valid    = xhat[valid]
    sigma_x_valid = sigma_x[valid]

    if model == "normal":
        mean_pred_valid, sd_pred_valid = ystar_pp_mean_sd_normal_vectorized(
            draws, xhat_valid, sigma_x_valid
        )
    else:
        with open(_p("input.json"), "r") as f:
            input_data = json.load(f)
        y_min = input_data.get("y_min")
        y_max = input_data.get("y_max")
        mean_pred_valid, sd_pred_valid = ystar_pp_mean_sd_tophat_vectorized(
            draws, xhat_valid, sigma_x_valid,
            y_min=y_min, y_max=y_max,
            on_bad_Z="floor", Z_floor=1e-300,
        )

    # 6. Map predictions back to full-length arrays (NaN for invalid rows)
    mean_pred_full = np.full(n_rows, np.nan)
    sd_pred_full   = np.full(n_rows, np.nan)
    mean_pred_full[valid] = mean_pred_valid
    sd_pred_full[valid]   = sd_pred_valid

    # 7. Compute new columns
    MU_TF       = app_corr - mean_pred_full
    MU_ERR      = np.sqrt(app_corr_err**2 + sd_pred_full**2)
    MU_ZCMB     = app - abs_mag                        # intermediate only
    LOGDIST     = 0.2 * (MU_ZCMB - MU_TF)
    LOGDIST_ERR = 0.2 * MU_ERR

    # 8. Compute MAIN flag using config.json selection cuts
    with open(_p("config.json"), "r") as f:
        cfg = json.load(f)

    main = valid & _apply_main_cuts(cfg, xhat, abs_mag)

    # 9. Write output FITS: original columns + five new columns
    new_cols = [
        fits.Column(name="MU_TF",       format="E", array=MU_TF.astype(np.float32)),
        fits.Column(name="MU_ERR",      format="E", array=MU_ERR.astype(np.float32)),
        fits.Column(name="LOGDIST",     format="E", array=LOGDIST.astype(np.float32)),
        fits.Column(name="LOGDIST_ERR", format="E", array=LOGDIST_ERR.astype(np.float32)),
        fits.Column(name="MAIN",        format="L", array=main),
    ]
    all_cols = fits.ColDefs(list(table_hdu.columns) + new_cols)
    new_table_hdu = fits.BinTableHDU.from_columns(all_cols)
    out_hdul = fits.HDUList([primary_hdu, new_table_hdu])
    out_path = _p(f"{model}_catalog.fits")
    out_hdul.writeto(out_path, overwrite=True)

    print(f"Written {n_rows} rows to {out_path}")
    print(f"  MAIN: {main.sum()} objects pass selection cuts")
    print(f"  MU_TF finite: {np.isfinite(MU_TF).sum()} objects")


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
    siy = draws["sigma_int_y"].to_numpy(float)  # (M,)
    mu0 = draws["mu_y_TF"].to_numpy(float)      # (M,)
    tau = draws["tau"].to_numpy(float)          # (M,)

    accum    = np.zeros((G, G), dtype=float)
    var_accum = np.zeros(G,     dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        sc    = s[start:end][:, None]
        cc    = c[start:end][:, None]
        six_c = six[start:end][:, None]
        siy_c = siy[start:end][:, None]
        mu0_c = mu0[start:end][:, None]
        V0    = tau[start:end][:, None] ** 2

        sigma_x_tot2 = six_c**2 + sigma_x_star[None, :]**2  # (B, G)
        mu_L = cc + sc * xhat_star[None, :]                  # (B, G)
        V_L  = sc**2 * sigma_x_tot2                          # (B, G)

        Vp       = 1.0 / (1.0 / V0 + 1.0 / V_L)
        mu_chunk = Vp * (mu0_c / V0 + mu_L / V_L)           # (B, G)

        mu_centered = mu_chunk - mean_y[None, :]
        accum += mu_centered.T @ mu_centered  # (G, G) rank-B update

        # Accumulate E_theta[Var(y_*|theta)] for diagonal correction
        V_ystar = Vp + siy_c**2                              # (B, G)
        var_accum += V_ystar.sum(axis=0)

    cov = accum / M
    np.fill_diagonal(cov, np.diag(cov) + var_accum / M)
    return cov


def plot_cov(cov, output_path, *, title="Posterior predictive covariance", vmax=None):
    """
    Save a visualisation of the (G, G) covariance matrix to a PNG.

    Shows two panels: the covariance matrix and the derived correlation matrix
    r[g1,g2] = cov[g1,g2] / sqrt(var[g1]*var[g2]), both with a symmetric
    diverging colormap centred at zero.

    Parameters
    ----------
    cov : (G, G) ndarray
    output_path : str or Path
        File to write (PNG).
    title : str
        Suptitle for the figure.
    vmax : float or None
        Colour-scale limit for the covariance panel.  Defaults to the 99th
        percentile of |cov| so a few large outliers don't wash out the image.
    """
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(cov), 99))
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im0 = ax.imshow(cov, origin="upper", aspect="auto",
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
    ax.set_xlabel("galaxy index")
    ax.set_ylabel("galaxy index")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved covariance image to {output_path}")


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
    siy = draws["sigma_int_y"].to_numpy(float)  # (M,)

    accum     = np.zeros((G, G), dtype=float)
    var_accum = np.zeros(G,      dtype=float)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        sc    = s[start:end][:, None]
        cc    = c[start:end][:, None]
        six_c = six[start:end][:, None]
        siy_c = siy[start:end][:, None]

        sigma_x_tot2 = six_c**2 + sigma_x_star[None, :]**2  # (B, G)
        mu_L    = cc + sc * xhat_star[None, :]               # (B, G)
        sigma_L = np.sqrt(sc**2 * sigma_x_tot2)              # (B, G)

        # Truncated-normal conditional mean and variance for each (draw, galaxy)
        mu_chunk  = np.empty_like(mu_L)
        var_chunk = np.empty_like(mu_L)

        deg = (sigma_L == 0.0)
        if np.any(deg):
            mu_deg = mu_L[deg]
            ok = (mu_deg >= a) & (mu_deg <= b)
            if not np.all(ok):
                raise ValueError(
                    "Encountered sigma_L == 0 with mu_L outside [y_min, y_max] "
                    "for at least one (draw, galaxy)."
                )
            mu_chunk[deg]  = mu_deg
            var_chunk[deg] = 0.0

        nd = ~deg
        if np.any(nd):
            mu  = mu_L[nd]
            sig = sigma_L[nd]

            alpha = (a - mu) / sig
            beta  = (b - mu) / sig

            use_sf    = alpha >= 0.0
            log_sf_a  = norm.logsf(alpha)
            log_sf_b  = norm.logsf(beta)
            log_cdf_a = norm.logcdf(alpha)
            log_cdf_b = norm.logcdf(beta)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_Z_sf  = log_sf_a  + np.log1p(-np.exp(np.clip(log_sf_b  - log_sf_a,  -np.inf, 0.0)))
                log_Z_cdf = log_cdf_b + np.log1p(-np.exp(np.clip(log_cdf_a - log_cdf_b, -np.inf, 0.0)))
            log_Z = np.where(use_sf, log_Z_sf, log_Z_cdf)

            if on_bad_Z == "raise":
                if np.any(~np.isfinite(log_Z)):
                    raise ValueError(
                        "log(Z) is non-finite for some (draw, galaxy)."
                    )
            elif on_bad_Z == "floor":
                log_Z = np.maximum(log_Z, np.log(Z_floor))
            else:
                raise ValueError("on_bad_Z must be 'raise' or 'floor'.")

            la = np.exp(norm.logpdf(alpha) - log_Z)
            lb = np.exp(norm.logpdf(beta)  - log_Z)
            t  = la - lb
            mu_chunk[nd]  = mu + sig * t
            u = alpha * la - beta * lb
            v = sig**2 * (1.0 + u - t**2)
            var_chunk[nd] = np.maximum(v, 0.0)

        mu_centered = mu_chunk - mean_y[None, :]
        accum += mu_centered.T @ mu_centered  # (G, G) rank-B update

        # Accumulate E_theta[Var(y_*|theta)] = E[var_yTF + sigma_int_y^2]
        var_accum += (var_chunk + siy_c**2).sum(axis=0)

    cov = accum / M
    np.fill_diagonal(cov, np.diag(cov) + var_accum / M)
    return cov


def write_cov(model, run_dir):
    """
    Compute and save the posterior predictive covariance matrix for model
    ('normal' or 'tophat') using draws from output/<run>/<model>_?.csv and
    the MAIN sample from the FITS catalog (as defined by config.json cuts).

    Output: output/<run>/<model>_cov.png
    """
    config_json = os.path.join(run_dir, "config.json")
    with open(config_json) as f:
        cfg = json.load(f)

    fits_path = cfg["source"]
    xhat_star_full, sigma_x_star_full, yhat_star_full, sigma_y_star_full, _ = \
        load_xy_and_uncertainties_from_desi(fits_path, row=None, sort_by_zobs=True)

    main = _apply_main_cuts(cfg, xhat_star_full, yhat_star_full)
    xhat_star    = xhat_star_full[main]
    sigma_x_star = sigma_x_star_full[main]
    sigma_y_star = sigma_y_star_full[main]

    csv_pattern = os.path.join(run_dir, f"{model}_?.csv")
    if model == "normal":
        draws = read_cmdstan_posterior(
            csv_pattern,
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
            drop_diagnostics=True,
        )
        cov = ystar_pp_cov_normal_vectorized(draws, xhat_star, sigma_x_star)
    elif model == "tophat":
        draws = read_cmdstan_posterior(
            csv_pattern,
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
            drop_diagnostics=True,
        )
        input_json = os.path.join(run_dir, "input.json")
        cov = ystar_pp_cov_tophat_vectorized(
            draws, xhat_star, sigma_x_star,
            bounds_json=input_json,
        )
    else:
        raise ValueError(f"Unknown model {model!r}")

    # Build the subset index before modifying the diagonal so both sub plots
    # use the same galaxies.
    G = cov.shape[0]
    n_sub = min(512, G)
    rng = np.random.default_rng(0)
    idx = rng.choice(G, size=n_sub, replace=False)
    idx.sort()

    # Plot subset without observed magnitude uncertainty on the diagonal.
    cov_sub_noobs = cov[np.ix_(idx, idx)]
    run_name = os.path.basename(run_dir)
    out_path_sub_noobs = os.path.join(run_dir, f"{model}_cov_sub_noobs.png")
    plot_cov(cov_sub_noobs, out_path_sub_noobs)

    np.fill_diagonal(cov, np.diag(cov) + sigma_y_star**2)

    fits_out = os.path.join(run_dir, f"{model}_cov.fits")
    hdr = fits.Header()
    hdr["COMMENT"] = "Posterior predictive covariance matrix (float32)"
    hdr["COMMENT"] = f"Row/col order: MAIN=True rows of {model}_catalog.fits"
    hdr["MODEL"]   = model
    hdr["RUN"]     = os.path.basename(run_dir)
    fits.writeto(fits_out, cov.astype(np.float32), header=hdr, overwrite=True)
    print(f"Saved covariance FITS to {fits_out}")

    out_path = os.path.join(run_dir, f"{model}_cov.png")
    plot_cov(cov, out_path)

    cov_sub = cov[np.ix_(idx, idx)]
    out_path_sub = os.path.join(run_dir, f"{model}_cov_sub.png")
    plot_cov(cov_sub, out_path_sub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Posterior predictions and diagnostics.')
    parser.add_argument('--run', default=None,
                        help='Run name; reads/writes output/<run>/ with standard filenames')
    parser.add_argument('--model', default='tophat', choices=['tophat', 'normal'],
                        help='Model to use (default: tophat)')
    parser.add_argument('--source', default='DESI', choices=['DESI', 'ariel', 'fullmocks'],
                        help='Data source (default: DESI)')
    parser.add_argument('--n_objects', type=int, default=None,
                        help='Number of objects to use for prediction (default: all)')
    parser.add_argument('--dir', default='data',
                        help='Directory containing FITS files (used with --source fullmocks)')
    parser.add_argument('--predict_run', default=None,
                        help='Simulation ID for the FITS file to predict on (default: same as --run)')
    parser.add_argument('--delta_haty_min',         type=float, default=0.0,
                        help='Offset added to input.json haty_min for prediction selection (default: 0)')
    parser.add_argument('--delta_haty_max',         type=float, default=0.0,
                        help='Offset added to input.json haty_max for prediction selection (default: 0)')
    parser.add_argument('--delta_z_obs_min',        type=float, default=0.0,
                        help='Offset added to input.json z_obs_min for prediction selection (default: 0)')
    parser.add_argument('--delta_z_obs_max',        type=float, default=0.0,
                        help='Offset added to input.json z_obs_max for prediction selection (default: 0)')
    parser.add_argument('--plane_cut',              action='store_true', default=True,
                        help='Apply oblique plane cut from input.json during prediction (default: on)')
    parser.add_argument('--delta_intercept_plane',  type=float, default=0.0,
                        help='Offset added to input.json intercept_plane (default: 0)')
    parser.add_argument('--delta_intercept_plane2', type=float, default=0.0,
                        help='Offset added to input.json intercept_plane2 (default: 0)')
    parser.add_argument('--input', default=None,
                        help='Input FITS catalog path (required with --catalog)')
    parser.add_argument('--catalog', action='store_true',
                        help='Write augmented catalog FITS to output/<run>/<model>_catalog.fits')
    args = parser.parse_args()

    run_dir = os.path.join('output', args.run) if args.run else None

    if args.source == 'DESI':
        DESI(args.model, run_dir=run_dir)
        if args.catalog:
            write_desi_catalog(
                args.model,
                run_dir,
                args.input or 'data/SGA-2020_iron_Vrot_VI_corr.fits',
            )
    elif args.source == 'fullmocks':
        fits_id = args.predict_run if args.predict_run else args.run
        pattern = os.path.join(args.dir, f'TF_extended_AbacusSummit_base_{fits_id}_*.fits')
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No FITS file found matching: {pattern}")
        galaxy_fits = sorted(matches)[0]
        fullmocks(args.model, galaxy_fits=galaxy_fits, n_objects=args.n_objects, run_dir=run_dir,
                  delta_haty_min=args.delta_haty_min,
                  delta_haty_max=args.delta_haty_max,
                  delta_z_obs_min=args.delta_z_obs_min,
                  delta_z_obs_max=args.delta_z_obs_max,
                  plane_cut=args.plane_cut,
                  delta_intercept_plane=args.delta_intercept_plane,
                  delta_intercept_plane2=args.delta_intercept_plane2)
    else:
        ariel(args.model, run_dir=run_dir)

    write_cov(args.model, run_dir)
