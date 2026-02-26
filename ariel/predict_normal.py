import numpy as np
import glob
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt 
from astropy.io import fits
import json

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
        Filename or glob pattern, e.g. "MOCK_normal_*.csv" or "MOCK_normal_?.csv".
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

def MOCK_main():
    draws = read_cmdstan_posterior(
        "MOCK_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )

    galaxy_csv = "data/TF_mock_tophat-mag_input.csv"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_csv(
        galaxy_csv, row=None, sort_by_zobs=True
    )

    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)

    # Your plot uses (predicted mean - observed yhat)
    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.01)
    plt.show()

def DESI_main():
    draws = read_cmdstan_posterior(
        "DESI_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )

    galaxy_fits = "data/DESI-DR1_TF_pv_cat_v15.fits"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_desi(
        galaxy_fits, row=None, sort_by_zobs=False
    )

    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)

    # Your plot uses (predicted mean - observed yhat)
    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred

    plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1)

    galaxy_json = "DESI_input.json"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_stan_json(
        galaxy_json)
    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat_star, sigma_x_star)
    mean_y = mean_pred - yhat_star
    sigma_y = sd_pred
    plt.errorbar(zobs_star, mean_y, yerr=sigma_y, fmt="o", alpha=0.1)

    plt.xscale("log")
    plt.xlabel(r"$z_{\text{obs}}$")
    plt.ylabel(r"$\mathbb{E}[y_* | \hat x_*, \sigma_x^*] - y_{\text{obs}}$ (mag)")
    plt.show()

if __name__ == "__main__":
    DESI_main()
    # MOCK_main()