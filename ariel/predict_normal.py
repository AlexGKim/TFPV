import numpy as np
import glob
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt 

import numpy as np

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


def main():
    draws = read_cmdstan_posterior(
        "MOCK_normal_?.csv",
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
        )
    galaxy_csv = "data/TF_mock_tophat-mag_input.csv"
    xhat_star, sigma_x_star, yhat_star, sigma_y_star, zobs_star = load_xy_and_uncertainties_from_csv(galaxy_csv, row=None, sort_by_zobs=True)



    N=100
    ans=[]
    for _, draw in draws.iterrows():   # draw is a pandas Series
        s = draw.slope
        c = draw["intercept.1"]
        sigma_int_x = draw.sigma_int_x
        sigma_int_y = draw.sigma_int_y
        mu_y_TF = draw.mu_y_TF
        tau = draw.tau
        ans.append(draw_ystar_posterior_predictive_normal(
            N,
            xhat_star[0],
            sigma_x_star[0],
            s,
            c,
            sigma_int_x,
            mu_y_TF,
            tau,
            sigma_int_y,
            rng=None,))
        
    ans = np.concatenate(ans)
    plt.hist(ans, bins=30, density=True)
    plt.show()

if __name__ == "__main__":
    main()