#!/usr/bin/env python3
"""
desi_galaxy_magnitudes.py

Predict posterior predictive absolute magnitudes for DESI galaxies using a
calibrated Tully-Fisher (TF) model with a Normal population prior on the
latent on-relation magnitude:

    y_TF ~ Normal(mu_TF, tau)

For each posterior draw theta^(m) = (s, c, sigma_int_x, sigma_int_y) and
k inner Monte Carlo samples, the two-step composition is:

    y_TF^(m,k) ~ TruncNormal( c^(m) + s^(m)*x_hat_star,
                               |s^(m)| * sigma_1_star^(m);
                               [y_min, y_max] )

    y_star^(m,k) ~ Normal( y_TF^(m,k), sigma_int_y^(m) )

where
    sigma_1_star^2 = sigma_x_star^2 + sigma_int_x^(m)^2

Pooling {y_star^(m,k)} over all (m, k) gives the posterior predictive
distribution from which means and credible intervals are computed.

The rotation-velocity proxy x is defined as

    x = log10(V / V0),   V0 = 100 km/s  (default)

with propagated uncertainty

    sigma_x = sigma_V / (V * ln 10)

The FITS columns --v-col / --v-unc-col supply the raw velocity V (km/s) and
its uncertainty sigma_V; the log-transform is applied internally.

Inputs
------
- Galaxy catalog : FITS file  (default: data/DESI-DR1_TF_pv_cat_v15.fits)
- TF posterior draws : CSV files matching a glob pattern
      Required columns: slope, intercept.1, sigma_int_x, sigma_int_y
- TF input JSON : provides y_min, y_max (and optionally mu_y_TF, tau)

Outputs
-------
A CSV with per-galaxy posterior predictive summaries and credible intervals.

Dependencies
------------
numpy, pandas, scipy, astropy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats


# ---------------------------------------------------------------------------
# Posterior container + loading
# ---------------------------------------------------------------------------

@dataclass
class PosteriorDraws:
    s: np.ndarray           # slope
    c: np.ndarray           # intercept
    sigma_int_x: np.ndarray
    sigma_int_y: np.ndarray

    @property
    def n_draws(self) -> int:
        return len(self.s)


def _require_columns(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {where}. "
            f"Available: {list(df.columns)}"
        )


def load_posterior_draws(
    posterior_files: List[str],
    col_s: str = "slope",
    col_c: str = "intercept.1",
    col_sigma_int_x: str = "sigma_int_x",
    col_sigma_int_y: str = "sigma_int_y",
    max_draws: Optional[int] = None,
    seed: int = 0,
) -> PosteriorDraws:
    """
    Load and concatenate posterior draws from one or more CSV files.
    Optionally subsample to at most *max_draws* (uniform, without replacement).
    """
    dfs = [pd.read_csv(pf, comment="#") for pf in posterior_files]
    all_df = pd.concat(dfs, ignore_index=True)

    req = [col_s, col_c, col_sigma_int_x, col_sigma_int_y]
    _require_columns(all_df, req, where="posterior draws")

    s    = all_df[col_s].to_numpy(float)
    c    = all_df[col_c].to_numpy(float)
    sigx = all_df[col_sigma_int_x].to_numpy(float)
    sigy = all_df[col_sigma_int_y].to_numpy(float)

    n = len(s)
    if max_draws is not None and max_draws < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_draws, replace=False)
        s, c, sigx, sigy = s[idx], c[idx], sigx[idx], sigy[idx]

    return PosteriorDraws(s=s, c=c, sigma_int_x=sigx, sigma_int_y=sigy)


# ---------------------------------------------------------------------------
# Core MC composition
# ---------------------------------------------------------------------------

def _sample_ytf_truncnormal_vectorised(
    loc: np.ndarray,
    scale: np.ndarray,
    y_min: float,
    y_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one sample per element from TruncNormal(loc, scale; [y_min, y_max]).

    Uses the inverse-CDF method fully vectorised.  Elements where scale <= 0
    are clamped to clip(loc, y_min, y_max) (degenerate case, |s| ~ 0).

    Parameters
    ----------
    loc, scale : (M,) arrays
    y_min, y_max : scalar bounds

    Returns
    -------
    samples : (M,) array
    """
    out = np.empty_like(loc, dtype=float)
    good = scale > 0
    if np.any(good):
        lg, sg = loc[good], scale[good]
        a = (y_min - lg) / sg
        b = (y_max - lg) / sg
        p_lo = stats.norm.cdf(a)
        p_hi = stats.norm.cdf(b)
        # guard against zero-width intervals (loc far outside [y_min, y_max])
        width = p_hi - p_lo
        width = np.where(width < 1e-300, 1e-300, width)
        u = rng.uniform(size=int(good.sum()))
        z = stats.norm.ppf(np.clip(p_lo + u * width, 1e-300, 1.0 - 1e-300))
        out[good] = lg + sg * z
    if np.any(~good):
        out[~good] = np.clip(loc[~good], y_min, y_max)
    return out


def infer_magnitude_single_galaxy(
    x_hat_star: float,
    sigma_x_star: float,
    posterior: PosteriorDraws,
    y_min: float,
    y_max: float,
    n_inner: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw posterior predictive samples for y_star for a single galaxy.

    Implements the two-step Monte Carlo composition:

        sigma_1^2 = sigma_x_star^2 + sigma_int_x^(m)^2
        y_TF^(m,k) ~ TruncNormal( c^(m) + s^(m)*x_hat_star,
                                   |s^(m)| * sigma_1^(m);
                                   [y_min, y_max] )
        y_star^(m,k) ~ Normal( y_TF^(m,k), sigma_int_y^(m) )

    Parameters
    ----------
    x_hat_star : float
        Observed log-ratio rotation-velocity proxy  x = log10(V/V0).
    sigma_x_star : float
        Measurement uncertainty on x_hat_star (>= 0).
    posterior : PosteriorDraws
        Posterior draws (M draws).
    y_min, y_max : float
        Truncation bounds for y_TF.
    n_inner : int
        Number of inner MC samples k per posterior draw m.
        Total samples returned = M * n_inner.
    rng : numpy Generator, optional

    Returns
    -------
    y_star_samples : (M * n_inner,) array
        Posterior predictive samples for y_star.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not np.isfinite(x_hat_star) or not np.isfinite(sigma_x_star):
        return np.array([], dtype=float)
    if sigma_x_star < 0:
        raise ValueError("sigma_x_star must be >= 0.")

    M = posterior.n_draws
    if M == 0:
        raise ValueError("No posterior draws provided.")

    s    = posterior.s            # (M,)
    c    = posterior.c            # (M,)
    sigx = posterior.sigma_int_x  # (M,)
    sigy = posterior.sigma_int_y  # (M,)

    # sigma_1^2 = sigma_x^2 + sigma_int_x^2
    sigma1 = np.sqrt(sigma_x_star**2 + sigx**2)   # (M,)

    # Likelihood centre and scale for y_TF
    loc_ytf   = c + s * x_hat_star                 # (M,)
    scale_ytf = np.abs(s) * sigma1                 # (M,)

    if n_inner == 1:
        # One sample per draw — no tiling needed
        y_tf   = _sample_ytf_truncnormal_vectorised(
            loc_ytf, scale_ytf, y_min, y_max, rng
        )                                           # (M,)
        y_star = rng.normal(loc=y_tf, scale=sigy)  # (M,)
        return y_star

    # n_inner > 1: tile each draw n_inner times
    loc_t   = np.repeat(loc_ytf,   n_inner)        # (M*n_inner,)
    scale_t = np.repeat(scale_ytf, n_inner)        # (M*n_inner,)
    sigy_t  = np.repeat(sigy,      n_inner)        # (M*n_inner,)

    y_tf   = _sample_ytf_truncnormal_vectorised(
        loc_t, scale_t, y_min, y_max, rng
    )
    y_star = rng.normal(loc=y_tf, scale=sigy_t)
    return y_star


# ---------------------------------------------------------------------------
# Credible intervals
# ---------------------------------------------------------------------------

def compute_credible_interval(
    samples: np.ndarray,
    credibility: float = 0.68,
) -> Tuple[float, float]:
    """Central credible interval from samples."""
    if samples.size == 0:
        return (np.nan, np.nan)
    alpha = (1.0 - credibility) / 2.0
    lo = float(np.percentile(samples, 100.0 * alpha))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha)))
    return lo, hi


# ---------------------------------------------------------------------------
# FITS I/O helpers
# ---------------------------------------------------------------------------

def load_fits_table(fits_file: str, hdu: int = 1) -> np.recarray:
    """Load a FITS binary table HDU as a numpy recarray."""
    fits_path = Path(fits_file)
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_file}")
    with fits.open(fits_file, memmap=True) as hdul:
        if hdu >= len(hdul):
            raise ValueError(
                f"Requested HDU={hdu}, but file has {len(hdul)} HDUs."
            )
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"HDU {hdu} contains no table data.")
        return data


def get_column_or_nan(
    data: np.recarray, name: str, length: int
) -> np.ndarray:
    """Return data[name] if present, else an array of NaNs."""
    if name in data.names:
        return np.asarray(data[name])
    return np.full(length, np.nan, dtype=float)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_galaxies(
    fits_file: str,
    posterior_files: List[str],
    output_file: str,
    v_col: str,
    v_unc_col: str,
    y_min: float,
    y_max: float,
    v0: float = 100.0,
    y_col: Optional[str] = None,
    y_unc_col: Optional[str] = None,
    z_col: Optional[str] = None,
    fits_hdu: int = 1,
    n_inner: int = 1,
    credibility_levels: List[float] = (0.68, 0.95),
    max_galaxies: Optional[int] = None,
    max_posterior_draws: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Process all galaxies and write posterior predictive summaries to CSV.

    Parameters
    ----------
    fits_file : str
        Path to DESI FITS galaxy catalog.
    posterior_files : list of str
        Paths to TF posterior draw CSV files.
    output_file : str
        Path for output CSV.
    v_col, v_unc_col : str
        FITS column names for the raw rotation velocity V (km/s) and its
        measurement uncertainty sigma_V.  The log-ratio proxy is computed as
            x       = log10(V / v0)
            sigma_x = sigma_V / (V * ln 10)
    y_min, y_max : float
        Truncation bounds for y_TF (from DESI_TF_input.json).
    v0 : float
        Reference velocity for the log-ratio (default: 100 km/s).
    y_col, y_unc_col : str, optional
        FITS column names for observed magnitude and its uncertainty
        (carried through to output for comparison).
    z_col : str, optional
        FITS column name for observed redshift.
    fits_hdu : int
        HDU index of the binary table.
    n_inner : int
        Inner MC samples per posterior draw.
    credibility_levels : list of float
        Credibility levels for posterior predictive intervals.
    max_galaxies : int, optional
        Cap on number of galaxies processed (for testing).
    max_posterior_draws : int, optional
        Subsample posterior draws to this many (for speed).
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)

    posterior = load_posterior_draws(
        posterior_files=posterior_files,
        max_draws=max_posterior_draws,
        seed=seed,
    )
    print(f"Loaded {posterior.n_draws} posterior draws.")
    print(f"y_TF truncation bounds: [{y_min:.4f}, {y_max:.4f}]")
    print(f"Reference velocity V0 = {v0} km/s")

    data = load_fits_table(fits_file, hdu=fits_hdu)
    n_all = len(data)
    n_use = n_all if max_galaxies is None else min(max_galaxies, n_all)
    print(f"FITS table: {n_all} rows; processing {n_use}.")

    if v_col not in data.names or v_unc_col not in data.names:
        raise ValueError(
            f"Required column(s) missing.\n"
            f"Need v_col='{v_col}' and v_unc_col='{v_unc_col}'.\n"
            f"Available: {list(data.names)}"
        )

    V_raw     = np.asarray(data[v_col],     dtype=float)[:n_use]
    V_unc_raw = np.asarray(data[v_unc_col], dtype=float)[:n_use]

    # Convert raw velocity → log-ratio proxy and propagate uncertainty
    #   x       = log10(V / V0)
    #   sigma_x = sigma_V / (V * ln10)
    with np.errstate(divide="ignore", invalid="ignore"):
        x     = np.where(V_raw > 0, np.log10(V_raw / v0), np.nan)
        x_unc = np.where(V_raw > 0, V_unc_raw / (V_raw * np.log(10.0)), np.nan)

    y     = None if y_col     is None else get_column_or_nan(data, y_col,     n_all)[:n_use]
    y_unc = None if y_unc_col is None else get_column_or_nan(data, y_unc_col, n_all)[:n_use]
    z     = None if z_col     is None else get_column_or_nan(data, z_col,     n_all)[:n_use]

    results: List[Dict] = []
    for i in range(n_use):
        if (i + 1) % 5000 == 0:
            print(f"  processed {i+1}/{n_use}")

        x_hat_star   = float(x[i])
        sigma_x_star = float(x_unc[i])

        samples = infer_magnitude_single_galaxy(
            x_hat_star=x_hat_star,
            sigma_x_star=sigma_x_star,
            posterior=posterior,
            y_min=y_min,
            y_max=y_max,
            n_inner=n_inner,
            rng=rng,
        )

        n_samp = int(samples.size)
        row: Dict = {
            "galaxy_id":      i,
            "V_obs":          float(V_raw[i]),
            "V_unc":          float(V_unc_raw[i]),
            "x_obs":          x_hat_star,
            "x_unc":          sigma_x_star,
            "y_pred_mean":    float(np.mean(samples))   if n_samp else np.nan,
            "y_pred_median":  float(np.median(samples)) if n_samp else np.nan,
            "y_pred_std":     float(np.std(samples))    if n_samp else np.nan,
            "n_pred_samples": n_samp,
        }

        if y     is not None: row["y_obs"]  = float(y[i])
        if y_unc is not None: row["y_unc"]  = float(y_unc[i])
        if z     is not None: row["zobs"]   = float(z[i])

        for cred in credibility_levels:
            lo, hi = compute_credible_interval(samples, credibility=cred)
            pct = int(round(cred * 100))
            row[f"y_pred_CI{pct}_lower"] = lo
            row[f"y_pred_CI{pct}_upper"] = hi

        results.append(row)

    df_out = pd.DataFrame(results)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {output_file}")
    return df_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict DESI galaxy absolute magnitudes from rotation velocities "
            "using a TF calibration with a Normal population prior on y_TF. "
            "Posterior predictive samples are drawn via two-step MC composition: "
            "y_TF ~ TruncNormal(c+s*x_hat, |s|*sigma_1; [y_min,y_max]), "
            "y_star ~ Normal(y_TF, sigma_int_y), "
            "where x_hat = log10(V/V0) is computed internally from the raw velocity."
        )
    )

    # --- inputs ---
    parser.add_argument(
        "--fits-file", default="data/DESI-DR1_TF_pv_cat_v15.fits",
        help="Path to DESI FITS galaxy catalog.",
    )
    parser.add_argument(
        "--fits-hdu", type=int, default=1,
        help="FITS HDU index of the binary table (default: 1).",
    )
    parser.add_argument(
        "--posterior-pattern", default="DESI_TF_?.csv",
        help="Glob pattern for TF posterior draw CSV files.",
    )
    parser.add_argument(
        "--tf-input-json", default="DESI_TF_input.json",
        help=(
            "Path to TF input JSON file. "
            "Reads y_min, y_max (required) and optionally mu_y_TF, tau."
        ),
    )
    parser.add_argument(
        "--output", default="desi_galaxy_magnitude_predictions.csv",
        help="Output CSV file.",
    )

    # --- FITS column names ---
    parser.add_argument(
        "--v-col", default="V_0p4R26",
        help="FITS column for raw rotation velocity V (km/s). "
             "Converted internally to x = log10(V/V0).",
    )
    parser.add_argument(
        "--v-unc-col", default="V_0p4R26_ERR",
        help="FITS column for uncertainty in V (km/s). "
             "Converted internally to sigma_x = sigma_V / (V * ln10).",
    )
    parser.add_argument(
        "--v0", type=float, default=100.0,
        help="Reference velocity V0 for log-ratio x = log10(V/V0) (default: 100 km/s).",
    )
    parser.add_argument(
        "--y-col", default="R_ABSMAG_SB26",
        help="(Optional) FITS column for observed absolute magnitude.",
    )
    parser.add_argument(
        "--y-unc-col", default="R_ABSMAG_SB26_ERR",
        help="(Optional) FITS column for magnitude uncertainty.",
    )
    parser.add_argument(
        "--z-col", default="Z_DESI_CMB",
        help="(Optional) FITS column for observed redshift.",
    )

    # --- truncation bounds (override JSON) ---
    parser.add_argument(
        "--y-min", type=float, default=None,
        help="Lower truncation bound for y_TF (overrides JSON key 'y_min').",
    )
    parser.add_argument(
        "--y-max", type=float, default=None,
        help="Upper truncation bound for y_TF (overrides JSON key 'y_max').",
    )

    # --- MC controls ---
    parser.add_argument(
        "--n-inner", type=int, default=1,
        help="Inner MC samples per posterior draw (default: 1).",
    )
    parser.add_argument(
        "--credibility", type=float, nargs="+", default=[0.68, 0.95],
        help="Credible interval levels (default: 0.68 0.95).",
    )
    parser.add_argument(
        "--max-galaxies", type=int, default=None,
        help="Process only the first N galaxies (for testing).",
    )
    parser.add_argument(
        "--max-posterior-draws", type=int, default=None,
        help="Subsample posterior draws to at most this many.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducibility.",
    )

    args = parser.parse_args()

    # --- locate posterior files ---
    posterior_files = sorted(glob(args.posterior_pattern))
    if not posterior_files:
        raise FileNotFoundError(
            f"No posterior files found matching: {args.posterior_pattern}"
        )
    print(f"Found {len(posterior_files)} posterior file(s): {posterior_files}")

    # --- load y_min / y_max from JSON ---
    y_min_val = args.y_min
    y_max_val = args.y_max
    if args.tf_input_json is not None:
        json_path = Path(args.tf_input_json)
        if json_path.exists():
            with open(json_path) as f:
                tf_json = json.load(f)
            if y_min_val is None and "y_min" in tf_json:
                y_min_val = float(tf_json["y_min"])
                print(f"Loaded y_min = {y_min_val:.6f} from {args.tf_input_json}")
            if y_max_val is None and "y_max" in tf_json:
                y_max_val = float(tf_json["y_max"])
                print(f"Loaded y_max = {y_max_val:.6f} from {args.tf_input_json}")
        else:
            print(f"Warning: JSON file not found: {args.tf_input_json}")

    if y_min_val is None or y_max_val is None:
        raise ValueError(
            "y_min and y_max are required. "
            "Provide them via --tf-input-json (keys 'y_min'/'y_max') "
            "or --y-min / --y-max."
        )

    # --- treat 'none'/'null'/'' as disabling optional columns ---
    def _noneify(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s2 = str(s).strip()
        return None if s2.lower() in {"none", "null", ""} else s2

    df = process_galaxies(
        fits_file=args.fits_file,
        fits_hdu=args.fits_hdu,
        posterior_files=posterior_files,
        output_file=args.output,
        v_col=args.v_col,
        v_unc_col=args.v_unc_col,
        v0=args.v0,
        y_col=_noneify(args.y_col),
        y_unc_col=_noneify(args.y_unc_col),
        z_col=_noneify(args.z_col),
        y_min=y_min_val,
        y_max=y_max_val,
        n_inner=args.n_inner,
        credibility_levels=list(args.credibility),
        max_galaxies=args.max_galaxies,
        max_posterior_draws=args.max_posterior_draws,
        seed=args.seed,
    )

    print("\nSummary:")
    print(f"  N galaxies       : {len(df)}")
    print(f"  mean(y_pred_mean): {df['y_pred_mean'].mean():.4f}")
    print(f"  mean(y_pred_std) : {df['y_pred_std'].mean():.4f}")


if __name__ == "__main__":
    main()
