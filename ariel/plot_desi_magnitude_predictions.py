"""
plot_desi_magnitude_predictions.py

Plot observed absolute magnitude (from DESI FITS) versus predicted absolute magnitude
with confidence regions from desi_normal_predictions.csv (output of
magnitude_prediction_normal.py).

Expected CSV columns (written by magnitude_prediction_normal.py --fits-file):
    x_obs          log10(V/V0) proxy
    y_obs          observed R_ABSMAG_SB26
    y_unc          uncertainty on y_obs
    z_obs          observed redshift (optional)
    p<N>           percentile columns, e.g. p2.5, p16, p50, p84, p97.5
    y_pred_median  posterior predictive median (always present)

x-axis: y_obs (R_ABSMAG_SB26) with error bar y_unc
y-axis: p50 (or y_pred_median) with error bars from the chosen CI percentile pair

In-sample and out-of-sample objects are shown in different colours.
The sample cuts mirror those applied in desi_data.py:
  1. y_obs < haty_max
  2. (optional two-sided plane cut)
       slope_plane * x_obs + intercept_plane  <=  y_obs
       y_obs  <=  min(haty_max, slope_plane * x_obs + intercept_plane2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# ---------------------------------------------------------------------------
# Sample-cut helper
# ---------------------------------------------------------------------------

def compute_insample_mask(
    x: np.ndarray,
    y: np.ndarray,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
) -> np.ndarray:
    """
    Return a boolean mask that is True for objects passing the same selection
    cuts used in desi_data.py.

    Parameters
    ----------
    x : array  log10(V/V0) proxy
    y : array  observed absolute magnitude
    haty_max : float
        Upper magnitude limit (objects must have y < haty_max).
    slope_plane, intercept_plane, intercept_plane2 : float
        Two-sided parallel plane-cut parameters:
            slope_plane * x + intercept_plane  <=  y
            y  <=  min(haty_max, slope_plane * x + intercept_plane2)
    """
    lower_bound   = slope_plane * x + intercept_plane
    upper_oblique = slope_plane * x + intercept_plane2
    upper_bound   = np.minimum(haty_max, upper_oblique)

    mask = (
        (y < haty_max) &
        (lower_bound <= y) &
        (y <= upper_bound)
    )
    return mask


# ---------------------------------------------------------------------------
# Column-resolution helpers
# ---------------------------------------------------------------------------

def _resolve_median_col(df: pd.DataFrame, median_pct: float = 50.0) -> str:
    """Return the column name for the predictive median."""
    pct_col = f"p{median_pct:g}"
    if pct_col in df.columns:
        return pct_col
    if "y_pred_median" in df.columns:
        return "y_pred_median"
    raise KeyError(
        f"Cannot find median column '{pct_col}' or 'y_pred_median' in CSV. "
        f"Available: {list(df.columns)}"
    )


def _resolve_ci_cols(
    df: pd.DataFrame,
    lo_pct: float,
    hi_pct: float,
) -> tuple[str, str]:
    """Return (lower_col, upper_col) for the chosen CI percentile pair."""
    lo_col = f"p{lo_pct:g}"
    hi_col = f"p{hi_pct:g}"
    missing = [c for c in [lo_col, hi_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"CI columns {missing} not found in CSV. "
            f"Available percentile columns: "
            f"{[c for c in df.columns if c.startswith('p')]}"
        )
    return lo_col, hi_col


# ---------------------------------------------------------------------------
# Shared data-loading helper
# ---------------------------------------------------------------------------

def _load_and_split(
    csv_file: str,
    max_galaxies,
    haty_max: float,
    slope_plane: float,
    intercept_plane: float,
    intercept_plane2: float,
):
    """
    Load predictions CSV, optionally cap rows, compute in/out-of-sample split.

    Returns
    -------
    df_in, df_out : DataFrames for in-sample and out-of-sample objects
    """
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["y_obs"])

    if max_galaxies is not None and max_galaxies < len(df):
        print(f"Using only first {max_galaxies} galaxies (out of {len(df)} total)")
        df = df.iloc[:max_galaxies]

    mask = compute_insample_mask(
        df["x_obs"].values,
        df["y_obs"].values,
        haty_max=haty_max,
        slope_plane=slope_plane,
        intercept_plane=intercept_plane,
        intercept_plane2=intercept_plane2,
    )
    df_in  = df[mask].copy()
    df_out = df[~mask].copy()
    print(f"  In-sample : {len(df_in):5d}  |  Out-of-sample: {len(df_out):5d}")
    return df_in, df_out


def _errorbar_kwargs(color, label):
    return dict(
        fmt="o", color=color, alpha=0.2, markersize=3,
        elinewidth=0.5, capsize=0, label=label,
    )


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_desi_magnitude_comparison(
    csv_file,
    output_file="desi_magnitude_comparison.png",
    max_galaxies=None,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
    ci_lo_pct: float = 16.0,
    ci_hi_pct: float = 84.0,
):
    """
    Plot observed R_ABSMAG_SB26 (x-axis) vs predicted magnitude (y-axis).

    In-sample objects (steelblue) and out-of-sample objects (tomato) are
    shown with alpha=0.2 per point.  Error bars on y use the percentile pair
    (ci_lo_pct, ci_hi_pct) from the predictions CSV.
    """
    print(f"Loading data from {csv_file}...")
    df_in, df_out = _load_and_split(
        csv_file, max_galaxies,
        haty_max, slope_plane, intercept_plane, intercept_plane2,
    )

    all_df = pd.concat([df_in, df_out])
    median_col = _resolve_median_col(all_df)
    ci_lo_col, ci_hi_col = _resolve_ci_cols(all_df, ci_lo_pct, ci_hi_pct)
    ci_label = f"{int(round(ci_hi_pct - ci_lo_pct))}% CI"
    print(f"  Median column : {median_col}")
    print(f"  CI columns    : [{ci_lo_col}, {ci_hi_col}]  ({ci_label})")

    fig, ax = plt.subplots(figsize=(10, 10))

    for df_sub, color, label in [
        (df_in,  "steelblue", f"In-sample (N={len(df_in)})"),
        (df_out, "tomato",    f"Out-of-sample (N={len(df_out)})"),
    ]:
        if len(df_sub) == 0:
            continue
        obs   = df_sub["y_obs"].values
        unc   = df_sub["y_unc"].values if "y_unc" in df_sub.columns else np.zeros(len(df_sub))
        pred  = df_sub[median_col].values
        ci_lo = df_sub[ci_lo_col].values
        ci_hi = df_sub[ci_hi_col].values
        ax.errorbar(
            obs, pred,
            xerr=unc,
            yerr=[pred - ci_lo, ci_hi - pred],
            **_errorbar_kwargs(color, label),
        )

    # 1:1 line
    all_obs  = all_df["y_obs"].values
    all_pred = all_df[median_col].values
    lo = min(np.nanmin(all_obs), np.nanmin(all_pred)) - 0.5
    hi = max(np.nanmax(all_obs), np.nanmax(all_pred)) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="1:1 line", zorder=10)

    ax.set_xlabel(r"Observed $M_{\rm abs}$ (R_ABSMAG_SB26)", fontsize=14)
    ax.set_ylabel(r"Predicted $M_{\rm abs}$ (median)", fontsize=14)
    ax.set_title(
        "DESI: Observed vs TF-Predicted Absolute Magnitudes\n"
        f"(x error: obs. unc.; y error: {ci_label})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_xaxis()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")

    # ---- statistics (in-sample only) ----
    if len(df_in):
        obs   = df_in["y_obs"].values
        pred  = df_in[median_col].values
        ci_lo = df_in[ci_lo_col].values
        ci_hi = df_in[ci_hi_col].values
        unc   = df_in["y_unc"].values if "y_unc" in df_in.columns else np.zeros(len(df_in))
        residuals = obs - pred
        within_ci = ((obs >= ci_lo) & (obs <= ci_hi)).sum()
        n = len(df_in)
        print(f"\nStatistics — in-sample ({n} galaxies):")
        print(f"  Mean observed M_abs : {obs.mean():.3f} ± {obs.std():.3f}")
        print(f"  Mean predicted M_abs: {pred.mean():.3f} ± {pred.std():.3f}")
        print(f"  Mean {ci_label} width   : {(ci_hi - ci_lo).mean():.3f}")
        print(f"  Mean obs. unc.      : {unc.mean():.3f}")
        print(f"  Mean residual       : {residuals.mean():.3f} ± {residuals.std():.3f}")
        print(f"  RMS residual        : {np.sqrt(np.mean(residuals**2)):.3f}")
        print(f"  Fraction within {ci_label}: {within_ci/n:.3f} ({within_ci}/{n})")


# ---------------------------------------------------------------------------
# Residuals plot
# ---------------------------------------------------------------------------

def plot_desi_residuals(
    csv_file,
    output_file="desi_magnitude_residuals.png",
    max_galaxies=None,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
    ci_lo_pct: float = 16.0,
    ci_hi_pct: float = 84.0,
):
    """
    Plot residuals (y_obs - median) versus observed magnitude.

    In-sample objects (steelblue) and out-of-sample objects (tomato) are
    shown with alpha=0.2 per point.
    """
    print(f"\nLoading data from {csv_file}...")
    df_in, df_out = _load_and_split(
        csv_file, max_galaxies,
        haty_max, slope_plane, intercept_plane, intercept_plane2,
    )

    all_df = pd.concat([df_in, df_out])
    median_col = _resolve_median_col(all_df)
    ci_lo_col, ci_hi_col = _resolve_ci_cols(all_df, ci_lo_pct, ci_hi_pct)
    ci_label = f"{int(round(ci_hi_pct - ci_lo_pct))}% CI"

    fig, ax = plt.subplots(figsize=(12, 6))

    for df_sub, color, label in [
        (df_in,  "steelblue", f"In-sample (N={len(df_in)})"),
        (df_out, "tomato",    f"Out-of-sample (N={len(df_out)})"),
    ]:
        if len(df_sub) == 0:
            continue
        obs   = df_sub["y_obs"].values
        unc   = df_sub["y_unc"].values if "y_unc" in df_sub.columns else np.zeros(len(df_sub))
        pred  = df_sub[median_col].values
        ci_lo = df_sub[ci_lo_col].values
        ci_hi = df_sub[ci_hi_col].values
        resid = obs - pred
        ax.errorbar(
            obs, resid,
            xerr=unc,
            yerr=[pred - ci_lo, ci_hi - pred],
            **_errorbar_kwargs(color, label),
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero residual")

    ax.set_xlabel(r"Observed $M_{\rm abs}$ (R_ABSMAG_SB26)", fontsize=14)
    ax.set_ylabel(r"Residual (Observed $-$ Predicted)", fontsize=14)
    ax.set_title(
        "DESI: Residuals vs Observed Magnitude\n"
        f"(x error: obs. unc.; y error: {ci_label})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # brighter magnitudes on right

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Residuals plot saved to: {output_file}")

    # ---- statistics (in-sample only) ----
    if len(df_in):
        resid = df_in["y_obs"].values - df_in[median_col].values
        n = len(df_in)
        print(f"\nResidual Statistics — in-sample ({n} galaxies):")
        print(f"  Mean residual : {resid.mean():.3f}")
        print(f"  Std residual  : {resid.std():.3f}")
        print(f"  RMS residual  : {np.sqrt(np.mean(resid**2)):.3f}")


# ---------------------------------------------------------------------------
# Residuals vs redshift plot
# ---------------------------------------------------------------------------

def plot_desi_residuals_vs_redshift(
    csv_file,
    output_file="desi_magnitude_residuals_vs_z.png",
    max_galaxies=None,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
    ci_lo_pct: float = 16.0,
    ci_hi_pct: float = 84.0,
):
    """
    Plot residuals (y_obs - median) versus redshift (z_obs).

    In-sample objects (steelblue) and out-of-sample objects (tomato) are
    shown with alpha=0.2 per point.  Requires the 'z_obs' column to be
    present in the predictions CSV (written when --z-col is set in
    magnitude_prediction_normal.py).
    """
    print(f"\nLoading data from {csv_file}...")
    df_in, df_out = _load_and_split(
        csv_file, max_galaxies,
        haty_max, slope_plane, intercept_plane, intercept_plane2,
    )

    all_df = pd.concat([df_in, df_out])

    # Accept both 'z_obs' (new) and 'zobs' (legacy)
    z_col = None
    for candidate in ["z_obs", "zobs"]:
        if candidate in all_df.columns:
            z_col = candidate
            break
    if z_col is None:
        raise KeyError(
            "'z_obs' column not found in predictions CSV. "
            "Re-run magnitude_prediction_normal.py with --z-col Z_DESI_CMB."
        )

    median_col = _resolve_median_col(all_df)
    ci_lo_col, ci_hi_col = _resolve_ci_cols(all_df, ci_lo_pct, ci_hi_pct)
    ci_label = f"{int(round(ci_hi_pct - ci_lo_pct))}% CI"

    fig, ax = plt.subplots(figsize=(12, 6))

    for df_sub, color, label in [
        (df_in,  "steelblue", f"In-sample (N={len(df_in)})"),
        (df_out, "tomato",    f"Out-of-sample (N={len(df_out)})"),
    ]:
        if len(df_sub) == 0:
            continue
        z     = df_sub[z_col].values
        pred  = df_sub[median_col].values
        ci_lo = df_sub[ci_lo_col].values
        ci_hi = df_sub[ci_hi_col].values
        resid = df_sub["y_obs"].values - pred
        ax.errorbar(
            z, resid,
            yerr=[pred - ci_lo, ci_hi - pred],
            **_errorbar_kwargs(color, label),
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.5, label="Zero residual")

    ax.set_xlabel(r"Redshift $z_{\rm obs}$", fontsize=14)
    ax.set_ylabel(r"Residual (Observed $-$ Predicted)", fontsize=14)
    ax.set_title(
        "DESI: Residuals vs Redshift\n"
        f"(y error: {ci_label})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Residuals-vs-redshift plot saved to: {output_file}")

    # ---- statistics (in-sample only) ----
    if len(df_in):
        resid = df_in["y_obs"].values - df_in[median_col].values
        z     = df_in[z_col].values
        n     = len(df_in)
        print(f"\nResidual vs z Statistics — in-sample ({n} galaxies):")
        print(f"  Redshift range : [{np.nanmin(z):.4f}, {np.nanmax(z):.4f}]")
        print(f"  Mean residual  : {resid.mean():.3f}")
        print(f"  Std residual   : {resid.std():.3f}")
        print(f"  RMS residual   : {np.sqrt(np.mean(resid**2)):.3f}")


# ---------------------------------------------------------------------------
# Covariance image plot
# ---------------------------------------------------------------------------

def plot_covariance_matrix(
    cov_file: str,
    csv_file: str | None = None,
    galaxy_ids_file: str | None = None,
    output_file: str = "desi_cov_matrix.png",
    sort_by: str = "x_obs",
    max_display: int = 1000,
    show_correlation: bool = True,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
) -> None:
    """
    Plot the posterior predictive covariance (or correlation) matrix as an image.

    For large matrices (N > max_display) the matrix is downsampled by averaging
    non-overlapping blocks so the image fits in max_display × max_display pixels.

    Parameters
    ----------
    cov_file : str
        Path to the .npy covariance matrix (output of magnitude_prediction_normal.py
        --save-cov).
    csv_file : str or None
        Path to the predictions CSV.  If provided and sort_by is set, galaxies are
        reordered by the chosen column before plotting.
    galaxy_ids_file : str or None
        Path to the companion galaxy-id index .npy file (FILE_galaxy_ids.npy).
        Required when csv_file is provided for sorting.
    output_file : str
        Output image file path.
    sort_by : str
        CSV column to sort galaxies by before plotting (default: 'x_obs').
        Set to '' or None to keep original order.
    max_display : int
        Maximum image resolution in pixels per axis.  The matrix is block-averaged
        down to at most max_display × max_display (default: 1000).
    show_correlation : bool
        If True (default), normalise to a correlation matrix before plotting.
        If False, plot the raw covariance.
    cmap : str
        Matplotlib colormap (default: 'RdBu_r').
    vmax : float or None
        Colour scale maximum.  If None, uses the 99th percentile of |values|.
    """
    print(f"Loading covariance matrix from {cov_file} ...")
    cov = np.load(cov_file).astype(np.float64)
    N = cov.shape[0]
    print(f"  Shape: {cov.shape}  dtype: {np.load(cov_file).dtype}")

    # ---- optional sort by a CSV column ----
    sort_order = np.arange(N)
    sort_label = "original order"
    if sort_by and csv_file is not None:
        if galaxy_ids_file is None:
            # Try default companion filename
            galaxy_ids_file = str(cov_file).replace(".npy", "_galaxy_ids.npy")
        try:
            gids = np.load(galaxy_ids_file).astype(int)
            df   = pd.read_csv(csv_file)
            if sort_by in df.columns:
                # Map galaxy_id → sort key
                id_to_key = dict(zip(df["galaxy_id"].values, df[sort_by].values))
                keys = np.array([id_to_key.get(gid, np.nan) for gid in gids])
                sort_order = np.argsort(keys)
                sort_label = f"sorted by {sort_by}"
                print(f"  Sorted {N} galaxies by '{sort_by}'")
            else:
                print(f"  Warning: column '{sort_by}' not found in CSV; using original order.")
        except FileNotFoundError as e:
            print(f"  Warning: {e}; using original order.")

    cov = cov[np.ix_(sort_order, sort_order)]

    # ---- convert to correlation if requested ----
    if show_correlation:
        std = np.sqrt(np.diag(cov))
        # Guard against zero-variance entries
        std = np.where(std > 0, std, 1.0)
        mat = cov / np.outer(std, std)
        np.clip(mat, -1.0, 1.0, out=mat)
        label = "Correlation"
        default_vmax = 1.0
    else:
        mat = cov
        label = r"Covariance (mag$^2$)"
        default_vmax = None

    if vmax is None:
        if show_correlation:
            vmax = 1.0
        else:
            vmax = float(np.percentile(np.abs(mat), 99))

    # ---- downsample to max_display × max_display ----
    if N > max_display:
        block = int(np.ceil(N / max_display))
        # Trim to exact multiple
        N_trim = (N // block) * block
        mat_trim = mat[:N_trim, :N_trim]
        # Block-average
        mat_ds = mat_trim.reshape(N_trim // block, block, N_trim // block, block).mean(axis=(1, 3))
        n_px = mat_ds.shape[0]
        print(f"  Downsampled {N}×{N} → {n_px}×{n_px} (block size {block})")
    else:
        mat_ds = mat
        n_px = N

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        mat_ds,
        origin="upper",
        aspect="equal",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=12)

    ax.set_title(
        f"Posterior Predictive {label} Matrix\n"
        f"N={N} galaxies ({sort_label})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Galaxy index", fontsize=11)
    ax.set_ylabel("Galaxy index", fontsize=11)

    # Tick labels in original galaxy units (approximate)
    n_ticks = 6
    tick_px  = np.linspace(0, n_px - 1, n_ticks, dtype=int)
    tick_gal = np.linspace(0, N - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_px)
    ax.set_xticklabels(tick_gal)
    ax.set_yticks(tick_px)
    ax.set_yticklabels(tick_gal)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Covariance image saved to: {output_file}")

    # ---- statistics ----
    off_mask = ~np.eye(N, dtype=bool)
    off_vals = mat[off_mask]
    print(f"\n{label} statistics:")
    print(f"  Diagonal  : mean={np.diag(mat).mean():.5f}, "
          f"range=[{np.diag(mat).min():.5f}, {np.diag(mat).max():.5f}]")
    print(f"  Off-diag  : mean={off_vals.mean():.6f}, "
          f"std={off_vals.std():.6f}, "
          f"max|val|={np.abs(off_vals).max():.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot DESI observed vs predicted absolute magnitudes.\n"
            "Reads the CSV produced by magnitude_prediction_normal.py --fits-file."
        )
    )
    parser.add_argument(
        "--input", type=str,
        default="desi_normal_predictions.csv",
        help="Path to predictions CSV (output of magnitude_prediction_normal.py).",
    )
    parser.add_argument(
        "--output", type=str,
        default="desi_magnitude_comparison.png",
        help="Output comparison plot file path.",
    )
    parser.add_argument(
        "--residuals-output", type=str,
        default="desi_magnitude_residuals.png",
        help="Output residuals plot file path.",
    )
    parser.add_argument(
        "--redshift-output", type=str,
        default="desi_magnitude_residuals_vs_z.png",
        help="Output residuals-vs-redshift plot file path.",
    )
    parser.add_argument(
        "--max-galaxies", type=int, default=None,
        help="Maximum number of galaxies to plot.",
    )
    parser.add_argument(
        "--plot-type", type=str, default="all",
        choices=["comparison", "residuals", "redshift", "covariance", "all"],
        help="Type of plot to generate (default: all).",
    )
    parser.add_argument(
        "--cov-file", type=str, default="desi_normal_cov.npy",
        help="Path to covariance matrix .npy file (for --plot-type covariance).",
    )
    parser.add_argument(
        "--cov-output", type=str, default="desi_cov_matrix.png",
        help="Output file for covariance image plot.",
    )
    parser.add_argument(
        "--cov-sort-by", type=str, default="x_obs",
        help="CSV column to sort galaxies by in the covariance image (default: x_obs). "
             "Set to '' to keep original order.",
    )
    parser.add_argument(
        "--cov-max-display", type=int, default=1000,
        help="Max image resolution per axis for covariance plot (default: 1000).",
    )
    parser.add_argument(
        "--cov-show-raw", action="store_true",
        help="Plot raw covariance instead of correlation matrix.",
    )

    # CI percentile pair
    parser.add_argument(
        "--ci-lo", type=float, default=16.0,
        help="Lower percentile for CI error bars (default: 16 → 68%% CI).",
    )
    parser.add_argument(
        "--ci-hi", type=float, default=84.0,
        help="Upper percentile for CI error bars (default: 84 → 68%% CI).",
    )

    # Sample-cut parameters (must match desi_data.py defaults)
    parser.add_argument(
        "--haty-max", type=float, default=-18.0,
        help="Upper magnitude limit for in-sample cut (default: -18.0).",
    )
    parser.add_argument(
        "--slope-plane", type=float, default=-6.5,
        help="Slope of the oblique plane cut (default: -6.5).",
    )
    parser.add_argument(
        "--intercept-plane", type=float, default=-20.5,
        help="Lower intercept of the plane cut (default: -20.5).",
    )
    parser.add_argument(
        "--intercept-plane2", type=float, default=-18.5,
        help="Upper intercept of the plane cut (default: -18.5).",
    )

    args = parser.parse_args()

    cut_kwargs = dict(
        haty_max=args.haty_max,
        slope_plane=args.slope_plane,
        intercept_plane=args.intercept_plane,
        intercept_plane2=args.intercept_plane2,
        ci_lo_pct=args.ci_lo,
        ci_hi_pct=args.ci_hi,
    )

    if args.plot_type in ["comparison", "all"]:
        plot_desi_magnitude_comparison(
            args.input, args.output, args.max_galaxies, **cut_kwargs
        )

    if args.plot_type in ["residuals", "all"]:
        plot_desi_residuals(
            args.input, args.residuals_output, args.max_galaxies, **cut_kwargs
        )

    if args.plot_type in ["redshift", "all"]:
        plot_desi_residuals_vs_redshift(
            args.input, args.redshift_output, args.max_galaxies, **cut_kwargs
        )

    if args.plot_type in ["covariance", "all"]:
        import os
        if not os.path.exists(args.cov_file):
            print(f"Warning: covariance file '{args.cov_file}' not found; "
                  f"skipping covariance plot.")
        else:
            plot_covariance_matrix(
                cov_file=args.cov_file,
                csv_file=args.input,
                output_file=args.cov_output,
                sort_by=args.cov_sort_by,
                max_display=args.cov_max_display,
                show_correlation=not args.cov_show_raw,
            )


if __name__ == "__main__":
    main()
