"""
Plot observed absolute magnitude (from DESI FITS) versus predicted absolute magnitude
with error bars from desi_galaxy_magnitude_predictions.csv.

x-axis: y_obs (R_ABSMAG_SB26) with error bar y_unc
y-axis: y_pred_median with error bar from y_pred_CI68_lower/upper

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
    lower_bound  = slope_plane * x + intercept_plane
    upper_oblique = slope_plane * x + intercept_plane2
    upper_bound  = np.minimum(haty_max, upper_oblique)

    mask = (
        (y < haty_max) &
        (lower_bound <= y) &
        (y <= upper_bound)
    )
    return mask


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
    df = df.dropna(subset=['y_obs'])

    if max_galaxies is not None and max_galaxies < len(df):
        print(f"Using only first {max_galaxies} galaxies (out of {len(df)} total)")
        df = df.iloc[:max_galaxies]

    mask = compute_insample_mask(
        df['x_obs'].values,
        df['y_obs'].values,
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
        fmt='o', color=color, alpha=0.2, markersize=3,
        elinewidth=0.5, capsize=0, label=label,
    )


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_desi_magnitude_comparison(
    csv_file,
    output_file='desi_magnitude_comparison.png',
    max_galaxies=None,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
):
    """
    Plot observed R_ABSMAG_SB26 (x-axis) vs predicted magnitude (y-axis).

    In-sample objects (steelblue) and out-of-sample objects (tomato) are
    shown with alpha=0.1 per point.
    """
    print(f"Loading data from {csv_file}...")
    df_in, df_out = _load_and_split(
        csv_file, max_galaxies,
        haty_max, slope_plane, intercept_plane, intercept_plane2,
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    for df_sub, color, label in [
        (df_in,  'steelblue', f'In-sample (N={len(df_in)})'),
        (df_out, 'tomato',    f'Out-of-sample (N={len(df_out)})'),
    ]:
        if len(df_sub) == 0:
            continue
        obs   = df_sub['y_obs'].values
        unc   = df_sub['y_unc'].values if 'y_unc' in df_sub.columns else np.zeros(len(df_sub))
        pred  = df_sub['y_pred_median'].values
        ci_lo = df_sub['y_pred_CI68_lower'].values
        ci_hi = df_sub['y_pred_CI68_upper'].values
        ax.errorbar(
            obs, pred,
            xerr=unc,
            yerr=[pred - ci_lo, ci_hi - pred],
            **_errorbar_kwargs(color, label),
        )

    # 1:1 line
    all_obs  = pd.concat([df_in, df_out])['y_obs'].values
    all_pred = pd.concat([df_in, df_out])['y_pred_median'].values
    lo = min(all_obs.min(), all_pred.min()) - 0.5
    hi = max(all_obs.max(), all_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='1:1 line', zorder=10)

    ax.set_xlabel(r'Observed $M_{\rm abs}$ (R_ABSMAG_SB26)', fontsize=14)
    ax.set_ylabel(r'Predicted $M_{\rm abs}$ (median)', fontsize=14)
    ax.set_title(
        'DESI: Observed vs TF-Predicted Absolute Magnitudes\n'
        '(x error: obs. unc.; y error: 68% CI)',
        fontsize=14, fontweight='bold',
    )
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_xaxis()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_file}")

    # ---- statistics (in-sample only) ----
    if len(df_in):
        obs   = df_in['y_obs'].values
        pred  = df_in['y_pred_median'].values
        ci_lo = df_in['y_pred_CI68_lower'].values
        ci_hi = df_in['y_pred_CI68_upper'].values
        unc   = df_in['y_unc'].values if 'y_unc' in df_in.columns else np.zeros(len(df_in))
        residuals  = obs - pred
        within_ci  = ((obs >= ci_lo) & (obs <= ci_hi)).sum()
        n = len(df_in)
        print(f"\nStatistics — in-sample ({n} galaxies):")
        print(f"  Mean observed M_abs : {obs.mean():.3f} ± {obs.std():.3f}")
        print(f"  Mean predicted M_abs: {pred.mean():.3f} ± {pred.std():.3f}")
        print(f"  Mean 68% CI width   : {(ci_hi - ci_lo).mean():.3f}")
        print(f"  Mean obs. unc.      : {unc.mean():.3f}")
        print(f"  Mean residual       : {residuals.mean():.3f} ± {residuals.std():.3f}")
        print(f"  RMS residual        : {np.sqrt(np.mean(residuals**2)):.3f}")
        print(f"  Fraction within 68% CI: {within_ci/n:.3f} ({within_ci}/{n})")


# ---------------------------------------------------------------------------
# Residuals plot
# ---------------------------------------------------------------------------

def plot_desi_residuals(
    csv_file,
    output_file='desi_magnitude_residuals.png',
    max_galaxies=None,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
):
    """
    Plot residuals (y_obs - y_pred_median) versus observed magnitude.

    In-sample objects (steelblue) and out-of-sample objects (tomato) are
    shown with alpha=0.1 per point.
    """
    print(f"\nLoading data from {csv_file}...")
    df_in, df_out = _load_and_split(
        csv_file, max_galaxies,
        haty_max, slope_plane, intercept_plane, intercept_plane2,
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for df_sub, color, label in [
        (df_in,  'steelblue', f'In-sample (N={len(df_in)})'),
        (df_out, 'tomato',    f'Out-of-sample (N={len(df_out)})'),
    ]:
        if len(df_sub) == 0:
            continue
        obs   = df_sub['y_obs'].values
        unc   = df_sub['y_unc'].values if 'y_unc' in df_sub.columns else np.zeros(len(df_sub))
        pred  = df_sub['y_pred_median'].values
        ci_lo = df_sub['y_pred_CI68_lower'].values
        ci_hi = df_sub['y_pred_CI68_upper'].values
        resid = obs - pred
        ax.errorbar(
            obs, resid,
            xerr=unc,
            yerr=[pred - ci_lo, ci_hi - pred],
            **_errorbar_kwargs(color, label),
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero residual')

    ax.set_xlabel(r'Observed $M_{\rm abs}$ (R_ABSMAG_SB26)', fontsize=14)
    ax.set_ylabel(r'Residual (Observed $-$ Predicted)', fontsize=14)
    ax.set_title(
        'DESI: Residuals vs Observed Magnitude\n'
        '(x error: obs. unc.; y error: 68% CI)',
        fontsize=14, fontweight='bold',
    )
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # brighter magnitudes on right

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Residuals plot saved to: {output_file}")

    # ---- statistics (in-sample only) ----
    if len(df_in):
        resid = df_in['y_obs'].values - df_in['y_pred_median'].values
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
    output_file='desi_magnitude_residuals_vs_z.png',
    max_galaxies=None,
    haty_max: float = -18.0,
    slope_plane: float = -6.5,
    intercept_plane: float = -20.5,
    intercept_plane2: float = -18.5,
):
    """
    Plot residuals (y_obs - y_pred_median) versus redshift (zobs).

    In-sample objects (steelblue) and out-of-sample objects (tomato) are
    shown with alpha=0.1 per point.  Requires the 'zobs' column to be
    present in the predictions CSV (written when --z-col is set).
    """
    print(f"\nLoading data from {csv_file}...")
    df_in, df_out = _load_and_split(
        csv_file, max_galaxies,
        haty_max, slope_plane, intercept_plane, intercept_plane2,
    )

    # Check redshift column exists
    all_df = pd.concat([df_in, df_out])
    if 'zobs' not in all_df.columns:
        raise KeyError(
            "'zobs' column not found in predictions CSV. "
            "Re-run desi_galaxy_magnitudes.py with --z-col Z_DESI_CMB."
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    for df_sub, color, label in [
        (df_in,  'steelblue', f'In-sample (N={len(df_in)})'),
        (df_out, 'tomato',    f'Out-of-sample (N={len(df_out)})'),
    ]:
        if len(df_sub) == 0:
            continue
        z     = df_sub['zobs'].values
        pred  = df_sub['y_pred_median'].values
        ci_lo = df_sub['y_pred_CI68_lower'].values
        ci_hi = df_sub['y_pred_CI68_upper'].values
        resid = df_sub['y_obs'].values - pred
        ax.errorbar(
            z, resid,
            yerr=[pred - ci_lo, ci_hi - pred],
            **_errorbar_kwargs(color, label),
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero residual')

    ax.set_xlabel(r'Redshift $z_{\rm obs}$', fontsize=14)
    ax.set_ylabel(r'Residual (Observed $-$ Predicted)', fontsize=14)
    ax.set_title(
        'DESI: Residuals vs Redshift\n'
        '(y error: 68% CI)',
        fontsize=14, fontweight='bold',
    )
    ax.legend(loc='best', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Residuals-vs-redshift plot saved to: {output_file}")

    # ---- statistics (in-sample only) ----
    if len(df_in):
        resid = df_in['y_obs'].values - df_in['y_pred_median'].values
        z     = df_in['zobs'].values
        n     = len(df_in)
        print(f"\nResidual vs z Statistics — in-sample ({n} galaxies):")
        print(f"  Redshift range : [{z.min():.4f}, {z.max():.4f}]")
        print(f"  Mean residual  : {resid.mean():.3f}")
        print(f"  Std residual   : {resid.std():.3f}")
        print(f"  RMS residual   : {np.sqrt(np.mean(resid**2)):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot DESI observed vs predicted absolute magnitudes'
    )
    parser.add_argument('--input', type=str,
                        default='desi_galaxy_magnitude_predictions.csv',
                        help='Path to DESI predictions CSV file')
    parser.add_argument('--output', type=str,
                        default='desi_magnitude_comparison.png',
                        help='Output comparison plot file path')
    parser.add_argument('--residuals-output', type=str,
                        default='desi_magnitude_residuals.png',
                        help='Output residuals plot file path')
    parser.add_argument('--redshift-output', type=str,
                        default='desi_magnitude_residuals_vs_z.png',
                        help='Output residuals-vs-redshift plot file path')
    parser.add_argument('--max-galaxies', type=int, default=None,
                        help='Maximum number of galaxies to plot')
    parser.add_argument('--plot-type', type=str, default='all',
                        choices=['comparison', 'residuals', 'redshift', 'all'],
                        help='Type of plot to generate (default: all)')

    # Sample-cut parameters (must match desi_data.py defaults)
    parser.add_argument('--haty-max', type=float, default=-18.0,
                        help='Upper magnitude limit for in-sample cut (default: -18.0)')
    parser.add_argument('--slope-plane', type=float, default=-6.5,
                        help='Slope of the oblique plane cut (default: -6.5)')
    parser.add_argument('--intercept-plane', type=float, default=-20.5,
                        help='Lower intercept of the plane cut (default: -20.5)')
    parser.add_argument('--intercept-plane2', type=float, default=-18.5,
                        help='Upper intercept of the plane cut (default: -18.5)')

    args = parser.parse_args()

    cut_kwargs = dict(
        haty_max=args.haty_max,
        slope_plane=args.slope_plane,
        intercept_plane=args.intercept_plane,
        intercept_plane2=args.intercept_plane2,
    )

    if args.plot_type in ['comparison', 'all']:
        plot_desi_magnitude_comparison(
            args.input, args.output, args.max_galaxies, **cut_kwargs
        )

    if args.plot_type in ['residuals', 'all']:
        plot_desi_residuals(
            args.input, args.residuals_output, args.max_galaxies, **cut_kwargs
        )

    if args.plot_type in ['redshift', 'all']:
        plot_desi_residuals_vs_redshift(
            args.input, args.redshift_output, args.max_galaxies, **cut_kwargs
        )


if __name__ == '__main__':
    main()
