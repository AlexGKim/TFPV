"""
Plot observed M_abs versus predicted M_abs with error bars.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_magnitude_comparison(csv_file, output_file='magnitude_comparison.png', max_galaxies=None):
    """
    Plot observed M_abs (x-axis) versus predicted M_abs (y-axis) with error bars.
    
    Parameters
    ----------
    csv_file : str
        Path to the predictions CSV file
    output_file : str
        Path to save the plot
    max_galaxies : int or None
        Maximum number of galaxies to plot (for clarity)
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if max_galaxies is not None and max_galaxies < len(df):
        print(f"Plotting only first {max_galaxies} galaxies (out of {len(df)} total)")
        df = df.iloc[:max_galaxies]
    
    n_galaxies = len(df)
    
    # Extract data
    M_abs_obs = df['M_abs'].values
    M_abs_obs_unc = df['M_abs_unc'].values
    M_abs_pred = df['M_abs_pred_median'].values
    M_abs_pred_lower = df['M_abs_pred_CI68_lower'].values
    M_abs_pred_upper = df['M_abs_pred_CI68_upper'].values
    
    # Calculate error bars for predicted values
    M_abs_pred_err_lower = M_abs_pred - M_abs_pred_lower
    M_abs_pred_err_upper = M_abs_pred_upper - M_abs_pred
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot with error bars
    ax.errorbar(M_abs_obs, M_abs_pred, 
                xerr=M_abs_obs_unc,
                yerr=[M_abs_pred_err_lower, M_abs_pred_err_upper],
                fmt='o', color='blue', alpha=0.5, markersize=3,
                elinewidth=0.8, capsize=0, label='Galaxies')
    
    # Plot 1:1 line
    min_val = min(M_abs_obs.min(), M_abs_pred.min())
    max_val = max(M_abs_obs.max(), M_abs_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='1:1 line', zorder=10)
    
    # Labels and formatting
    ax.set_xlabel('Observed M_abs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted M_abs (median)', fontsize=14, fontweight='bold')
    ax.set_title('Observed vs Predicted Absolute Magnitudes\n(Error bars: obs. unc. and 68% CI)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Invert both axes (brighter = more negative = up/left)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure (without showing)
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to avoid display
    print(f"Plot saved successfully!")
    
    # Print statistics
    print("\nStatistics:")
    print(f"  Number of galaxies plotted: {n_galaxies}")
    print(f"  Mean observed M_abs: {M_abs_obs.mean():.3f} ± {M_abs_obs.std():.3f}")
    print(f"  Mean predicted M_abs: {M_abs_pred.mean():.3f} ± {M_abs_pred.std():.3f}")
    print(f"  Mean 68% CI width: {(M_abs_pred_upper - M_abs_pred_lower).mean():.3f}")
    print(f"  Mean obs. uncertainty: {M_abs_obs_unc.mean():.3f}")
    
    # Calculate residuals
    residuals = M_abs_obs - M_abs_pred
    print(f"  Mean residual (obs - pred): {residuals.mean():.3f} ± {residuals.std():.3f}")
    
    # Calculate how many observed values fall within predicted CI
    within_ci = ((M_abs_obs >= M_abs_pred_lower) & 
                 (M_abs_obs <= M_abs_pred_upper)).sum()
    fraction_within = within_ci / n_galaxies
    print(f"  Fraction of observed M_abs within predicted 68% CI: {fraction_within:.3f} ({within_ci}/{n_galaxies})")


def plot_residuals(csv_file, output_file='magnitude_residuals.png', max_galaxies=None):
    """
    Plot residuals: (M_abs - M_abs_pred_median) versus observed M_abs.
    
    Parameters
    ----------
    csv_file : str
        Path to the predictions CSV file
    output_file : str
        Path to save the plot
    max_galaxies : int or None
        Maximum number of galaxies to plot
    """
    # Load data
    print(f"\nLoading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if max_galaxies is not None and max_galaxies < len(df):
        print(f"Plotting only first {max_galaxies} galaxies (out of {len(df)} total)")
        df = df.iloc[:max_galaxies]
    
    # Calculate residuals
    M_abs_obs = df['M_abs'].values
    M_abs_obs_unc = df['M_abs_unc'].values
    M_abs_pred = df['M_abs_pred_median'].values
    residuals = M_abs_obs - M_abs_pred
    
    # Predicted uncertainties
    M_abs_pred_lower = df['M_abs_pred_CI68_lower'].values
    M_abs_pred_upper = df['M_abs_pred_CI68_upper'].values
    pred_err_lower = M_abs_pred - M_abs_pred_lower
    pred_err_upper = M_abs_pred_upper - M_abs_pred
    
    # Combined uncertainty (quadrature sum)
    total_unc = np.sqrt(M_abs_obs_unc**2 + ((pred_err_lower + pred_err_upper)/2)**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot residuals vs observed M_abs
    ax.errorbar(M_abs_obs, residuals, 
                xerr=M_abs_obs_unc,
                yerr=[pred_err_lower, pred_err_upper],
                fmt='o', color='blue', alpha=0.5, markersize=3,
                elinewidth=0.8, capsize=0, label='Residuals')
    
    # Zero line
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    
    # Labels and formatting
    ax.set_xlabel('Observed M_abs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Residual (Observed - Predicted)', fontsize=14, fontweight='bold')
    ax.set_title('Residuals vs Observed Magnitude\n(Error bars: obs. unc. and 68% CI)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Brighter magnitudes on right
    
    plt.tight_layout()
    
    # Save figure (without showing)
    print(f"Saving residuals plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close figure to avoid display
    print(f"Residuals plot saved successfully!")
    
    # Print statistics
    print("\nResidual Statistics:")
    print(f"  Mean residual: {residuals.mean():.3f}")
    print(f"  Std residual: {residuals.std():.3f}")
    print(f"  RMS residual: {np.sqrt(np.mean(residuals**2)):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot observed vs predicted galaxy magnitudes'
    )
    parser.add_argument('--input', type=str, 
                        default='galaxy_magnitude_predictions.csv',
                        help='Path to predictions CSV file')
    parser.add_argument('--output', type=str,
                        default='magnitude_comparison.png',
                        help='Output plot file path')
    parser.add_argument('--residuals-output', type=str,
                        default='magnitude_residuals.png',
                        help='Output residuals plot file path')
    parser.add_argument('--max-galaxies', type=int, default=None,
                        help='Maximum number of galaxies to plot (for clarity)')
    parser.add_argument('--plot-type', type=str, default='both',
                        choices=['comparison', 'residuals', 'both'],
                        help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if args.plot_type in ['comparison', 'both']:
        plot_magnitude_comparison(args.input, args.output, args.max_galaxies)
    
    if args.plot_type in ['residuals', 'both']:
        plot_residuals(args.input, args.residuals_output, args.max_galaxies)


if __name__ == '__main__':
    main()
