"""
Infer the absolute magnitude of individual galaxies from their rotation velocities
using the Tully-Fisher relation calibration.

This implements the procedure described in Section "Inferring the Absolute Magnitude 
of an Individual Galaxy from its Rotation Velocity" using a bounded top-hat prior 
on y_TF with Monte Carlo composition.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import argparse


def truncated_normal_sample(loc, scale, lower, upper, size=1):
    """
    Sample from a truncated normal distribution.
    
    Parameters
    ----------
    loc : float or array
        Mean of the untruncated normal
    scale : float or array
        Standard deviation of the untruncated normal
    lower : float
        Lower bound
    upper : float
        Upper bound
    size : int or tuple
        Number of samples
        
    Returns
    -------
    samples : array
        Samples from the truncated normal distribution
    """
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)


def infer_magnitude_single_galaxy(x_hat_star, sigma_x_star, posterior_draws,
                                   y_min, y_max, n_samples_per_draw=1):
    """
    Infer the absolute magnitude y_star for a single galaxy given its observed
    rotation velocity x_hat_star and the TF calibration posterior.
    
    Uses Monte Carlo composition with a bounded top-hat prior on y_TF.
    
    Parameters
    ----------
    x_hat_star : float
        Observed rotation velocity proxy (log_V_V0) for the galaxy
    sigma_x_star : float
        Measurement uncertainty in x_hat_star
    posterior_draws : DataFrame
        Posterior draws with columns: slope (s), intercept.1 (c),
        sigma_int_x, sigma_int_y
    y_min : float
        Lower bound for y_TF prior
    y_max : float
        Upper bound for y_TF prior
    n_samples_per_draw : int
        Number of Monte Carlo samples per posterior draw
        
    Returns
    -------
    y_star_samples : array
        Posterior predictive samples for y_star
    """
    # Extract parameters from posterior draws
    s = posterior_draws['slope'].values
    c = posterior_draws['intercept.1'].values
    sigma_int_x = posterior_draws['sigma_int_x'].values
    sigma_int_y = posterior_draws['sigma_int_y'].values
    
    n_draws = len(s)
    
    # Vectorize calculations
    sigma_1_star = np.sqrt(sigma_x_star**2 + sigma_int_x**2)
    loc_ytf = c + s * x_hat_star
    scale_ytf = np.abs(s) * sigma_1_star
    
    # Compute truncation bounds
    a = (y_min - loc_ytf) / scale_ytf
    b = (y_max - loc_ytf) / scale_ytf
    
    # Sample for each posterior draw
    y_star_samples = np.zeros(n_draws * n_samples_per_draw)
    
    for m in range(n_draws):
        # Sample y_TF from truncated normal
        y_tf = stats.truncnorm.rvs(a[m], b[m], loc=loc_ytf[m],
                                    scale=scale_ytf[m], size=n_samples_per_draw)
        
        # Sample y_star | y_TF from normal
        y_star = np.random.normal(loc=y_tf, scale=sigma_int_y[m], size=n_samples_per_draw)
        
        y_star_samples[m*n_samples_per_draw:(m+1)*n_samples_per_draw] = y_star
    
    return y_star_samples


def compute_credible_interval(samples, credibility=0.68):
    """
    Compute central credible interval from samples.
    
    Parameters
    ----------
    samples : array
        Posterior samples
    credibility : float
        Credibility level (e.g., 0.68 for 68%, 0.95 for 95%)
        
    Returns
    -------
    lower, upper : float
        Lower and upper bounds of the credible interval
    """
    alpha = (1 - credibility) / 2
    lower = np.percentile(samples, alpha * 100)
    upper = np.percentile(samples, (1 - alpha) * 100)
    return lower, upper


def decompose_variance(x_hat_star, sigma_x_star, posterior_draws):
    """
    Decompose the conditional variance into contributions from measurement
    uncertainty, intrinsic scatter in x, and intrinsic scatter in y.
    
    Parameters
    ----------
    x_hat_star : float
        Observed rotation velocity proxy
    sigma_x_star : float
        Measurement uncertainty in x_hat_star
    posterior_draws : DataFrame
        Posterior draws
        
    Returns
    -------
    var_components : dict
        Dictionary with variance components (mean over posterior draws)
    """
    s = posterior_draws['slope'].values
    sigma_int_x = posterior_draws['sigma_int_x'].values
    sigma_int_y = posterior_draws['sigma_int_y'].values
    
    # Variance components
    var_measurement = s**2 * sigma_x_star**2
    var_int_x = s**2 * sigma_int_x**2
    var_int_y = sigma_int_y**2
    var_total = var_measurement + var_int_x + var_int_y
    
    return {
        'var_measurement': np.mean(var_measurement),
        'var_int_x': np.mean(var_int_x),
        'var_int_y': np.mean(var_int_y),
        'var_total': np.mean(var_total),
        'std_total': np.sqrt(np.mean(var_total))
    }


def process_galaxies(galaxy_file, posterior_files, y_min, y_max,
                     output_file='galaxy_magnitude_predictions.csv',
                     n_samples_per_draw=1, credibility_levels=[0.68, 0.95],
                     max_galaxies=None):
    """
    Process all galaxies and compute magnitude predictions with credible intervals.
    
    Parameters
    ----------
    galaxy_file : str
        Path to CSV file with galaxy data (columns: log_V_V0, M_abs, log_V_V0_unc, M_abs_unc, zobs)
    posterior_files : list of str
        List of paths to posterior draw CSV files
    y_min : float
        Lower bound for y_TF prior (absolute magnitude)
    y_max : float
        Upper bound for y_TF prior (absolute magnitude)
    output_file : str
        Path to output CSV file
    n_samples_per_draw : int
        Number of Monte Carlo samples per posterior draw
    credibility_levels : list of float
        Credibility levels for intervals (e.g., [0.68, 0.95])
    max_galaxies : int or None
        Maximum number of galaxies to process (for testing)
    """
    # Load galaxy data
    print(f"Loading galaxy data from {galaxy_file}...")
    galaxies = pd.read_csv(galaxy_file)
    
    # Limit number of galaxies if specified
    if max_galaxies is not None and max_galaxies < len(galaxies):
        print(f"Processing only first {max_galaxies} galaxies (out of {len(galaxies)} total)")
        galaxies = galaxies.iloc[:max_galaxies]
    
    # Load posterior draws from all chains
    print(f"Loading posterior draws from {len(posterior_files)} files...")
    posterior_draws_list = []
    for pfile in posterior_files:
        # Read CSV, skipping comment lines
        df = pd.read_csv(pfile, comment='#')
        # Keep only the parameters we need
        cols_needed = ['slope', 'intercept.1', 'sigma_int_x', 'sigma_int_y']
        if all(col in df.columns for col in cols_needed):
            posterior_draws_list.append(df[cols_needed])
    
    # Combine all posterior draws
    posterior_draws = pd.concat(posterior_draws_list, ignore_index=True)
    print(f"Total posterior draws: {len(posterior_draws)}")
    
    # Process each galaxy
    results = []
    n_galaxies = len(galaxies)
    
    print(f"Processing {n_galaxies} galaxies...")
    for idx, row in galaxies.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{n_galaxies} galaxies...")
        
        x_hat_star = row['log_V_V0']
        sigma_x_star = row['log_V_V0_unc']
        
        # Infer magnitude
        y_star_samples = infer_magnitude_single_galaxy(
            x_hat_star, sigma_x_star, posterior_draws, 
            y_min, y_max, n_samples_per_draw
        )
        
        # Compute summary statistics
        result = {
            'galaxy_id': idx,
            'log_V_V0_obs': x_hat_star,
            'log_V_V0_unc': sigma_x_star,
            'M_abs': row['M_abs'],
            'M_abs_unc': row['M_abs_unc'],
            'zobs': row['zobs'],
            'M_abs_pred_median': np.median(y_star_samples),
            'M_abs_pred_mean': np.mean(y_star_samples),
            'M_abs_pred_std': np.std(y_star_samples)
        }
        
        # Add credible intervals
        for cred in credibility_levels:
            lower, upper = compute_credible_interval(y_star_samples, cred)
            cred_pct = int(cred * 100)
            result[f'M_abs_pred_CI{cred_pct}_lower'] = lower
            result[f'M_abs_pred_CI{cred_pct}_upper'] = upper
        
        # Decompose variance
        var_components = decompose_variance(x_hat_star, sigma_x_star, posterior_draws)
        result.update({
            'var_measurement': var_components['var_measurement'],
            'var_int_x': var_components['var_int_x'],
            'var_int_y': var_components['var_int_y'],
            'var_total': var_components['var_total'],
            'std_total': var_components['std_total']
        })
        
        results.append(result)
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    print(f"Saving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    print(f"Done! Processed {n_galaxies} galaxies.")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean predicted M_abs: {results_df['M_abs_pred_mean'].mean():.3f}")
    print(f"  Mean prediction uncertainty (std): {results_df['M_abs_pred_std'].mean():.3f}")
    print(f"  Mean variance from measurement: {results_df['var_measurement'].mean():.6f}")
    print(f"  Mean variance from intrinsic x: {results_df['var_int_x'].mean():.6f}")
    print(f"  Mean variance from intrinsic y: {results_df['var_int_y'].mean():.6f}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Infer galaxy absolute magnitudes from rotation velocities using TF calibration'
    )
    parser.add_argument('--galaxy-file', type=str, 
                        default='data/TF_mock_tophat-mag_input.csv',
                        help='Path to galaxy data CSV file')
    parser.add_argument('--posterior-pattern', type=str,
                        default='output_base_n10000_[1-4].csv',
                        help='Glob pattern for posterior draw files')
    parser.add_argument('--y-min', type=float, default=-23.0,
                        help='Lower bound for y_TF prior (absolute magnitude)')
    parser.add_argument('--y-max', type=float, default=-15.0,
                        help='Upper bound for y_TF prior (absolute magnitude)')
    parser.add_argument('--output', type=str,
                        default='galaxy_magnitude_predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of Monte Carlo samples per posterior draw')
    parser.add_argument('--credibility', type=float, nargs='+',
                        default=[0.68, 0.95],
                        help='Credibility levels for intervals (e.g., 0.68 0.95)')
    parser.add_argument('--max-galaxies', type=int, default=None,
                        help='Maximum number of galaxies to process (for testing)')
    
    args = parser.parse_args()
    
    # Find posterior files
    from glob import glob
    posterior_files = sorted(glob(args.posterior_pattern))
    
    if not posterior_files:
        print(f"Error: No posterior files found matching pattern '{args.posterior_pattern}'")
        return
    
    print(f"Found {len(posterior_files)} posterior files:")
    for pf in posterior_files:
        print(f"  {pf}")
    
    # Process galaxies
    process_galaxies(
        galaxy_file=args.galaxy_file,
        posterior_files=posterior_files,
        y_min=args.y_min,
        y_max=args.y_max,
        output_file=args.output,
        n_samples_per_draw=args.n_samples,
        credibility_levels=args.credibility,
        max_galaxies=args.max_galaxies
    )


if __name__ == '__main__':
    main()
