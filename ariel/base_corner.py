#!/usr/bin/env python3
"""
Create corner plot from base.stan MCMC output files using ChainConsumer.

This script reads the output_base_?.csv files and creates a corner plot
of the key parameters: slope, intercept.1, sigma_int_x, sigma_int_y
"""

import pandas as pd
import numpy as np
from chainconsumer import ChainConsumer, Chain, Truth
import glob


def load_stan_csv(filename):
    """
    Load a Stan CSV output file, skipping the comment lines.
    
    Parameters
    ----------
    filename : str
        Path to the Stan CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the MCMC samples
    """
    # Read the file, skipping comment lines that start with #
    df = pd.read_csv(filename, comment='#')
    return df


def create_corner_plot(file_pattern='output_base_*.csv', output_file='base_corner_plot.png',
                       include_theta_int=False, truth_values=None):
    """
    Create a corner plot from combined Stan output files.
    
    Parameters
    ----------
    file_pattern : str
        Glob pattern to match Stan output files (default: 'output_base_*.csv')
    output_file : str
        Path to save the corner plot (default: 'base_corner_plot.png')
    include_theta_int : bool
        Whether to include theta_int in the corner plot (default: False)
    truth_values : dict, optional
        Dictionary of true parameter values to plot as reference lines
        Example: {"slope": -8.0, "intercept": -20.0, "sigma_int_x": 0.03, "sigma_int_y": 0.03}
    """
    # Find all matching files
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(files)} output files:")
    for f in files:
        print(f"  - {f}")
    
    # Load and combine all chains
    all_chains = []
    for filename in files:
        df = load_stan_csv(filename)
        all_chains.append(df)
        print(f"Loaded {filename}: {len(df)} samples")
    
    # Combine all chains
    combined_df = pd.concat(all_chains, ignore_index=True)
    print(f"\nTotal combined samples: {len(combined_df)}")
    
    # Extract the parameters of interest (non-standardized versions)
    # Note: Using the original scale parameters, not the _std versions
    params = {
        'slope': combined_df['slope'].values,
        'intercept': combined_df['intercept.1'].values,
        'sigma_int_x': combined_df['sigma_int_x'].values,
        'sigma_int_y': combined_df['sigma_int_y'].values
    }
    
    # Optionally include theta_int
    if include_theta_int:
        params['theta_int'] = combined_df['theta_int'].values
    
    # Print parameter statistics
    print("\nParameter statistics:")
    for name, values in params.items():
        print(f"  {name}:")
        print(f"    mean = {np.mean(values):.6f}")
        print(f"    std  = {np.std(values):.6f}")
        print(f"    median = {np.median(values):.6f}")
    
    # Create Chain object
    # Convert params dict to DataFrame for ChainConsumer
    params_df = pd.DataFrame(params)
    chain = Chain(samples=params_df, name="Base Model")
    
    # Create ChainConsumer object and add the chain
    c = ChainConsumer()
    c.add_chain(chain)
    
    # Add truth values if provided
    if truth_values is not None:
        c.add_truth(Truth(location=truth_values))
    
    # Generate the corner plot
    fig = c.plotter.plot()
    
    # Save the figure
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCorner plot saved to: {output_file}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    summary = c.analysis.get_summary()
    for param in params.keys():
        if param in summary:
            print(f"\n{param}:")
            for key, value in summary[param].items():
                print(f"  {key}: {value}")
    
    return fig


if __name__ == '__main__':
    # Create the corner plot from all output_base_*.csv files
    infile = "DESI_TF_*.csv"
    outfile = "DESI_TF.png"
    infile = "TF_mock_base_*.csv"
    outfile = "TF_mock_base.png"
    # Define true parameter values
    truth = {
        "slope": -8.0,
        "intercept": -20.0,
        "sigma_int_x": 0.03,
        "sigma_int_y": 0.03
    }

    create_corner_plot(file_pattern=infile, output_file=outfile, truth_values=truth)
