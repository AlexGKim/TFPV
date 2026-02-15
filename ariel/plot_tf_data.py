#!/usr/bin/env python3
"""
Create a scatter plot with error bars from TF mock data JSON file.

This script reads the Tully-Fisher mock data from a JSON file and creates
a scatter plot showing x (log_V_V0) vs y (M_abs) with error bars.
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def plot_tf_data(json_file, output_file='tf_scatter_plot.png'):
    """
    Create scatter plot with error bars from TF data JSON file.
    
    Parameters
    ----------
    json_file : str
        Path to input JSON file containing TF data
    output_file : str
        Path to output plot file (default: 'tf_scatter_plot.png')
    """
    
    # Read JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract data arrays
    x = np.array(data['x'])
    y = np.array(data['y'])
    sigma_x = np.array(data['sigma_x'])
    sigma_y = np.array(data['sigma_y'])
    
    # Get additional info
    N_total = data['N_total']
    haty_max = data.get('haty_max', None)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with error bars
    ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, 
                fmt='o', markersize=3, alpha=0.6,
                elinewidth=0.5, capsize=0,
                label=f'N = {N_total}')
    
    # Add horizontal line for haty_max if present
    if haty_max is not None:
        ax.axhline(y=haty_max, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7,
                   label=f'$\\hat{{y}}_{{\\rm max}}$ = {haty_max}')
    
    # Labels and title
    ax.set_xlabel(r'$\hat{x}$ = log($V_{\rm rot}/V_0$)', fontsize=12)
    ax.set_ylabel(r'$\hat{y}$ = $M_{\rm abs}$ (absolute magnitude)', fontsize=12)
    ax.set_title('Tully-Fisher Mock Data', fontsize=14, fontweight='bold')
    
    # Invert y-axis (brighter magnitudes are more negative)
    ax.invert_yaxis()
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nData summary:")
    print(f"  Number of galaxies: {N_total}")
    print(f"  x range: [{np.min(x):.3f}, {np.max(x):.3f}]")
    print(f"  y range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"  sigma_x: mean = {np.mean(sigma_x):.4f}, range = [{np.min(sigma_x):.4f}, {np.max(sigma_x):.4f}]")
    print(f"  sigma_y: mean = {np.mean(sigma_y):.4f}, range = [{np.min(sigma_y):.4f}, {np.max(sigma_y):.4f}]")
    if haty_max is not None:
        print(f"  haty_max: {haty_max}")
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    import sys
    
    # Check if filename argument is provided
    if len(sys.argv) < 2:
        print("Usage: python plot_tf_data.py <json_file> [output_file]")
        print("Example: python plot_tf_data.py TF_mock_input_n10000.json")
        sys.exit(1)
    
    # Get filename from command line argument
    json_file = sys.argv[1]
    
    # Optional output filename (default based on input filename)
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generate output filename from input filename
        output_file = json_file.replace('.json', '_plot.png')
    
    # Create plot
    plot_tf_data(json_file, output_file)
