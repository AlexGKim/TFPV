#!/usr/bin/env python3
"""
Convert TF_mock_input.csv to JSON format for base.stan

This script reads the Tully-Fisher mock data and converts it to the format
expected by base.stan with:
- N_bins = 1 (single redshift bin)
- sigma_x = 0 (no x uncertainties)
- sigma_y = 0 (no y uncertainties)
"""

import csv
import json


def convert_tf_data_to_stan(csv_file, output_file):
    """
    Convert TF_mock_input.csv to Stan JSON format.
    
    Parameters
    ----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output JSON file
    """
    # Read the CSV file
    x_data = []  # log(Vrot/V0) - this is x in the model
    y_data = []  # Absolute magnitude - this is y in the model
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_data.append(float(row['log_V_V0']))
            y_data.append(float(row['M_abs']))
    
    # Number of galaxies
    N_total = len(x_data)
    
    # Since N_bins = 1, all galaxies are in the same bin
    N_bins = 1
    N_gal = [N_total]
    
    # Create bin assignment (all galaxies in bin 1)
    bin_idx = [1] * N_total
    
    # Set uncertainties to small non-zero values (Stan model requires positive values)
    # Using a small value instead of exactly 0.0 to avoid numerical issues
    sigma_x = [0.01] * N_total
    sigma_y = [0.01] * N_total
    
    # Create the data dictionary for Stan
    stan_data = {
        'N_bins': N_bins,
        'N_total': N_total,
        'x': x_data,
        'sigma_x': sigma_x,
        'y': y_data,
        'sigma_y': sigma_y,
        'bin_idx': bin_idx
    }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(stan_data, f, indent=2)
    
    # Print summary statistics
    print(f"Data conversion complete!")
    print(f"Output file: {output_file}")
    print(f"\nSummary:")
    print(f"  Number of redshift bins: {N_bins}")
    print(f"  Total number of galaxies: {N_total}")
    print(f"  Galaxies per bin: {N_gal}")
    print(f"\nData ranges:")
    print(f"  x (log_V_V0): [{min(x_data):.3f}, {max(x_data):.3f}]")
    print(f"  y (M_abs): [{min(y_data):.3f}, {max(y_data):.3f}]")
    print(f"  sigma_x: all set to {sigma_x[0]}")
    print(f"  sigma_y: all set to {sigma_y[0]}")


if __name__ == '__main__':
    # Input and output file paths
    input_csv = 'TF_mock_input.csv'
    output_json = 'TF_mock_input.json'
    
    # Convert the data
    convert_tf_data_to_stan(input_csv, output_json)
