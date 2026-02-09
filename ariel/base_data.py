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
import numpy as np


def process_tf_data(csv_file, data_output_file, init_output_file, haty_max=-16):
    """
    Process TF mock data: convert to Stan JSON format and create initial conditions.
    
    This consolidated method performs both data conversion and initial condition
    generation in a single pass through the CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to input CSV file
    data_output_file : str
        Path to output JSON file for Stan data
    init_output_file : str
        Path to output JSON file for initial conditions
    haty_max : float, optional
        Selection function threshold for filtering galaxies (default: -16)
        Only galaxies with M_abs < haty_max are included
    """
    
    # ============================================================================
    # SECTION 1: READ CSV DATA
    # ============================================================================
    x_data = []  # log(Vrot/V0) - this is x in the model
    y_data = []  # Absolute magnitude - this is y in the model
    sigma_x_data = []  # Uncertainty in log(Vrot/V0)
    sigma_y_data = []  # Uncertainty in M_abs
    
    # Track filtering statistics
    total_rows = 0
    filtered_rows = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            y_val = float(row['M_abs'])
            
            # Only include rows with y < haty_max
            if y_val < haty_max:
                x_data.append(float(row['log_V_V0']))
                y_data.append(y_val)
                sigma_x_data.append(float(row['log_V_V0_unc']))
                sigma_y_data.append(float(row['M_abs_unc']))
                filtered_rows += 1
            # Note: zobs column is present but not used by base.stan
    
    # Convert to numpy arrays for calculations
    x = np.array(x_data)
    y = np.array(y_data)
    N_total = len(x)
    
    # ============================================================================
    # SECTION 2: CREATE STAN DATA DICTIONARY
    # ============================================================================
    # Since N_bins = 1, all galaxies are in the same bin
    N_bins = 1
    N_gal = [N_total]
    
    # Create bin assignment (all galaxies in bin 1)
    bin_idx = [1] * N_total
    
    # Create the data dictionary for Stan
    # Using uncertainties from CSV file (log_V_V0_unc and M_abs_unc columns)
    stan_data = {
        'N_bins': N_bins,
        'N_total': N_total,
        'x': x_data,
        'sigma_x': sigma_x_data,
        'y': y_data,
        'sigma_y': sigma_y_data,
        'haty_max': haty_max,
        'bin_idx': bin_idx
    }
    
    # Write Stan data to JSON file
    with open(data_output_file, 'w') as f:
        json.dump(stan_data, f, indent=2)
    
    # ============================================================================
    # SECTION 3: CALCULATE STANDARDIZATION AND LINEAR REGRESSION
    # ============================================================================
    # Calculate standardization parameters (same as in base.stan transformed data)
    mean_x = np.mean(x)
    sd_x = np.std(x, ddof=1)  # ddof=1 for sample standard deviation
    
    # Standardize x values: x_std = (x - mean_x) / sd_x
    x_std = (x - mean_x) / sd_x
    
    # Linear regression: y = intercept_std + slope_std * x_std
    # Use numpy.polyfit with deg=1 to calculate slope and intercept
    # polyfit returns [slope, intercept] for degree 1 polynomial
    slope_std, intercept_std = np.polyfit(x_std, y, deg=1)
    
    # Calculate slope and intercept in terms of original x
    # y = intercept_std + slope_std * x_std
    # y = intercept_std + slope_std * (x - mean_x) / sd_x
    # y = (intercept_std - slope_std * mean_x / sd_x) + (slope_std / sd_x) * x
    slope_orig = slope_std / sd_x
    intercept_orig = intercept_std - slope_std * mean_x / sd_x
    
    # For N_bins=1, intercept_std is a single value, but Stan expects a vector
    # We'll provide it as a list with one element
    intercept_std_vec = [float(intercept_std)]
    
    # ============================================================================
    # SECTION 4: CREATE INITIAL CONDITIONS DICTIONARY
    # ============================================================================
    # Convert numpy arrays to lists for JSON serialization
    init_data = {
        # 'x_TF_std': x_std.tolist(),
        'slope_std': float(slope_std),
        'intercept_std': intercept_std_vec,
        'sigma_int_tot_y': 0.05,
        'theta_int': np.pi/4
    }

    # Write initial conditions to JSON file
    with open(init_output_file, 'w') as f:
        json.dump(init_data, f, indent=2)
    
    # ============================================================================
    # SECTION 5: PRINT SUMMARY STATISTICS
    # ============================================================================
    print(f"Data conversion complete!")
    print(f"Stan data output file: {data_output_file}")
    print(f"Initial conditions output file: {init_output_file}")
    print(f"\nFiltering:")
    print(f"  Total rows in CSV: {total_rows}")
    print(f"  Rows with y < {haty_max}: {filtered_rows}")
    print(f"  Rows filtered out: {total_rows - filtered_rows}")
    print(f"  haty_max (selection threshold): {haty_max}")
    print(f"\nSummary:")
    print(f"  Number of redshift bins: {N_bins}")
    print(f"  Total number of galaxies: {N_total}")
    print(f"  Galaxies per bin: {N_gal}")
    print(f"\nData ranges:")
    print(f"  x (log_V_V0): [{np.min(x):.3f}, {np.max(x):.3f}]")
    print(f"  y (M_abs): [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"  sigma_x (log_V_V0_unc): {np.array(sigma_x_data).mean():.3f} (mean), range [{np.min(sigma_x_data):.3f}, {np.max(sigma_x_data):.3f}]")
    print(f"  sigma_y (M_abs_unc): {np.array(sigma_y_data).mean():.3f} (mean), range [{np.min(sigma_y_data):.3f}, {np.max(sigma_y_data):.3f}]")
    print(f"\nStandardization statistics:")
    print(f"  mean(x): {mean_x:.3f}")
    print(f"  sd(x): {sd_x:.3f}")
    print(f"  x_TF_std (= x_std) range: [{np.min(x_std):.3f}, {np.max(x_std):.3f}]")
    print(f"\nLinear regression (y ~ x_TF_std):")
    print(f"  slope_std: {float(slope_std):.6f}")
    print(f"  intercept_std: {float(intercept_std):.6f}")
    print(f"\nLinear regression (y ~ x_original):")
    print(f"  slope_orig: {slope_orig:.6f}")
    print(f"  intercept_orig: {intercept_orig:.6f}")
    print(f"\nInitial scatter parameters:")
    print(f"  sigma_int_tot_y: {init_data['sigma_int_tot_y']}")
    print(f"  theta_int: {init_data['theta_int']}")


if __name__ == '__main__':
    # Input and output file paths
    # input_csv = 'TF_mock_input.csv'
    input_csv = 'TF_mock_tophat-mag_input.csv'
    output_json = 'TF_mock_input.json'
    init_json = 'TF_mock_init.json'
    
    # Process TF data: convert to Stan format and create initial conditions
    process_tf_data(input_csv, output_json, init_json)
