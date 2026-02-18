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
import matplotlib.pyplot as plt


def process_tf_data(csv_file, data_output_file, init_output_file, haty_max=-16, sample_size=None,
                    plane_cut=False, slope_plane=None, intercept_plane=None, intercept_plane2=None):
    """
    Process TF mock data: convert to Stan JSON format and create initial conditions.

    Parameters
    ----------
    ...
    plane_cut : bool, optional
        Whether to apply half-plane cut(s) (default: False)

    slope_plane : float, optional
        Slope parameter (bar_s) for half-plane cut(s): y = bar_s * x + bar_c

    intercept_plane : float, optional
        Intercept parameter (bar_c1) for lower bound:
            bar_s * x + intercept_plane <= y
        Required if plane_cut=True

    intercept_plane2 : float, optional
        Optional second intercept (bar_c2) for upper oblique bound:
            y <= bar_s * x + intercept_plane2
        If provided, the sample cut becomes two-sided parallel:
            bar_s*x + c1 <= y <= min(haty_max, bar_s*x + c2)
    """

    # Validate plane cut parameters
    if plane_cut and (slope_plane is None or intercept_plane is None):
        raise ValueError("slope_plane and intercept_plane must be provided when plane_cut=True")

    # If two-sided, enforce ordering c1 < c2 (so lower line is always below upper line)
    two_sided = plane_cut and (intercept_plane2 is not None)
    if two_sided and not (intercept_plane < intercept_plane2):
        raise ValueError(
            f"For a two-sided parallel cut, require intercept_plane < intercept_plane2. "
            f"Got {intercept_plane} and {intercept_plane2}."
        )

    # ============================================================================
    # SECTION 1: READ CSV DATA
    # ============================================================================
    x_data = []        # log(Vrot/V0)
    y_data = []        # Absolute magnitude
    sigma_x_data = []  # Uncertainty in log(Vrot/V0)
    sigma_y_data = []  # Uncertainty in M_abs

    # Track filtering statistics
    total_rows = 0
    y_filtered_rows = 0
    plane_pass_rows = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            x_val = float(row['log_V_V0'])
            y_val = float(row['M_abs'])

            # Apply y upper limit (note: magnitudes: "brighter" is more negative)
            # Your original code uses y_val < haty_max; keep that behavior.
            if y_val < haty_max:
                y_filtered_rows += 1

                if plane_cut:
                    lower_bound = slope_plane * x_val + intercept_plane

                    if not two_sided:
                        # One-sided: lower_bound <= y
                        if lower_bound <= y_val:
                            x_data.append(x_val)
                            y_data.append(y_val)
                            sigma_x_data.append(float(row['log_V_V0_unc']))
                            sigma_y_data.append(float(row['M_abs_unc']))
                            plane_pass_rows += 1
                    else:
                        # Two-sided: lower_bound <= y <= min(haty_max, upper_bound)
                        upper_bound_oblique = slope_plane * x_val + intercept_plane2
                        upper_bound = min(haty_max, upper_bound_oblique)

                        if (lower_bound <= y_val) and (y_val <= upper_bound):
                            x_data.append(x_val)
                            y_data.append(y_val)
                            sigma_x_data.append(float(row['log_V_V0_unc']))
                            sigma_y_data.append(float(row['M_abs_unc']))
                            plane_pass_rows += 1
                else:
                    # No plane cut
                    x_data.append(x_val)
                    y_data.append(y_val)
                    sigma_x_data.append(float(row['log_V_V0_unc']))
                    sigma_y_data.append(float(row['M_abs_unc']))

    # Convert to numpy arrays for calculations
    x = np.array(x_data)
    y = np.array(y_data)
    sigma_x = np.array(sigma_x_data)
    sigma_y = np.array(sigma_y_data)

    # ============================================================================
    # SECTION 1.5: RANDOM SAMPLING (if sample_size is specified)
    # ============================================================================
    N_filtered = len(x)

    if sample_size is not None and sample_size < N_filtered:
        np.random.seed(42)
        sample_indices = np.random.choice(N_filtered, size=sample_size, replace=False)

        x = x[sample_indices]
        y = y[sample_indices]
        sigma_x = sigma_x[sample_indices]
        sigma_y = sigma_y[sample_indices]

        print(f"\nRandom sampling:")
        print(f"  Filtered data size: {N_filtered}")
        print(f"  Requested sample size: {sample_size}")
        print(f"  Final sample size: {len(x)}")
    elif sample_size is not None and sample_size >= N_filtered:
        print(f"\nNote: Requested sample size ({sample_size}) >= filtered data size ({N_filtered})")
        print(f"  Using all {N_filtered} filtered galaxies")

    N_total = len(x)

    # Convert back to lists for JSON serialization
    x_data = x.tolist()
    y_data = y.tolist()
    sigma_x_data = sigma_x.tolist()
    sigma_y_data = sigma_y.tolist()

    # ============================================================================
    # SECTION 2: CREATE STAN DATA DICTIONARY
    # ============================================================================
    N_bins = 1
    bin_idx = [1] * N_total

    stan_data = {
        'N_bins': N_bins,
        'N_total': N_total,
        'x': x_data,
        'sigma_x': sigma_x_data,
        'y': y_data,
        'sigma_y': sigma_y_data,
        'haty_max': haty_max,
        'y_min': -23.0,
        'y_max': -15.0,
        'bin_idx': bin_idx
    }

    if plane_cut:
        stan_data['slope_plane'] = slope_plane
        stan_data['intercept_plane'] = intercept_plane  # keep original name (c1)
        if two_sided:
            stan_data['intercept_plane2'] = intercept_plane2  # new (c2)

    with open(data_output_file, 'w') as f:
        json.dump(stan_data, f, indent=2)

    # ============================================================================
    # SECTION 3: CALCULATE STANDARDIZATION AND LINEAR REGRESSION
    # ============================================================================
    mean_x = np.mean(x)
    sd_x = np.std(x, ddof=1)

    x_std = (x - mean_x) / sd_x

    slope_std, intercept_std = np.polyfit(x_std, y, deg=1)

    slope_orig = slope_std / sd_x
    intercept_orig = intercept_std - slope_std * mean_x / sd_x

    intercept_std_vec = [float(intercept_std)]

    # ============================================================================
    # SECTION 4: CREATE INITIAL CONDITIONS DICTIONARY
    # ============================================================================
    init_data = {
        'slope_std': float(slope_std),
        'intercept_std': intercept_std_vec,
        'sigma_int_tot_y': 0.05,
        'theta_int': np.pi/4
    }

    with open(init_output_file, 'w') as f:
        json.dump(init_data, f, indent=2)

    # ============================================================================
    # SECTION 5: PRINT SUMMARY STATISTICS
    # ============================================================================
    print(f"\nData conversion complete!")
    print(f"Stan data output file: {data_output_file}")
    print(f"Initial conditions output file: {init_output_file}")

    print(f"\nFiltering:")
    print(f"  Total rows in CSV: {total_rows}")
    print(f"  Rows with y < {haty_max}: {y_filtered_rows}")

    if plane_cut:
        if not two_sided:
            print(f"  Rows passing plane cut (bar_s * x + c1 <= y): {plane_pass_rows}")
            print(f"  Rows filtered out by plane cut: {y_filtered_rows - plane_pass_rows}")
            print(f"  Plane parameters: bar_s = {slope_plane}, c1 = {intercept_plane}")
        else:
            print(f"  Rows passing two-sided plane cut (c1 <= y - bar_s*x <= c2): {plane_pass_rows}")
            print(f"  Rows filtered out by plane cut: {y_filtered_rows - plane_pass_rows}")
            print(f"  Plane parameters: bar_s = {slope_plane}, c1 = {intercept_plane}, c2 = {intercept_plane2}")

    print(f"  Rows filtered out (by y cut only): {total_rows - y_filtered_rows}")
    print(f"  haty_max (selection threshold): {haty_max}")

    print(f"\nSummary:")
    print(f"  Number of redshift bins: {N_bins}")
    print(f"  Final sample size: {N_total}")

    if N_total > 0:
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
    else:
        print("\nWARNING: Final sample size is 0 after cuts.")


def plot_tf_data(json_file, output_file='tf_scatter_plot.png'):
    """
    Create scatter plot with error bars from TF data JSON file.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    x = np.array(data['x'])
    y = np.array(data['y'])
    sigma_x = np.array(data['sigma_x'])
    sigma_y = np.array(data['sigma_y'])

    N_total = data['N_total']
    haty_max = data.get('haty_max', None)
    slope_plane = data.get('slope_plane', None)
    intercept_plane = data.get('intercept_plane', None)
    intercept_plane2 = data.get('intercept_plane2', None)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y,
                fmt='o', markersize=3, alpha=0.6,
                elinewidth=0.5, capsize=0,
                label=f'N = {N_total}')

    if haty_max is not None:
        ax.axhline(y=haty_max, color='red', linestyle='--',
                   linewidth=2, alpha=0.8,
                   label=f'$\\hat{{y}}_{{\\rm max}}$ = {haty_max}')

    # Plot one or two parallel plane-cut boundaries if present
    if slope_plane is not None and intercept_plane is not None and len(x) > 0:
        x_range = np.array([np.min(x) - 0.1, np.max(x) + 0.1])

        y_plane1 = slope_plane * x_range + intercept_plane
        ax.plot(x_range, y_plane1, 'g--', linewidth=2, alpha=0.8,
                label=f'Plane cut 1: y = {slope_plane:.1f}x + {intercept_plane:.1f}')

        if intercept_plane2 is not None:
            y_plane2 = slope_plane * x_range + intercept_plane2
            ax.plot(x_range, y_plane2, 'g-.', linewidth=2, alpha=0.8,
                    label=f'Plane cut 2: y = {slope_plane:.1f}x + {intercept_plane2:.1f}')

    ax.set_xlabel(r'$\hat{x}$ = log($V_{\rm rot}/V_0$)', fontsize=12)
    ax.set_ylabel(r'$\hat{y}$ = $M_{\rm abs}$ (absolute magnitude)', fontsize=12)
    ax.set_title('Tully-Fisher Mock Data', fontsize=14, fontweight='bold')

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to: {output_file}")

    print(f"\nData summary:")
    print(f"  Number of galaxies: {N_total}")
    if len(x) > 0:
        print(f"  x range: [{np.min(x):.3f}, {np.max(x):.3f}]")
        print(f"  y range: [{np.min(y):.3f}, {np.max(y):.3f}]")
        print(f"  sigma_x: mean = {np.mean(sigma_x):.4f}, range = [{np.min(sigma_x):.4f}, {np.max(sigma_x):.4f}]")
        print(f"  sigma_y: mean = {np.mean(sigma_y):.4f}, range = [{np.min(sigma_y):.4f}, {np.max(sigma_y):.4f}]")
    if haty_max is not None:
        print(f"  haty_max: {haty_max}")
    if slope_plane is not None and intercept_plane is not None:
        print(f"  Plane cut 1: y = {slope_plane}x + {intercept_plane}")
        if intercept_plane2 is not None:
            print(f"  Plane cut 2: y = {slope_plane}x + {intercept_plane2}")


if __name__ == '__main__':
    input_csv = 'data/TF_mock_tophat-mag_input.csv'

    haty_max = -16

    plane_cut = True
    slope_plane = -8.5
    intercept_plane = -20.5     # c1 (lower oblique bound)
    intercept_plane2 = -19.1    # c2 (upper oblique bound); set to None for one-sided

    sample_size = 10000  # or None

    if sample_size is not None:
        output_json = f'TF_mock_input_n{sample_size}.json'
        init_json = f'TF_mock_init_n{sample_size}.json'
    else:
        output_json = 'TF_mock_input.json'
        init_json = 'TF_mock_init.json'

    process_tf_data(input_csv, output_json, init_json,
                    haty_max=haty_max, sample_size=sample_size,
                    plane_cut=plane_cut, slope_plane=slope_plane,
                    intercept_plane=intercept_plane, intercept_plane2=intercept_plane2)

    plot_output = output_json.replace('.json', '_plot.png')
    plot_tf_data(output_json, plot_output)