#!/usr/bin/env python3
"""
Convert DESI-DR1_TF_pv_cat_v15.fits to JSON format for base.stan

This script reads the DESI Tully-Fisher data and converts it to the format
expected by base.stan.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def process_desi_tf_data(fits_file, data_output_file, init_output_file, haty_max=-16,
                         plane_cut=False, slope_plane=None, intercept_plane=None, intercept_plane2=None):
    """
    Process DESI TF data: convert to Stan JSON format and create initial conditions.

    Parameters
    ----------
    fits_file : str
        Path to FITS file containing DESI TF data
    data_output_file : str
        Output JSON file for Stan data
    init_output_file : str
        Output JSON file for Stan initial conditions
    haty_max : float, optional
        Upper limit on absolute magnitude (default: -16)
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
    # SECTION 1: READ FITS DATA
    # ============================================================================
    print(f"Reading FITS file: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        
        # Extract velocity and magnitude data
        V_0p4R26 = data['V_0p4R26']
        V_0p4R26_ERR = data['V_0p4R26_ERR']
        R_ABSMAG_SB26 = data['R_ABSMAG_SB26']
        R_ABSMAG_SB26_ERR = data['R_ABSMAG_SB26_ERR']
    
    # Convert velocities to log velocities
    # Assuming V0 = 100 km/s as a reference (adjust if needed)
    V0 = 100.0
    
    # Filter out invalid data (NaN, inf, non-positive velocities)
    valid_mask = (
        np.isfinite(V_0p4R26) & 
        np.isfinite(V_0p4R26_ERR) & 
        np.isfinite(R_ABSMAG_SB26) & 
        np.isfinite(R_ABSMAG_SB26_ERR) &
        (V_0p4R26 > 0) &
        (V_0p4R26_ERR > 0)
    )
    
    V_0p4R26 = V_0p4R26[valid_mask]
    V_0p4R26_ERR = V_0p4R26_ERR[valid_mask]
    R_ABSMAG_SB26 = R_ABSMAG_SB26[valid_mask]
    R_ABSMAG_SB26_ERR = R_ABSMAG_SB26_ERR[valid_mask]
    
    total_rows = len(data)
    valid_rows = len(V_0p4R26)
    
    # Convert to log velocities: x = log(V / V0)
    x_all = np.log10(V_0p4R26 / V0)
    
    # Propagate uncertainties: sigma_x = sigma_V / (V * ln(10))
    sigma_x_all = V_0p4R26_ERR / (V_0p4R26 * np.log(10))
    
    # Magnitude data
    y_all = R_ABSMAG_SB26
    sigma_y_all = R_ABSMAG_SB26_ERR
    
    # ============================================================================
    # SECTION 2: APPLY SELECTION CUTS
    # ============================================================================
    x_data = []
    y_data = []
    sigma_x_data = []
    sigma_y_data = []
    
    # Track filtering statistics
    y_filtered_rows = 0
    plane_pass_rows = 0
    
    for i in range(len(x_all)):
        x_val = x_all[i]
        y_val = y_all[i]
        
        # Apply y upper limit (note: magnitudes: "brighter" is more negative)
        if y_val < haty_max:
            y_filtered_rows += 1
            
            if plane_cut:
                lower_bound = slope_plane * x_val + intercept_plane
                
                if not two_sided:
                    # One-sided: lower_bound <= y
                    if lower_bound <= y_val:
                        x_data.append(x_val)
                        y_data.append(y_val)
                        sigma_x_data.append(sigma_x_all[i])
                        sigma_y_data.append(sigma_y_all[i])
                        plane_pass_rows += 1
                else:
                    # Two-sided: lower_bound <= y <= min(haty_max, upper_bound)
                    upper_bound_oblique = slope_plane * x_val + intercept_plane2
                    upper_bound = min(haty_max, upper_bound_oblique)
                    
                    if (lower_bound <= y_val) and (y_val <= upper_bound):
                        x_data.append(x_val)
                        y_data.append(y_val)
                        sigma_x_data.append(sigma_x_all[i])
                        sigma_y_data.append(sigma_y_all[i])
                        plane_pass_rows += 1
            else:
                # No plane cut
                x_data.append(x_val)
                y_data.append(y_val)
                sigma_x_data.append(sigma_x_all[i])
                sigma_y_data.append(sigma_y_all[i])
    
    # Convert to numpy arrays for calculations
    x = np.array(x_data)
    y = np.array(y_data)
    sigma_x = np.array(sigma_x_data)
    sigma_y = np.array(sigma_y_data)
    
    N_total = len(x)
    
    # ============================================================================
    # SECTION 3: CREATE STAN DATA DICTIONARY
    # ============================================================================
    N_bins = 1
    
    # Calculate y_min and y_max based on selected sample
    if N_total > 0:
        y_min_data = float(np.min(y) - 0.5)
        y_max_data = float(np.max(y) + 0.5)
    else:
        y_min_data = -23.0
        y_max_data = -15.0
    
    stan_data = {
        'N_bins': N_bins,
        'N_total': N_total,
        'x': x_data,
        'sigma_x': sigma_x_data,
        'y': y_data,
        'sigma_y': sigma_y_data,
        'haty_max': haty_max,
        'y_min': y_min_data,
        'y_max': y_max_data
    }
    
    if plane_cut:
        stan_data['slope_plane'] = slope_plane
        stan_data['intercept_plane'] = intercept_plane  # keep original name (c1)
        if two_sided:
            stan_data['intercept_plane2'] = intercept_plane2  # new (c2)
    
    with open(data_output_file, 'w') as f:
        json.dump(stan_data, f, indent=2)
    
    # ============================================================================
    # SECTION 4: CALCULATE STANDARDIZATION AND LINEAR REGRESSION
    # ============================================================================
    if N_total > 0:
        mean_x = np.mean(x)
        sd_x = np.std(x, ddof=1)
        
        x_std = (x - mean_x) / sd_x
        
        slope_std, intercept_std = np.polyfit(x_std, y, deg=1)
        
        slope_orig = slope_std / sd_x
        intercept_orig = intercept_std - slope_std * mean_x / sd_x
        
        intercept_std_vec = [float(intercept_std)]
    else:
        slope_std = 0.0
        intercept_std_vec = [0.0]
        slope_orig = 0.0
        intercept_orig = 0.0
        mean_x = 0.0
        sd_x = 1.0
    
    # ============================================================================
    # SECTION 5: CREATE INITIAL CONDITIONS DICTIONARY
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
    # SECTION 6: PRINT SUMMARY STATISTICS
    # ============================================================================
    print(f"\nData conversion complete!")
    print(f"Stan data output file: {data_output_file}")
    print(f"Initial conditions output file: {init_output_file}")
    
    print(f"\nFiltering:")
    print(f"  Total rows in FITS: {total_rows}")
    print(f"  Valid rows (finite, positive velocities): {valid_rows}")
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
    
    print(f"  Rows filtered out (by y cut only): {valid_rows - y_filtered_rows}")
    print(f"  haty_max (selection threshold): {haty_max}")
    
    print(f"\nSummary:")
    print(f"  Number of redshift bins: {N_bins}")
    print(f"  Final sample size: {N_total}")
    
    if N_total > 0:
        print(f"\nData ranges:")
        print(f"  x (log(V/V0)): [{np.min(x):.3f}, {np.max(x):.3f}]")
        print(f"  y (R_ABSMAG_SB26): [{np.min(y):.3f}, {np.max(y):.3f}]")
        print(f"  sigma_x: {np.mean(sigma_x):.4f} (mean), range [{np.min(sigma_x):.4f}, {np.max(sigma_x):.4f}]")
        print(f"  sigma_y: {np.mean(sigma_y):.4f} (mean), range [{np.min(sigma_y):.4f}, {np.max(sigma_y):.4f}]")
        
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
    
    # Return data for plotting
    return x_all, y_all, sigma_x_all, sigma_y_all, x, y, sigma_x, sigma_y


def plot_desi_tf_data(x_all, y_all, sigma_x_all, sigma_y_all, 
                      x_selected, y_selected, sigma_x_selected, sigma_y_selected,
                      haty_max=None, slope_plane=None, intercept_plane=None, intercept_plane2=None,
                      output_file='desi_tf_scatter_plot.png'):
    """
    Create scatter plot showing complete sample (low alpha) and selected sample (high alpha).
    
    Parameters
    ----------
    x_all, y_all, sigma_x_all, sigma_y_all : array-like
        Complete sample data
    x_selected, y_selected, sigma_x_selected, sigma_y_selected : array-like
        Selected sample data after cuts
    haty_max : float, optional
        Upper magnitude limit
    slope_plane, intercept_plane, intercept_plane2 : float, optional
        Plane cut parameters
    output_file : str
        Output filename for plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot complete sample with low alpha
    ax.errorbar(x_all, y_all, xerr=sigma_x_all, yerr=sigma_y_all,
                fmt='o', markersize=2, alpha=0.2, color='gray',
                elinewidth=0.3, capsize=0,
                label=f'Complete sample (N = {len(x_all)})')
    
    # Plot selected sample with high alpha
    ax.errorbar(x_selected, y_selected, xerr=sigma_x_selected, yerr=sigma_y_selected,
                fmt='o', markersize=3, alpha=0.8, color='blue',
                elinewidth=0.5, capsize=0,
                label=f'Selected sample (N = {len(x_selected)})')
    
    if haty_max is not None:
        ax.axhline(y=haty_max, color='red', linestyle='--',
                   linewidth=2, alpha=0.8,
                   label=f'$\\hat{{y}}_{{\\rm max}}$ = {haty_max}')
    
    # Plot one or two parallel plane-cut boundaries if present
    if slope_plane is not None and intercept_plane is not None and len(x_all) > 0:
        x_range = np.array([np.min(x_all) - 0.1, np.max(x_all) + 0.1])
        
        y_plane1 = slope_plane * x_range + intercept_plane
        ax.plot(x_range, y_plane1, 'g--', linewidth=2, alpha=0.8,
                label=f'Plane cut 1: y = {slope_plane:.1f}x + {intercept_plane:.1f}')
        
        if intercept_plane2 is not None:
            y_plane2 = slope_plane * x_range + intercept_plane2
            ax.plot(x_range, y_plane2, 'g-.', linewidth=2, alpha=0.8,
                    label=f'Plane cut 2: y = {slope_plane:.1f}x + {intercept_plane2:.1f}')
    
    ax.set_xlabel(r'$\hat{x}$ = log($V_{0.4R26}/V_0$)', fontsize=12)
    ax.set_ylabel(r'$\hat{y}$ = $R\_ABSMAG\_SB26$ (absolute magnitude)', fontsize=12)
    ax.set_title('DESI DR1 Tully-Fisher Data', fontsize=14, fontweight='bold')
    
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to: {output_file}")
    
    print(f"\nPlot summary:")
    print(f"  Complete sample: {len(x_all)} galaxies")
    print(f"  Selected sample: {len(x_selected)} galaxies")
    if len(x_all) > 0:
        print(f"  Complete x range: [{np.min(x_all):.3f}, {np.max(x_all):.3f}]")
        print(f"  Complete y range: [{np.min(y_all):.3f}, {np.max(y_all):.3f}]")
    if len(x_selected) > 0:
        print(f"  Selected x range: [{np.min(x_selected):.3f}, {np.max(x_selected):.3f}]")
        print(f"  Selected y range: [{np.min(y_selected):.3f}, {np.max(y_selected):.3f}]")


if __name__ == '__main__':
    input_fits = 'data/DESI-DR1_TF_pv_cat_v15.fits'
    
    haty_max = -17.5
    
    plane_cut = True
    slope_plane = -6.5
    intercept_plane = -20.5     # c1 (lower oblique bound)
    intercept_plane2 = -18.5    # c2 (upper oblique bound); set to None for one-sided
    
    output_json = 'DESI_TF_input.json'
    init_json = 'DESI_TF_init.json'
    
    # Process data and get both complete and selected samples
    x_all, y_all, sigma_x_all, sigma_y_all, x_sel, y_sel, sigma_x_sel, sigma_y_sel = process_desi_tf_data(
        input_fits, output_json, init_json,
        haty_max=haty_max,
        plane_cut=plane_cut, 
        slope_plane=slope_plane,
        intercept_plane=intercept_plane, 
        intercept_plane2=intercept_plane2
    )
    
    # Create plot showing both complete and selected samples
    plot_output = output_json.replace('.json', '_plot.png')
    plot_desi_tf_data(
        x_all, y_all, sigma_x_all, sigma_y_all,
        x_sel, y_sel, sigma_x_sel, sigma_y_sel,
        haty_max=haty_max,
        slope_plane=slope_plane,
        intercept_plane=intercept_plane,
        intercept_plane2=intercept_plane2,
        output_file=plot_output
    )
