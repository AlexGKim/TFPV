#!/usr/bin/env python3
"""
Convert DESI-DR1_TF_pv_cat_v15.fits to JSON format for tophat.stan

This script reads the DESI Tully-Fisher data and converts it to the format
expected by tophat.stan.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def process_desi_tf_data(
    fits_file,
    data_output_file,
    init_output_file,
    haty_max=-16,
    haty_min=-1.0e9,
    plane_cut=False,
    slope_plane=None,
    intercept_plane=None,
    intercept_plane2=None,
    n_objects=None,
    random_seed=None,
    *,
    z_col="Z_DESI",
    z_col_candidates=(
        "zobs",
        "ZOBS",
        "Z",
        "ZHELIO",
        "Z_CMB",
        "ZDESI",
        "ZTRUE",
        "Z_DESI",
    ),
    z_obs_min=None,  # <<< NEW: Minimum redshift for inclusion
):
    """
    Process DESI TF data: convert to Stan JSON format and create initial conditions.
    (Modified to correctly load/propagate z_obs and apply an optional redshift cut.)
    """

    # Validate plane cut parameters
    if plane_cut and (slope_plane is None or intercept_plane is None):
        raise ValueError(
            "slope_plane and intercept_plane must be provided when plane_cut=True"
        )

    # If two-sided, enforce ordering c1 < c2
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
        names = set(data.dtype.names or ())

        # Resolve redshift column name
        if z_col in names:
            z_col_use = z_col
        else:
            z_col_use = None
            for cand in z_col_candidates:
                if cand in names:
                    z_col_use = cand
                    break
            if z_col_use is None:
                raise ValueError(
                    f"Could not find a redshift column. Tried z_col={z_col!r} and candidates "
                    f"{z_col_candidates}. Available columns include: {sorted(list(names))[:30]} ..."
                )

        # Extract velocity, magnitude, and redshift data
        V_0p4R26 = np.asarray(data["V_0p4R26"], dtype=float)
        V_0p4R26_ERR = np.asarray(data["V_0p4R26_ERR"], dtype=float)
        R_ABSMAG_SB26 = np.asarray(data["R_ABSMAG_SB26"], dtype=float)
        if "R_ABSMAG_SB26_ERR" in names:
            R_ABSMAG_SB26_ERR = np.asarray(data["R_ABSMAG_SB26_ERR"], dtype=float)
        else:
            print("Warning: R_ABSMAG_SB26_ERR absent; falling back to R_MAG_SB26_ERR")
            R_ABSMAG_SB26_ERR = np.asarray(data["R_MAG_SB26_ERR"], dtype=float)
        z_all_raw = np.asarray(data[z_col_use], dtype=float)

    total_rows = len(V_0p4R26)

    # Convert velocities to log velocities
    V0 = 100.0  # km/s reference

    # Filter out invalid data (NaN, inf, non‑positive velocities, etc.)
    valid_mask = (
        np.isfinite(V_0p4R26)
        & np.isfinite(V_0p4R26_ERR)
        & np.isfinite(R_ABSMAG_SB26)
        & np.isfinite(R_ABSMAG_SB26_ERR)
        & np.isfinite(z_all_raw)
        & (V_0p4R26 > 0)
        & (V_0p4R26_ERR > 0)
        & (R_ABSMAG_SB26_ERR >= 0)
    )

    V_0p4R26 = V_0p4R26[valid_mask]
    V_0p4R26_ERR = V_0p4R26_ERR[valid_mask]
    R_ABSMAG_SB26 = R_ABSMAG_SB26[valid_mask]
    R_ABSMAG_SB26_ERR = R_ABSMAG_SB26_ERR[valid_mask]
    z_all_raw = z_all_raw[valid_mask]

    valid_rows = len(V_0p4R26)

    # Convert to log velocities: x = log10(V / V0)
    x_all = np.log10(V_0p4R26 / V0)

    # Propagate uncertainties: sigma_x = sigma_V / (V * ln(10))
    sigma_x_all = V_0p4R26_ERR / (V_0p4R26 * np.log(10))

    # Magnitude data
    y_all = R_ABSMAG_SB26
    sigma_y_all = R_ABSMAG_SB26_ERR

    # Redshift data (aligned to x_all/y_all by construction)
    zobs_all = z_all_raw

    # ============================================================================
    # SECTION 2: APPLY SELECTION CUTS (NOW INCLUDES haty_min AND z_obs_min)
    # ============================================================================
    x_data, y_data, sigma_x_data, sigma_y_data, z_data = [], [], [], [], []

    # Track filtering statistics
    y_filtered_rows = 0
    z_filtered_rows = 0
    plane_pass_rows = 0

    for i in range(len(x_all)):
        x_val = x_all[i]
        y_val = y_all[i]

        # Apply BOTH y limits (magnitudes: "brighter" is more negative)
        if (y_val < haty_max) and (y_val > haty_min):
            y_filtered_rows += 1

            # ---- NEW REDSHIFT CUT ----
            if (z_obs_min is not None) and (zobs_all[i] <= z_obs_min):
                continue
            z_filtered_rows += 1
            # ---------------------------

            if plane_cut:
                lower_bound_oblique = slope_plane * x_val + intercept_plane
                lower_bound = max(haty_min, lower_bound_oblique)

                if not two_sided:
                    # One‑sided: lower_bound <= y
                    if lower_bound <= y_val:
                        x_data.append(x_val)
                        y_data.append(y_val)
                        sigma_x_data.append(sigma_x_all[i])
                        sigma_y_data.append(sigma_y_all[i])
                        z_data.append(zobs_all[i])
                        plane_pass_rows += 1
                else:
                    # Two‑sided: lower_bound <= y <= min(haty_max, upper_bound_oblique)
                    upper_bound_oblique = slope_plane * x_val + intercept_plane2
                    upper_bound = min(haty_max, upper_bound_oblique)

                    if (lower_bound <= y_val) and (y_val <= upper_bound):
                        x_data.append(x_val)
                        y_data.append(y_val)
                        sigma_x_data.append(sigma_x_all[i])
                        sigma_y_data.append(sigma_y_all[i])
                        z_data.append(zobs_all[i])
                        plane_pass_rows += 1
            else:
                # No plane cut (just the y‑range and optional redshift cut)
                x_data.append(x_val)
                y_data.append(y_val)
                sigma_x_data.append(sigma_x_all[i])
                sigma_y_data.append(sigma_y_all[i])
                z_data.append(zobs_all[i])

    # Convert to numpy arrays for calculations
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    sigma_x = np.array(sigma_x_data, dtype=float)
    sigma_y = np.array(sigma_y_data, dtype=float)
    z_obs = np.array(z_data, dtype=float)

    N_after_cuts = len(x)

    # Subsample if n_objects is specified
    if n_objects is not None and n_objects < N_after_cuts:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(N_after_cuts, size=n_objects, replace=False)
        idx.sort()
        x = x[idx]
        y = y[idx]
        sigma_x = sigma_x[idx]
        sigma_y = sigma_y[idx]
        z_obs = z_obs[idx]
        print(
            f"  Subsampled from {N_after_cuts} to {n_objects} objects (random_seed={random_seed})"
        )

    # Convert back to lists for JSON serialization
    x_data = x.tolist()
    y_data = y.tolist()
    sigma_x_data = sigma_x.tolist()
    sigma_y_data = sigma_y.tolist()
    z_obs_data = z_obs.tolist()

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

    mu_y_TF = float(np.mean(y)) if N_total > 0 else 0.0
    tau = (
        1.5 * float(np.std(y, ddof=1)) if N_total > 1 else 1.0
    )

    stan_data = {
        "N_bins": N_bins,
        "N_total": N_total,
        "x": x_data,
        "sigma_x": sigma_x_data,
        "y": y_data,
        "sigma_y": sigma_y_data,
        "haty_max": float(haty_max),
        "haty_min": float(haty_min),
        "y_min": float(haty_min) - 0.5,
        "y_max": float(haty_max) + 1,
        "mu_y_TF": mu_y_TF,
        "tau": tau,
        "z_obs": z_obs_data,  # now defined, aligned, and JSON‑serializable
        "z_obs_min": float(z_obs_min) if z_obs_min is not None else None,
    }

    if plane_cut:
        stan_data["slope_plane"] = float(slope_plane)
        stan_data["intercept_plane"] = float(intercept_plane)
        if two_sided:
            stan_data["intercept_plane2"] = float(intercept_plane2)

    with open(data_output_file, "w") as f:
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
        intercept_std = 0.0
        intercept_std_vec = [0.0]
        slope_orig = 0.0
        intercept_orig = 0.0
        mean_x = 0.0
        sd_x = 1.0
        x_std = np.array([])

    # ============================================================================
    # SECTION 5: CREATE INITIAL CONDITIONS DICTIONARY
    # ============================================================================
    init_data = {
        "slope_std": float(slope_std),
        "intercept_std": intercept_std_vec,
        "slope_orig": float(slope_orig),
        "intercept_orig": float(intercept_orig),
        "sigma_int_x": 0.1,
        "sigma_int_y": 0.1,
        "mean_x": float(mean_x),
        "sd_x": float(sd_x),
    }

    with open(init_output_file, "w") as f:
        json.dump(init_data, f, indent=2)

    # ============================================================================
    # SECTION 6: PRINT SUMMARY STATISTICS
    # ============================================================================
    print("\nData conversion complete!")
    print(f"Stan data output file: {data_output_file}")
    print(f"Initial conditions output file: {init_output_file}")

    print("\nFiltering:")
    print(f"  Total rows in FITS: {total_rows}")
    print(f"  Valid rows (finite, positive velocities, finite z): {valid_rows}")
    print(f"  Rows with {haty_min} < y < {haty_max}: {y_filtered_rows}")

    if z_obs_min is not None:
        print(f"  Rows with z_obs > {z_obs_min}: {z_filtered_rows}")

    if plane_cut:
        if not two_sided:
            print(
                f"  Rows passing plane cut (max({haty_min}, bar_s*x+c1) <= y): {plane_pass_rows}"
            )
            print(
                f"  Rows filtered out by plane cut: {y_filtered_rows - plane_pass_rows}"
            )
            print(
                f"  Plane parameters: bar_s = {slope_plane}, c1 = {intercept_plane}"
            )
        else:
            print(f"  Rows passing two‑sided plane cut: {plane_pass_rows}")
            print(
                f"  Rows filtered out by plane cut: {y_filtered_rows - plane_pass_rows}"
            )
            print(
                f"  Plane parameters: bar_s = {slope_plane}, c1 = {intercept_plane}, c2 = {intercept_plane2}"
            )

    print(f"  Rows filtered out (by y cut only): {valid_rows - y_filtered_rows}")
    print(f"  haty_max (selection upper threshold): {haty_max}")
    print(f"  haty_min (selection lower threshold): {haty_min}")

    print("\nSummary:")
    print(f"  Number of redshift bins: {N_bins}")
    print(f"  Final sample size: {N_total}")
    if N_total > 0:
        print(f"  z_obs range: [{np.min(z_obs):.6f}, {np.max(z_obs):.6f}]")

    # Return data for plotting (plot doesn't use z, but we return it for completeness)
    return (
        x_all,
        y_all,
        sigma_x_all,
        sigma_y_all,
        zobs_all,
        x,
        y,
        sigma_x,
        sigma_y,
        z_obs,
    )


def plot_desi_tf_data(
    x_all,
    y_all,
    sigma_x_all,
    sigma_y_all,
    x_selected,
    y_selected,
    sigma_x_selected,
    sigma_y_selected,
    haty_max=None,
    haty_min=None,
    slope_plane=None,
    intercept_plane=None,
    intercept_plane2=None,
    output_file="desi_tf_scatter_plot.png",
):
    """
    Create scatter plot showing complete sample (low alpha) and selected sample (high alpha).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot complete sample with low alpha
    ax.errorbar(
        x_all,
        y_all,
        xerr=sigma_x_all,
        yerr=sigma_y_all,
        fmt="o",
        markersize=2,
        alpha=0.2,
        color="gray",
        elinewidth=0.3,
        capsize=0,
        label=f"Complete sample (N = {len(x_all)})",
    )

    # Plot selected sample with high alpha
    ax.errorbar(
        x_selected,
        y_selected,
        xerr=sigma_x_selected,
        yerr=sigma_y_selected,
        fmt="o",
        markersize=3,
        alpha=0.8,
        color="blue",
        elinewidth=0.5,
        capsize=0,
        label=f"Selected sample (N = {len(x_selected)})",
    )

    if haty_max is not None:
        ax.axhline(
            y=haty_max,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"$\\hat{{y}}_{{\\rm max}}$ = {haty_max}",
        )
    if haty_min is not None:
        ax.axhline(
            y=haty_min,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"$\\hat{{y}}_{{\\rm min}}$ = {haty_min}",
        )

    # Plot one or two parallel plane‑cut boundaries if present
    if slope_plane is not None and intercept_plane is not None and len(x_all) > 0:
        x_range = np.array([np.min(x_all) - 0.1, np.max(x_all) + 0.1])

        y_plane1 = slope_plane * x_range + intercept_plane
        ax.plot(
            x_range,
            y_plane1,
            "g--",
            linewidth=2,
            alpha=0.8,
            label=f"Plane cut 1: y = {slope_plane:.1f}x + {intercept_plane:.1f}",
        )

        if intercept_plane2 is not None:
            y_plane2 = slope_plane * x_range + intercept_plane2
            ax.plot(
                x_range,
                y_plane2,
                "g-.",
                linewidth=2,
                alpha=0.8,
                label=f"Plane cut 2: y = {slope_plane:.1f}x + {intercept_plane2:.1f}",
            )

    ax.set_xlabel(r"$\hat{x}$ = log($V_{0.4R26}/V_0$)", fontsize=12)
    ax.set_ylabel(
        r"$\hat{y}$ = $R\_ABSMAG\_SB26$ (absolute magnitude)",
        fontsize=12,
    )
    ax.set_title("DESI DR1 Tully‑Fisher Data", fontsize=14, fontweight="bold")

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {output_file}")

    print("\nPlot summary:")
    print(f"  Complete sample: {len(x_all)} galaxies")
    print(f"  Selected sample: {len(x_selected)} galaxies")
    if len(x_all) > 0:
        print(f"  Complete x range: [{np.min(x_all):.3f}, {np.max(x_all):.3f}]")
        print(f"  Complete y range: [{np.min(y_all):.3f}, {np.max(y_all):.3f}]")
    if len(x_selected) > 0:
        print(f"  Selected x range: [{np.min(x_selected):.3f}, {np.max(x_selected):.3f}]")
        print(f"  Selected y range: [{np.min(y_selected):.3f}, {np.max(y_selected):.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare DESI TF data for Stan.')
    parser.add_argument('--run', default=None,
                        help='Run name; outputs go to output/<run>/ with standard filenames')
    parser.add_argument('--input', default='data/DESI-DR1_TF_pv_cat_v15.fits',
                        help='Input FITS file')
    parser.add_argument('--output', default=None,
                        help='Output data JSON file (default: DESI_input.json or output/<run>/input.json)')
    parser.add_argument('--init', default=None,
                        help='Output init JSON file (default: DESI_init.json or output/<run>/init.json)')
    parser.add_argument('--plot', default=None,
                        help='Output plot PNG file (default: DESI_input.png or output/<run>/data.png)')
    parser.add_argument('--haty_max', type=float, default=-19.0,
                        help='Upper apparent magnitude selection limit')
    parser.add_argument('--haty_min', type=float, default=-22.0,
                        help='Lower apparent magnitude selection limit')
    parser.add_argument('--n_objects', type=int, default=None,
                        help='Subsample size (None for all)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for reproducible subsampling')
    parser.add_argument('--z_obs_min', type=float, default=0.01,
                        help='Minimum redshift')
    parser.add_argument('--slope_plane', type=float, default=-6.5,
                        help='Slope of oblique selection cut')
    parser.add_argument('--intercept_plane', type=float, default=-20.5,
                        help='Intercept of lower oblique cut (c1)')
    parser.add_argument('--intercept_plane2', type=float, default=-18.5,
                        help='Intercept of upper oblique cut (c2)')
    args = parser.parse_args()

    if args.run is not None:
        run_dir = os.path.join('output', args.run)
        os.makedirs(run_dir, exist_ok=True)
        output_json = args.output or os.path.join(run_dir, 'input.json')
        init_json   = args.init   or os.path.join(run_dir, 'init.json')
        plot_file   = args.plot   or os.path.join(run_dir, 'data.png')
        config = {
            'source': args.input,
            'haty_max': args.haty_max,
            'haty_min': args.haty_min,
            'n_objects': args.n_objects,
            'random_seed': args.random_seed,
            'z_obs_min': args.z_obs_min,
            'slope_plane': args.slope_plane,
            'intercept_plane': args.intercept_plane,
            'intercept_plane2': args.intercept_plane2,
        }
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config written to {os.path.join(run_dir, 'config.json')}")
    else:
        output_json = args.output or 'DESI_input.json'
        init_json   = args.init   or 'DESI_init.json'
        plot_file   = args.plot   or 'DESI_input.png'

    # Process data and get both complete and selected samples
    (
        x_all,
        y_all,
        sigma_x_all,
        sigma_y_all,
        z_all,
        x_sel,
        y_sel,
        sigma_x_sel,
        sigma_y_sel,
        z_sel,
    ) = process_desi_tf_data(
        args.input,
        output_json,
        init_json,
        haty_max=args.haty_max,
        haty_min=args.haty_min,
        plane_cut=True,
        slope_plane=args.slope_plane,
        intercept_plane=args.intercept_plane,
        intercept_plane2=args.intercept_plane2,
        n_objects=args.n_objects,
        random_seed=args.random_seed,
        z_obs_min=args.z_obs_min,
    )

    # Create plot showing both complete and selected samples
    plot_desi_tf_data(
        x_all,
        y_all,
        sigma_x_all,
        sigma_y_all,
        x_sel,
        y_sel,
        sigma_x_sel,
        sigma_y_sel,
        haty_max=args.haty_max,
        haty_min=args.haty_min,
        slope_plane=args.slope_plane,
        intercept_plane=args.intercept_plane,
        intercept_plane2=args.intercept_plane2,
        output_file=plot_file,
    )