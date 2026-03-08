#!/usr/bin/env python3
"""
Convert TF_extended_AbacusSummit_*.fits mocks to JSON format for tophat.stan / normal.stan.

Each file in --dir matching TF_extended_AbacusSummit_base_c???_ph???_r???_z0.11.fits is
processed into its own output/<run>/ subdirectory, where <run> is derived from the filename
(e.g. c000_ph000_r000).

Use --one to process only the first matching file (for debugging).
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def process_fullmocks_tf_data(
    fits_file,
    data_output_file,
    init_output_file,
    haty_max=-19.0,
    haty_min=-22.5,
    plane_cut=False,
    slope_plane=None,
    intercept_plane=None,
    intercept_plane2=None,
    n_objects=None,
    random_seed=None,
    z_obs_min=None,
):
    """
    Process a TF_extended AbacusSummit mock FITS file into Stan JSON + init JSON.

    Columns used:
      x        = LOGVROT - 2.0   (log10(V_rot / 100 km/s))
      sigma_x  = LOGVROT_ERR
      y        = R_ABSMAG_SB26
      sigma_y  = R_ABSMAG_SB26_ERR
      z_obs    = ZOBS

    Only rows with MAIN=True are considered.
    Returns (x_main, y_main, sigma_x_main, sigma_y_main,
             x_sel,  y_sel,  sigma_x_sel,  sigma_y_sel)
    for plotting.
    """

    if plane_cut and (slope_plane is None or intercept_plane is None):
        raise ValueError(
            "slope_plane and intercept_plane must be provided when plane_cut=True"
        )

    two_sided = plane_cut and (intercept_plane2 is not None)
    if two_sided and not (intercept_plane < intercept_plane2):
        raise ValueError(
            f"For a two-sided parallel cut, require intercept_plane < intercept_plane2. "
            f"Got {intercept_plane} and {intercept_plane2}."
        )

    # =========================================================================
    # SECTION 1: READ FITS DATA
    # =========================================================================
    print(f"Reading FITS file: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        total_rows = len(data)

        main_mask = np.asarray(data["MAIN"], dtype=bool)
        data_main = data[main_mask]

    print(f"  Total rows: {total_rows}  |  MAIN=True: {np.sum(main_mask)}")

    logvrot      = np.asarray(data_main["LOGVROT"],         dtype=float)
    logvrot_err  = np.asarray(data_main["LOGVROT_ERR"],     dtype=float)
    absmag       = np.asarray(data_main["R_ABSMAG_SB26"],   dtype=float)
    absmag_err   = np.asarray(data_main["R_ABSMAG_SB26_ERR"], dtype=float)
    zobs         = np.asarray(data_main["ZOBS"],            dtype=float)

    # Convert: x = log10(V / 100 km/s) = LOGVROT - 2;  sigma_x = LOGVROT_ERR
    x_raw       = logvrot - 2.0
    sigma_x_raw = logvrot_err
    y_raw       = absmag
    sigma_y_raw = absmag_err

    # =========================================================================
    # SECTION 2: VALIDITY FILTER
    # =========================================================================
    valid = (
        np.isfinite(x_raw)
        & np.isfinite(sigma_x_raw)
        & np.isfinite(y_raw)
        & np.isfinite(sigma_y_raw)
        & np.isfinite(zobs)
        & (logvrot > 0)          # exclude LOGVROT == 0 (unphysical/missing)
        & (sigma_x_raw > 0)
        & (sigma_y_raw >= 0)
    )

    x_all       = x_raw[valid]
    sigma_x_all = sigma_x_raw[valid]
    y_all       = y_raw[valid]
    sigma_y_all = sigma_y_raw[valid]
    zobs_all    = zobs[valid]

    valid_rows = len(x_all)
    print(f"  Valid rows after MAIN + validity filter: {valid_rows}")

    # =========================================================================
    # SECTION 3: SELECTION CUTS
    # =========================================================================
    x_data, y_data, sigma_x_data, sigma_y_data, z_data = [], [], [], [], []

    y_filtered_rows = 0
    z_filtered_rows = 0
    plane_pass_rows = 0

    for i in range(len(x_all)):
        x_val = x_all[i]
        y_val = y_all[i]

        if not (haty_min < y_val < haty_max):
            continue
        y_filtered_rows += 1

        if z_obs_min is not None and zobs_all[i] <= z_obs_min:
            continue
        z_filtered_rows += 1

        if plane_cut:
            lower_bound_oblique = slope_plane * x_val + intercept_plane
            lower_bound = max(haty_min, lower_bound_oblique)

            if not two_sided:
                if lower_bound <= y_val:
                    x_data.append(x_val)
                    y_data.append(y_val)
                    sigma_x_data.append(sigma_x_all[i])
                    sigma_y_data.append(sigma_y_all[i])
                    z_data.append(zobs_all[i])
                    plane_pass_rows += 1
            else:
                upper_bound_oblique = slope_plane * x_val + intercept_plane2
                upper_bound = min(haty_max, upper_bound_oblique)
                if lower_bound <= y_val <= upper_bound:
                    x_data.append(x_val)
                    y_data.append(y_val)
                    sigma_x_data.append(sigma_x_all[i])
                    sigma_y_data.append(sigma_y_all[i])
                    z_data.append(zobs_all[i])
                    plane_pass_rows += 1
        else:
            x_data.append(x_val)
            y_data.append(y_val)
            sigma_x_data.append(sigma_x_all[i])
            sigma_y_data.append(sigma_y_all[i])
            z_data.append(zobs_all[i])

    # Full selected sample (used for plotting)
    x_sel       = np.array(x_data,       dtype=float)
    y_sel       = np.array(y_data,       dtype=float)
    sigma_x_sel = np.array(sigma_x_data, dtype=float)
    sigma_y_sel = np.array(sigma_y_data, dtype=float)
    z_obs_sel   = np.array(z_data,       dtype=float)

    N_after_cuts = len(x_sel)

    # Subsample for Stan (n_objects controls how many go into input.json)
    if n_objects is not None and n_objects < N_after_cuts:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(N_after_cuts, size=n_objects, replace=False)
        idx.sort()
        x       = x_sel[idx]
        y       = y_sel[idx]
        sigma_x = sigma_x_sel[idx]
        sigma_y = sigma_y_sel[idx]
        z_obs   = z_obs_sel[idx]
        print(f"  Subsampled from {N_after_cuts} to {n_objects} objects for Stan (seed={random_seed})")
    else:
        x, y, sigma_x, sigma_y, z_obs = x_sel, y_sel, sigma_x_sel, sigma_y_sel, z_obs_sel

    N_total = len(x)

    # =========================================================================
    # SECTION 4: BUILD STAN DATA DICT
    # =========================================================================
    mu_y_TF = float(np.mean(y)) if N_total > 0 else 0.0
    tau     = float(1.5 * np.std(y, ddof=1)) if N_total > 1 else 1.0

    stan_data = {
        "N_bins":   1,
        "N_total":  N_total,
        "x":        x.tolist(),
        "sigma_x":  sigma_x.tolist(),
        "y":        y.tolist(),
        "sigma_y":  sigma_y.tolist(),
        "haty_max": float(haty_max),
        "haty_min": float(haty_min),
        "y_min":    float(haty_min) - 0.5,
        "y_max":    float(haty_max) + 1.0,
        "mu_y_TF":  mu_y_TF,
        "tau":      tau,
        "z_obs":    z_obs.tolist(),
        "z_obs_min": float(z_obs_min) if z_obs_min is not None else None,
    }
    if plane_cut:
        stan_data["slope_plane"]     = float(slope_plane)
        stan_data["intercept_plane"] = float(intercept_plane)
        if two_sided:
            stan_data["intercept_plane2"] = float(intercept_plane2)

    with open(data_output_file, "w") as f:
        json.dump(stan_data, f, indent=2)

    # =========================================================================
    # SECTION 5: INITIAL CONDITIONS
    # =========================================================================
    if N_total > 0:
        mean_x = np.mean(x)
        sd_x   = np.std(x, ddof=1)
        x_std  = (x - mean_x) / sd_x
        slope_std, intercept_std = np.polyfit(x_std, y, deg=1)
        slope_orig    = slope_std / sd_x
        intercept_orig = intercept_std - slope_std * mean_x / sd_x
    else:
        mean_x = 0.0;  sd_x = 1.0
        slope_std = 0.0;  intercept_std = 0.0
        slope_orig = 0.0; intercept_orig = 0.0

    init_data = {
        "slope_std":     float(slope_std),
        "intercept_std": [float(intercept_std)],
        "slope_orig":    float(slope_orig),
        "intercept_orig": float(intercept_orig),
        "sigma_int_x":   0.1,
        "sigma_int_y":   0.1,
        "mean_x":        float(mean_x),
        "sd_x":          float(sd_x),
    }

    with open(init_output_file, "w") as f:
        json.dump(init_data, f, indent=2)

    # =========================================================================
    # SECTION 6: SUMMARY
    # =========================================================================
    print(f"\nData output:   {data_output_file}")
    print(f"Init output:   {init_output_file}")
    print(f"\nFiltering:")
    print(f"  MAIN rows (valid):                 {valid_rows}")
    print(f"  After magnitude cut [{haty_min}, {haty_max}]: {y_filtered_rows}")
    if z_obs_min is not None:
        print(f"  After redshift cut z > {z_obs_min}:        {z_filtered_rows}")
    if plane_cut:
        label = "two-sided" if two_sided else "one-sided"
        print(f"  After {label} plane cut:               {plane_pass_rows}")
    print(f"  After selection cuts:              {N_after_cuts}")
    print(f"  Final sample (plotted + Stan):     {N_total}")
    if N_total > 0:
        print(f"  z_obs range: [{np.min(z_obs):.4f}, {np.max(z_obs):.4f}]")

    return x_all, y_all, sigma_x_all, sigma_y_all, x, y, sigma_x, sigma_y


def plot_fullmocks_tf_data(
    x_all, y_all, sigma_x_all, sigma_y_all,
    x_selected, y_selected, sigma_x_selected, sigma_y_selected,
    haty_max=None,
    haty_min=None,
    slope_plane=None,
    intercept_plane=None,
    intercept_plane2=None,
    output_file="fullmocks_data.png",
    title="AbacusSummit TF Mock",
):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.errorbar(
        x_all, y_all, xerr=sigma_x_all, yerr=sigma_y_all,
        fmt="o", markersize=2, alpha=0.2, color="gray",
        elinewidth=0.3, capsize=0,
        label=f"MAIN sample (N = {len(x_all)})",
    )
    ax.errorbar(
        x_selected, y_selected, xerr=sigma_x_selected, yerr=sigma_y_selected,
        fmt="o", markersize=3, alpha=0.8, color="blue",
        elinewidth=0.5, capsize=0,
        label=f"Stan sample (N = {len(x_selected)})",
    )

    if haty_max is not None:
        ax.axhline(haty_max, color="red", linestyle="--", linewidth=2, alpha=0.8,
                   label=f"$\\hat{{y}}_{{\\rm max}}$ = {haty_max}")
    if haty_min is not None:
        ax.axhline(haty_min, color="orange", linestyle="--", linewidth=2, alpha=0.8,
                   label=f"$\\hat{{y}}_{{\\rm min}}$ = {haty_min}")

    if slope_plane is not None and intercept_plane is not None and len(x_all) > 0:
        x_range = np.array([np.min(x_all) - 0.1, np.max(x_all) + 0.1])
        ax.plot(x_range, slope_plane * x_range + intercept_plane,
                "g--", linewidth=2, alpha=0.8,
                label=f"Plane cut 1: y = {slope_plane:.1f}x + {intercept_plane:.1f}")
        if intercept_plane2 is not None:
            ax.plot(x_range, slope_plane * x_range + intercept_plane2,
                    "g-.", linewidth=2, alpha=0.8,
                    label=f"Plane cut 2: y = {slope_plane:.1f}x + {intercept_plane2:.1f}")

    ax.set_xlabel(r"$\hat{x}$ = log($V_{\rm rot}$/100 km/s)", fontsize=12)
    ax.set_ylabel(r"$\hat{y}$ = R_ABSMAG_SB26", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_file}")


def run_name_from_path(fits_path):
    """Derive a short run name from the FITS filename.

    TF_extended_AbacusSummit_base_c000_ph000_r000_z0.11.fits -> c000_ph000_r000
    """
    stem = os.path.splitext(os.path.basename(fits_path))[0]
    m = re.search(r"(c\d+_ph\d+_r\d+)", stem)
    return m.group(1) if m else stem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare AbacusSummit TF mock FITS files for Stan."
    )
    parser.add_argument(
        "--dir",
        default="/global/cfs/cdirs/desi/science/td/pv/mocks/TF_mocks/fullmocks/v0.5.4",
        help="Directory containing TF_extended_AbacusSummit_*.fits files",
    )
    parser.add_argument(
        "--one", action="store_true",
        help="Process only the first matching file (for debugging)",
    )
    parser.add_argument("--haty_max",  type=float, default=-20.0)
    parser.add_argument("--haty_min",  type=float, default=-21.8)
    parser.add_argument("--z_obs_min", type=float, default=0.01)
    parser.add_argument("--slope_plane",      type=float, default=-6.5)
    parser.add_argument("--intercept_plane",  type=float, default=-20.)
    parser.add_argument("--intercept_plane2", type=float, default=-18.)
    parser.add_argument("--n_objects",   type=int, default=10000,
                        help="Number of objects saved to input.json for Stan (None = all selected)")
    parser.add_argument("--random_seed", type=int, default=None)
    args = parser.parse_args()

    pattern = os.path.join(args.dir, "TF_extended_AbacusSummit_*.fits")
    fits_files = sorted(glob.glob(pattern))

    if not fits_files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    if args.one:
        fits_files = fits_files[:1]

    print(f"Found {len(fits_files)} file(s) to process.\n")

    for fits_file in fits_files:
        run = run_name_from_path(fits_file)
        run_dir = os.path.join("output", run)
        os.makedirs(run_dir, exist_ok=True)

        output_json = os.path.join(run_dir, "input.json")
        init_json   = os.path.join(run_dir, "init.json")
        plot_file   = os.path.join(run_dir, "data.png")

        config = {
            "source":           fits_file,
            "haty_max":         args.haty_max,
            "haty_min":         args.haty_min,
            "z_obs_min":        args.z_obs_min,
            "slope_plane":      args.slope_plane,
            "intercept_plane":  args.intercept_plane,
            "intercept_plane2": args.intercept_plane2,
            "n_objects":        args.n_objects,
            "random_seed":      args.random_seed,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"=== {run} ===")
        x_all, y_all, sx_all, sy_all, x_sel, y_sel, sx_sel, sy_sel = (
            process_fullmocks_tf_data(
                fits_file,
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
        )

        plot_fullmocks_tf_data(
            x_all, y_all, sx_all, sy_all,
            x_sel, y_sel, sx_sel, sy_sel,
            haty_max=args.haty_max,
            haty_min=args.haty_min,
            slope_plane=args.slope_plane,
            intercept_plane=args.intercept_plane,
            intercept_plane2=args.intercept_plane2,
            output_file=plot_file,
            title=f"AbacusSummit TF Mock — {run}",
        )
        print()
