#!/usr/bin/env python3
"""
set_fiducial.py — Interactively set fiducial selection criteria.

Prompts the user for:
  - perpendicular cut width in sigma units (n_sigma_perp)
  - haty_min  (bright-end magnitude limit)
  - haty_max  (dim-end magnitude limit)

Computes the oblique cut intercepts from the GMM ellipse stored in
output/<run>/selection_ellipse.json and writes the result to
output/<run>/select_v2_fiducial.json for use by desi_data.py.

Usage:
  python set_fiducial.py --run DESI
"""

import argparse
import json
import os
import numpy as np

from ellipse_sweep import _cuts_at_nsigma
from select_v2 import _save_pull_plot


def main():
    parser = argparse.ArgumentParser(
        description="Interactively set fiducial selection criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True,
                        help="Run name (reads/writes output/<run>/)")
    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)

    # Load GMM ellipse
    ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
    if not os.path.exists(ellipse_path):
        raise FileNotFoundError(
            f"{ellipse_path} not found — run selection_ellipse.py first.")
    with open(ellipse_path) as f:
        ell = json.load(f)

    mu    = np.array(ell["mean"])
    sigma = np.array(ell["covariance"])

    # Show reference values from the 1-sigma ellipse
    cuts1 = _cuts_at_nsigma(mu, sigma, 1.0)
    print()
    print("GMM core component:")
    print(f"  mean      = ({mu[0]:.4f}, {mu[1]:.4f})")
    print(f"  slope     = {cuts1['slope_plane']:.4f}")
    print()
    print("Reference cut values at n_sigma_perp = 1:")
    print(f"  intercept_plane  = {cuts1['intercept_plane']:.4f}")
    print(f"  intercept_plane2 = {cuts1['intercept_plane2']:.4f}")
    print(f"  haty_min (1σ)    = {cuts1['haty_min']:.4f}")
    print(f"  haty_max (1σ)    = {cuts1['haty_max']:.4f}")
    print()

    def prompt_float(msg, default=None):
        while True:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            try:
                return float(raw)
            except ValueError:
                print("  Please enter a number.")

    # Prompt user
    n_sigma_perp = prompt_float("Enter n_sigma_perp (perpendicular cut width in sigma units): ")
    haty_min     = prompt_float("Enter haty_min (bright-end magnitude limit, e.g. -22): ")
    haty_max     = prompt_float("Enter haty_max (dim-end   magnitude limit, e.g. -19.5): ")
    z_obs_min    = prompt_float("Enter z_obs_min (minimum redshift) [default 0.03]: ", default=0.03)
    z_obs_max    = prompt_float("Enter z_obs_max (maximum redshift) [default 0.1]: ", default=0.1)

    if haty_min >= haty_max:
        raise ValueError(f"haty_min ({haty_min}) must be less than haty_max ({haty_max}).")
    if n_sigma_perp <= 0:
        raise ValueError(f"n_sigma_perp must be positive, got {n_sigma_perp}.")
    if z_obs_min < 0:
        raise ValueError(f"z_obs_min must be non-negative, got {z_obs_min}.")
    if z_obs_max <= z_obs_min:
        raise ValueError(f"z_obs_max ({z_obs_max}) must be greater than z_obs_min ({z_obs_min}).")

    # Compute oblique intercepts at the requested n_sigma_perp
    cuts = _cuts_at_nsigma(mu, sigma, n_sigma_perp)

    fiducial = {
        "haty_min":         haty_min,
        "haty_max":         haty_max,
        "slope_plane":      cuts["slope_plane"],
        "intercept_plane":  cuts["intercept_plane"],
        "intercept_plane2": cuts["intercept_plane2"],
        "n_sigma_perp":     n_sigma_perp,
        "z_obs_min":        z_obs_min,
        "z_obs_max":        z_obs_max,
    }

    print()
    print("Fiducial selection criteria:")
    print(f"  haty_min         = {fiducial['haty_min']}")
    print(f"  haty_max         = {fiducial['haty_max']}")
    print(f"  z_obs_min        = {fiducial['z_obs_min']}")
    print(f"  z_obs_max        = {fiducial['z_obs_max']}")
    print(f"  n_sigma_perp     = {fiducial['n_sigma_perp']}")
    print(f"  slope_plane      = {fiducial['slope_plane']:.6f}")
    print(f"  intercept_plane  = {fiducial['intercept_plane']:.6f}")
    print(f"  intercept_plane2 = {fiducial['intercept_plane2']:.6f}")

    fiducial_path = os.path.join(run_dir, "select_v2_fiducial.json")
    with open(fiducial_path, "w") as f:
        json.dump(fiducial, f, indent=2)
    print()
    print(f"Saved → {fiducial_path}")

    # Generate pull plot with fiducial cut lines
    pull_stats_path = os.path.join(run_dir, "select_v2_pull_stats.json")
    mle_path        = os.path.join(run_dir, "select_v2_mle.json")
    if not os.path.exists(pull_stats_path) or not os.path.exists(mle_path):
        print(f"Warning: {pull_stats_path} or {mle_path} not found — "
              "run select_v2.py first to generate pull stats.")
        return
    with open(pull_stats_path) as f:
        ps = json.load(f)
    with open(mle_path) as f:
        params = json.load(f)

    _save_pull_plot(
        run_dir, args.run,
        n_all=ps["n_all"], n_sel=ps["n_sel"],
        params=params,
        bin_centers=np.array(ps["bin_centers"]),
        bin_widths=np.array(ps["bin_widths"]),
        pulls=np.array(ps["pulls"]),
        wt_means=np.array(ps["wt_means"]),
        wt_uncs=np.array(ps["wt_uncs"]),
        haty_lines={"haty_min": haty_min, "haty_max": haty_max},
        filename="select_v2_fiducial_pull.png",
    )


if __name__ == "__main__":
    main()
