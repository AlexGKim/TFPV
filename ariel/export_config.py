#!/usr/bin/env python3
"""
export_config.py — Export a run's fiducial choices to a portable config file.

Run this after completing the interactive Phase A workflow (steps 1–3,
including set_fiducial.py) to capture all parameter choices in a single JSON
file that can be committed to git and used to reproduce or vary the run.

Usage:
  python export_config.py --run DR1 --out configs/dr1_default.json
"""

import argparse
import json
import os
import sys


FIELD_DEFAULTS = {
    "fits_file": "data/DESI-DR1_TF_pv_cat_v15.fits",
    "exe":       "tophat",
    "source":    "DESI",
    "model":     "tophat",
    "n_sigma":   3.0,
}


def prompt_str(msg, default=None):
    if default is not None:
        raw = input(f"{msg} [{default}]: ").strip()
        return raw if raw else default
    raw = input(f"{msg}: ").strip()
    if not raw:
        print("  Value required.")
        return prompt_str(msg, default)
    return raw


def prompt_float(msg, default=None):
    while True:
        raw = input(f"{msg} [{default}]: " if default is not None else f"{msg}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def main():
    parser = argparse.ArgumentParser(
        description="Export a completed run's fiducial choices to a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True,
                        help="Run name (reads output/<run>/select_v2_fiducial.json)")
    parser.add_argument("--out", required=True,
                        help="Output config path (e.g. configs/dr1_default.json)")
    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)
    fiducial_path = os.path.join(run_dir, "select_v2_fiducial.json")

    if not os.path.exists(fiducial_path):
        print(f"Error: {fiducial_path} not found.", file=sys.stderr)
        print("Run set_fiducial.py first to generate it.", file=sys.stderr)
        sys.exit(1)

    with open(fiducial_path) as f:
        fiducial = json.load(f)

    # Read source from config.json if available
    config_path = os.path.join(run_dir, "config.json")
    source_default = FIELD_DEFAULTS["source"]
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if "source" in cfg:
            source_default = cfg["source"]

    print()
    print(f"Fiducial parameters from {fiducial_path}:")
    for k, v in fiducial.items():
        print(f"  {k} = {v}")
    print()
    print("Enter additional pipeline settings (press Enter to accept defaults):")

    run_name = prompt_str("run name (output directory)", default=args.run)
    exe       = prompt_str("exe (Stan binary)", default=FIELD_DEFAULTS["exe"])
    source    = prompt_str("source", default=source_default)
    model     = prompt_str("model (tophat or normal)", default=FIELD_DEFAULTS["model"])
    n_sigma   = prompt_float("n_sigma (GMM ellipse sigma for selection_ellipse.py)",
                             default=FIELD_DEFAULTS["n_sigma"])

    config = {
        "run":      run_name,
        "fits_file": FIELD_DEFAULTS["fits_file"],
        "exe":      exe,
        "source":   source,
        "model":    model,
        "n_sigma":  n_sigma,
        **fiducial,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Config written to {args.out}")
    print("Commit this file to git to version-control your parameter choices.")


if __name__ == "__main__":
    main()
