#!/usr/bin/env python3
"""
run_pipeline.py — Run the full TFR pipeline non-interactively from a config file.

Use this for Phase B (variant runs) after Phase A (interactive discovery) has
been completed and a config file exported with export_config.py.

The fiducial selection criteria (Step 3) are written directly from the config,
bypassing the interactive set_fiducial.py prompts.  Stan sampling (Step 5) is
printed as a command for you to run manually.

Usage:
  python run_pipeline.py configs/dr1_default.json
  python run_pipeline.py configs/dr1_zmax015.json --steps 1-4
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from ellipse_sweep import _cuts_at_nsigma


REQUIRED_KEYS = [
    "run", "fits_file", "exe", "source", "model",
    "z_obs_min", "z_obs_max", "haty_min", "haty_max",
    "n_sigma_perp", "n_sigma",
]


def parse_steps(spec):
    """Parse a steps spec like '1-4' or '3' into a set of ints."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return set(range(int(lo), int(hi) + 1))
    return {int(spec)}


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], **kwargs)
    if result.returncode != 0:
        print(f"Error: command exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the TFR pipeline non-interactively from a config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="Path to config JSON (e.g. configs/dr1_default.json)")
    parser.add_argument("--steps", default="1-7",
                        help="Steps to run, e.g. '1-4' or '3' (default: 1-7)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        print(f"Error: config is missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    steps = parse_steps(args.steps)
    run_name = cfg["run"]
    run_dir  = os.path.join("output", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run: {run_name}  →  {run_dir}")
    print(f"Config: {args.config}")
    print(f"Steps: {sorted(steps)}")
    print()
    print("To run with manual fiducial (set_fiducial.py):")
    print(f"  python run_pipeline.py {args.config} --steps 1-2")
    print(f"  python set_fiducial.py --run {run_name}")
    print(f"  python run_pipeline.py {args.config} --steps 4-7")
    print()

    # ── Step 1: Estimate core distribution ────────────────────────────────────
    if 1 in steps:
        print("\n── Step 1: selection_ellipse.py ──")
        run(["python", "selection_ellipse.py", "--config", args.config])

    # ── Step 2: MLE fit and pull-profile diagnostic ────────────────────────────
    if 2 in steps:
        print("\n── Step 2: select_v2.py (MLE) ──")
        run(["python", "select_v2.py", "--config", args.config])

    # ── Step 3: Write fiducial JSON from config ────────────────────────────────
    if 3 in steps:
        print("\n── Step 3: write select_v2_fiducial.json from config ──")
        ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
        if not os.path.exists(ellipse_path):
            print(f"Error: {ellipse_path} not found — run step 1 first.", file=sys.stderr)
            sys.exit(1)
        with open(ellipse_path) as f:
            ell = json.load(f)
        mu    = np.array(ell["mean"])
        sigma = np.array(ell["covariance"])
        cuts  = _cuts_at_nsigma(mu, sigma, cfg["n_sigma_perp"])

        fiducial = {
            "haty_min":         cfg["haty_min"],
            "haty_max":         cfg["haty_max"],
            "slope_plane":      cuts["slope_plane"],
            "intercept_plane":  cuts["intercept_plane"],
            "intercept_plane2": cuts["intercept_plane2"],
            "n_sigma_perp":     cfg["n_sigma_perp"],
            "z_obs_min":        cfg["z_obs_min"],
            "z_obs_max":        cfg["z_obs_max"],
        }
        fiducial_path = os.path.join(run_dir, "select_v2_fiducial.json")
        with open(fiducial_path, "w") as f:
            json.dump(fiducial, f, indent=2)
        print(f"Saved → {fiducial_path}")
        for k, v in fiducial.items():
            print(f"  {k} = {v}")

    # ── Step 4: Prepare Stan data ─────────────────────────────────────────────
    if 4 in steps:
        # Promote fiducial JSON → config file (captures manual set_fiducial.py edits)
        fiducial_path = os.path.join(run_dir, "select_v2_fiducial.json")
        if os.path.exists(fiducial_path):
            with open(fiducial_path) as f:
                fiducial = json.load(f)
            changed = {k: v for k, v in fiducial.items() if cfg.get(k) != v}
            if changed:
                cfg.update(fiducial)
                with open(args.config, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"\n── Promoted fiducial → {args.config} ──")
                for k, v in changed.items():
                    print(f"  {k} = {v}")

        print("\n── Step 4: desi_data.py ──")
        run(["python", "desi_data.py", "--config", args.config])

    # ── Step 5: Stan sampling (manual) ───────────────────────────────────────
    if 5 in steps:
        print("\n── Step 5: Stan sampling (run manually) ──")
        for model in (cfg["model"],) if cfg.get("model") else ("tophat", "normal"):
            cmd = (
                f"./{model} sample num_warmup=500 num_samples=1000 num_chains=4 \\\n"
                f"    adapt save_metric=1 \\\n"
                f"    data file={run_dir}/input.json \\\n"
                f"    init={run_dir}/init.json \\\n"
                f"    output file={run_dir}/{model}.csv"
            )
            print(f"\n{cmd}")

    # ── Step 6: Corner plot ───────────────────────────────────────────────────
    if 6 in steps:
        print("\n── Step 6: corner.py ──")
        run(["python", "corner.py", "--run", run_name, "--model", cfg["model"]])

    # ── Step 7: Predict absolute magnitudes ───────────────────────────────────
    if 7 in steps:
        print("\n── Step 7: predict.py ──")
        run(["python", "predict.py", "--config", args.config, "--catalog"])

    print(f"\nDone. Outputs in {run_dir}/")


if __name__ == "__main__":
    main()
