#!/usr/bin/env python3
"""
Generate per-file pipeline configs for a directory of "spec"-family
AbacusSummit mock FITS files (TF_AbacusSummit_spec_c###_ph###_r###.fits --
no MAIN column, source=DESI; distinct from the "base"/fullmocks family that
make_batch_configs.py targets).

Unlike make_batch_configs.py (which clones a single frozen base config,
including slope_plane/intercept_plane/intercept_plane2, onto every file),
this script freezes only haty_min/haty_max/z_obs_min/z_obs_max/n_sigma_perp
across the batch and re-derives slope_plane/intercept_plane/intercept_plane2
per file from that file's own Maximum-Likelihood fit (Step 2's Stan MLE),
via:

    sigma_minor = sqrt(smaller eigenvalue of the GMM ellipse covariance)
    slope_plane = MLE_slope
    half_width  = n_sigma_perp * sigma_minor * sqrt(1 + slope_plane**2)
    intercept_plane  = MLE_intercept - half_width
    intercept_plane2 = MLE_intercept + half_width

i.e. the selection band is centered on the MLE-fitted line itself, with the
same n_sigma_perp * sigma_minor width the GMM-ellipse construction
(_cuts_at_nsigma in ellipse_sweep.py) already uses -- just applied
perpendicular to the MLE slope instead of the ellipse's minor axis.

For each file this runs Step 1 (selection_ellipse.py) and Step 2
(select_v2.py, non-interactive diagnostic mode) as subprocesses if not
already done, then computes and writes output/<run>/select_v2_fiducial.json
(same format set_fiducial.py writes) and configs/<outdir>/<run>.json
directly -- no interactive prompts anywhere in this path.

Usage:
    python3 make_spec_batch_configs.py \
        --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.8 \
        --outdir configs/batch_v0.5.8
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

RUN_TOKEN_RE = re.compile(r"(c\d+_ph\d+_r\d+)")

# Frozen across the whole batch (from this session's local determination on
# TF_AbacusSummit_spec_c000_ph000_r001.fits).
HATY_MIN = -22.0
HATY_MAX = -17.8
Z_OBS_MIN = 0.01
Z_OBS_MAX = 0.065
N_SIGMA_PERP = 3.0

# Step 1's loose pre-filter (not the final cut) -- matches DR2_2COLOR.md.
LOOSE_HATY_MIN = -23.0
LOOSE_HATY_MAX = -18.0

# Fixed pipeline settings for every run.
EXE = "2color"
SOURCE = "DESI"
MODEL = "2color"
N_SIGMA = 3.0


def derive_run_name(fits_path):
    stem = os.path.basename(fits_path)
    m = RUN_TOKEN_RE.search(stem)
    return m.group(1) if m else None


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {' '.join(cmd)}")
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        return False
    return True


def ensure_step1(run, fits_path):
    """selection_ellipse.py -- loose pre-filter, non-interactive."""
    out_path = os.path.join("output", run, "selection_ellipse.json")
    if os.path.exists(out_path):
        return True
    print(f"  Step 1: selection_ellipse.py for {run}")
    return _run([
        "python3", "selection_ellipse.py",
        "--file", fits_path, "--run", run, "--source", SOURCE,
        "--z_obs_min", str(Z_OBS_MIN), "--z_obs_max", str(Z_OBS_MAX),
        "--haty_min", str(LOOSE_HATY_MIN), "--haty_max", str(LOOSE_HATY_MAX),
    ])


def ensure_step2(run, fits_path):
    """select_v2.py -- diagnostic/non-interactive mode, writes select_v2_mle.json."""
    out_path = os.path.join("output", run, "select_v2_mle.json")
    if os.path.exists(out_path):
        return True
    print(f"  Step 2: select_v2.py (MLE) for {run}")
    return _run([
        "python3", "select_v2.py",
        "--run", run, "--fits_file", fits_path, "--exe", "./tophat",
        "--z_obs_min", str(Z_OBS_MIN), "--z_obs_max", str(Z_OBS_MAX),
    ])


def compute_fiducial(run):
    """Read selection_ellipse.json + select_v2_mle.json, return the fiducial dict."""
    with open(os.path.join("output", run, "selection_ellipse.json")) as f:
        ell = json.load(f)
    with open(os.path.join("output", run, "select_v2_mle.json")) as f:
        mle = json.load(f)

    cov = np.array(ell["covariance"])
    vals = np.linalg.eigvalsh(cov)
    sigma_minor = float(math.sqrt(vals[0]))

    slope_plane = float(mle["slope"])
    mle_intercept = float(mle["intercept.1"])
    half_width = N_SIGMA_PERP * sigma_minor * math.sqrt(1.0 + slope_plane ** 2)

    return {
        "haty_min": HATY_MIN,
        "haty_max": HATY_MAX,
        "slope_plane": slope_plane,
        "intercept_plane": mle_intercept - half_width,
        "intercept_plane2": mle_intercept + half_width,
        "n_sigma_perp": N_SIGMA_PERP,
        "z_obs_min": Z_OBS_MIN,
        "z_obs_max": Z_OBS_MAX,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True,
                        help="Directory of spec-family mock FITS files to process")
    parser.add_argument("--outdir", required=True,
                        help="Directory to write per-file config JSONs into")
    parser.add_argument("--pattern", default="*.fits",
                        help="Glob pattern for FITS files within --dir (default: *.fits)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run steps 1-2 and overwrite existing config JSONs")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"ERROR: --dir not found: {args.dir}")

    fits_files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not fits_files:
        sys.exit(f"ERROR: no files matching '{args.pattern}' in {args.dir}")

    os.makedirs(args.outdir, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_failed = 0
    seen_tokens = {}
    for fits_path in fits_files:
        token = derive_run_name(fits_path)
        if token is None:
            print(f"  WARN: no c/ph/r token in {os.path.basename(fits_path)} — skipping")
            n_skipped += 1
            continue
        if token in seen_tokens:
            print(f"  WARN: run token '{token}' already taken by {seen_tokens[token]} — "
                  f"skipping duplicate {os.path.basename(fits_path)}")
            n_skipped += 1
            continue
        seen_tokens[token] = os.path.basename(fits_path)

        cfg_path = os.path.join(args.outdir, f"{token}.json")
        if os.path.exists(cfg_path) and not args.overwrite:
            print(f"  skip (exists): {cfg_path}")
            n_skipped += 1
            continue

        os.makedirs(os.path.join("output", token), exist_ok=True)
        fits_abspath = os.path.abspath(fits_path)

        if args.overwrite:
            for name in ("selection_ellipse.json", "select_v2_mle.json"):
                p = os.path.join("output", token, name)
                if os.path.exists(p):
                    os.remove(p)

        if not ensure_step1(token, fits_abspath):
            n_failed += 1
            continue
        if not ensure_step2(token, fits_abspath):
            n_failed += 1
            continue

        fiducial = compute_fiducial(token)

        fiducial_path = os.path.join("output", token, "select_v2_fiducial.json")
        with open(fiducial_path, "w") as f:
            json.dump(fiducial, f, indent=2)

        cfg = {
            "run": token,
            "fits_file": fits_abspath,
            "exe": EXE,
            "source": SOURCE,
            "model": MODEL,
            "n_sigma": N_SIGMA,
            **fiducial,
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  wrote {cfg_path}  (slope_plane={fiducial['slope_plane']:.4f})")
        n_written += 1

    print("---")
    print(f"FITS files found : {len(fits_files)}")
    print(f"configs written  : {n_written}  (in {args.outdir})")
    print(f"skipped          : {n_skipped}")
    print(f"failed           : {n_failed}")


if __name__ == "__main__":
    main()
