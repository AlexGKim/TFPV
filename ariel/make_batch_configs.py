#!/usr/bin/env python3
"""
Generate per-file pipeline configs for a directory of AbacusSummit mock FITS files.

Each FITS file becomes its own run, derived from the c???_ph???_r??? token in the
filename (e.g. TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits ->
run name "c000_ph000_r001"). The generated config is a copy of a base config
(default configs/abacus_2color.json) with only "run" and "fits_file" overridden,
so the frozen selection cuts, n_objects, random_seed, exe, model, etc. are shared
across every mock — see BATCH_MOCKS.md for the rationale.

Optionally seeds each run's output directory with a reusable metric.json so step5e
(the ~7h metric build) can be skipped entirely.

Usage:
    python3 make_batch_configs.py \
        --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
        --base configs/abacus_2color.json \
        --outdir configs/batch_v0.5.7 \
        --metric output/abacus_2color/metric.json
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

# Run-name token shared with fullmocks_data.py: c<NN>_ph<NN>_r<NN>
RUN_TOKEN_RE = re.compile(r"(c\d+_ph\d+_r\d+)")


def derive_run_name(fits_path):
    """Return the c???_ph???_r??? token from a mock filename, or None if absent."""
    stem = os.path.basename(fits_path)
    m = RUN_TOKEN_RE.search(stem)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True,
                        help="Directory of mock FITS files to process")
    parser.add_argument("--base", default="configs/abacus_2color.json",
                        help="Base config to clone for every run (default: configs/abacus_2color.json)")
    parser.add_argument("--outdir", required=True,
                        help="Directory to write per-file config JSONs into")
    parser.add_argument("--metric", default=None,
                        help="Optional metric.json to copy into each output/<run>/ "
                             "(skips step5e). Typically output/abacus_2color/metric.json")
    parser.add_argument("--pattern", default="*.fits",
                        help="Glob pattern for FITS files within --dir (default: *.fits)")
    parser.add_argument("--run-suffix", default="",
                        help="Append this to every derived run name (e.g. '_dbg') to "
                             "isolate debug output dirs from real runs")
    parser.add_argument("--output-root", default="output",
                        help="Root under which per-run output dirs live (default: output)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-file config JSONs (default: skip existing)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"ERROR: --dir not found: {args.dir}")
    with open(args.base) as f:
        base_cfg = json.load(f)

    if args.metric is not None and not os.path.isfile(args.metric):
        sys.exit(f"ERROR: --metric not found: {args.metric}")

    fits_files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not fits_files:
        sys.exit(f"ERROR: no files matching '{args.pattern}' in {args.dir}")

    os.makedirs(args.outdir, exist_ok=True)

    n_written = 0
    n_skipped = 0
    seen_runs = {}
    for fits_path in fits_files:
        run = derive_run_name(fits_path)
        if run is None:
            print(f"  WARN: no c/ph/r token in {os.path.basename(fits_path)} — skipping")
            n_skipped += 1
            continue
        run = run + args.run_suffix
        if run in seen_runs:
            print(f"  WARN: run name '{run}' already taken by {seen_runs[run]} — "
                  f"skipping duplicate {os.path.basename(fits_path)}")
            n_skipped += 1
            continue
        seen_runs[run] = os.path.basename(fits_path)

        cfg = dict(base_cfg)
        cfg["run"] = run
        cfg["fits_file"] = os.path.abspath(fits_path)

        cfg_path = os.path.join(args.outdir, f"{run}.json")
        if os.path.exists(cfg_path) and not args.overwrite:
            print(f"  skip (exists): {cfg_path}")
            n_skipped += 1
        else:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            n_written += 1

        # Seed the run's output dir + metric so step5e can be skipped.
        run_dir = os.path.join(args.output_root, run)
        os.makedirs(run_dir, exist_ok=True)
        if args.metric is not None:
            dst = os.path.join(run_dir, "metric.json")
            shutil.copyfile(args.metric, dst)

    print("---")
    print(f"FITS files found : {len(fits_files)}")
    print(f"configs written  : {n_written}  (in {args.outdir})")
    print(f"skipped          : {n_skipped}")
    if args.metric is not None:
        print(f"metric seeded    : {args.metric} -> {args.output_root}/<run>/metric.json")


if __name__ == "__main__":
    main()
