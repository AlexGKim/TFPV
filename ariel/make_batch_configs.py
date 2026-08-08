#!/usr/bin/env python3
"""
Generate per-file, per-subset pipeline configs for a directory of AbacusSummit
mock FITS files.

Each FITS file is split into `--n-subsets` disjoint slices (see BATCH_MOCKS.md
"Subset Partition Mode"), and a config is written for each slice named by
`--subsets`. Run names are derived from the c???_ph???_r??? token in the filename
(e.g. TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits -> token
"c000_ph000_r001") plus a subset suffix, e.g. "c000_ph000_r001_s00". Each
generated config is a copy of a base config (default configs/abacus_2color.json)
with "run", "fits_file", "subset_index", "n_subsets", and "n_objects"
overridden, so the frozen selection cuts, the frozen init (`fixed_init`),
random_seed, exe, model, etc. are shared across every mock and every slice — see
BATCH_MOCKS.md for the rationale.

`--n-subsets` sets the slice *size*; `--subsets` sets which slices actually run.
These are independent, and conflating them is a trap worth stating plainly:

    --n-subsets 5 --subsets 0      slice = 1/5 of the file, only s00 runs
    --n-subsets 1                  slice = the WHOLE file  <-- do not do this

The standard mock batch is the first form. **Do not** express "only one slice"
as `--n-subsets 1`: that makes the single slice the entire file (~170,781 valid
rows for the reference mock), and step 8's covariance dimension becomes the
whole file's MAIN count, G ~ 91,000 — a ~67 GB dense matrix that cannot finish
inside step8's 30-minute walltime. Keep --n-subsets at 5 and narrow with
--subsets.

Optionally seeds each (file, subset) run's output directory with a reusable
metric.json so step5e (the ~7h metric build) can be skipped entirely — metric.json
is data-type-level, not per-file or per-subset, so the same one is copied to every
run dir generated here.

Usage:
    python3 make_batch_configs.py \
        --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
        --base configs/abacus_2color.json \
        --outdir configs/batch_v0.5.7 \
        --n-subsets 5 --subsets 0 --n-objects 5000
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
                        help="Directory to write per-file-per-subset config JSONs into")
    parser.add_argument("--n-subsets", type=int, default=5,
                        help="Disjoint slices to split each FITS file into (default: 5). "
                             "This sets the slice SIZE, not how many run — narrow with "
                             "--subsets. Do not set this to 1 to get a single slice: that "
                             "makes the slice the whole file and step8's covariance ~67 GB.")
    parser.add_argument("--subsets", default=None,
                        help="Comma-separated subset_index values to emit, e.g. '0' for the "
                             "standard s00-only mock batch, or '0,2,4'. Default: all of "
                             "range(--n-subsets).")
    parser.add_argument("--n-objects", type=int, default=5000,
                        help="Training sample size within each subset (default: 5000)")
    parser.add_argument("--metric", default=None,
                        help="Optional metric.json to copy into each output/<run>/ "
                             "(skips step5e). Typically output/abacus_2color/metric.json. "
                             "One metric is shared across every file and every subset "
                             "of the same data type — see BATCH_MOCKS.md decision #3.")
    parser.add_argument("--pattern", default="*.fits",
                        help="Glob pattern for FITS files within --dir (default: *.fits)")
    parser.add_argument("--run-suffix", default="",
                        help="Append this to every derived file token (e.g. '_dbg') to "
                             "isolate debug output dirs from real runs")
    parser.add_argument("--output-root", default="output",
                        help="Root under which per-run output dirs live (default: output)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-run config JSONs (default: skip existing)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"ERROR: --dir not found: {args.dir}")
    if args.n_subsets < 1:
        sys.exit(f"ERROR: --n-subsets must be >= 1, got {args.n_subsets}")

    if args.subsets is None:
        subset_indices = list(range(args.n_subsets))
    else:
        try:
            subset_indices = [int(s) for s in args.subsets.split(",") if s.strip() != ""]
        except ValueError:
            sys.exit(f"ERROR: --subsets must be comma-separated integers, got {args.subsets!r}")
        if not subset_indices:
            sys.exit("ERROR: --subsets was empty")
        bad = [i for i in subset_indices if not (0 <= i < args.n_subsets)]
        if bad:
            sys.exit(f"ERROR: --subsets {bad} out of range for --n-subsets "
                     f"{args.n_subsets} (valid: 0..{args.n_subsets - 1})")
    print(f"Slices: n_subsets={args.n_subsets} (slice size), emitting subset_index "
          f"{subset_indices}")
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
    seen_tokens = {}
    for fits_path in fits_files:
        token = derive_run_name(fits_path)
        if token is None:
            print(f"  WARN: no c/ph/r token in {os.path.basename(fits_path)} — skipping")
            n_skipped += 1
            continue
        token = token + args.run_suffix
        if token in seen_tokens:
            print(f"  WARN: run token '{token}' already taken by {seen_tokens[token]} — "
                  f"skipping duplicate {os.path.basename(fits_path)}")
            n_skipped += 1
            continue
        seen_tokens[token] = os.path.basename(fits_path)

        for subset_index in subset_indices:
            run = f"{token}_s{subset_index:02d}"

            cfg = dict(base_cfg)
            cfg["run"] = run
            cfg["fits_file"] = os.path.abspath(fits_path)
            cfg["subset_index"] = subset_index
            cfg["n_subsets"] = args.n_subsets
            cfg["n_objects"] = args.n_objects

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

    n_runs = len(seen_tokens) * args.n_subsets
    print("---")
    print(f"FITS files found : {len(fits_files)}")
    print(f"subsets per file : {args.n_subsets}")
    print(f"runs (file×subset): {n_runs}")
    print(f"configs written  : {n_written}  (in {args.outdir})")
    print(f"skipped          : {n_skipped}")
    if args.metric is not None:
        print(f"metric seeded    : {args.metric} -> {args.output_root}/<run>/metric.json")


if __name__ == "__main__":
    main()
