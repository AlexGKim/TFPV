#!/usr/bin/env python3
"""
Generate one pipeline config per AbacusSummit mock FITS file in a directory.

Run names are the c???_ph???_r??? token in the filename (e.g.
TF_AbacusSummit_base_c000_ph000_r001_zsnap0.20_zmax0.11.fits -> run
"c000_ph000_r001"), so each mock file maps to exactly one run and one output dir.
Each generated config is a copy of a base config (default
configs/abacus_2color.json) with only "run", "fits_file", and "n_objects"
overridden, so the frozen selection cuts, the frozen init (`fixed_init`),
`target_main_count`, random_seed, exe, model, etc. are shared across every mock —
see BATCH_MOCKS.md for the rationale.

There is no slice partitioning. Sample size is set by `target_main_count` in the
base config: step 4 applies the frozen trapezoid cuts to the whole file and then
draws exactly that many cut-passing galaxies as the analysed sample (17,234, the
MAIN count of DESI-DR2_TF_pv_cat_v5b.fits). That draw is what keeps step 8's
dense covariance tractable — the whole file's cut-passing population is
G ~ 91,000, a ~67 GB matrix that cannot finish inside step8's 30-minute
walltime — so do not remove `target_main_count` from the base config.

There is no metric-seeding step: every chain starts from the identity metric and
adapts a dense one during warmup (see slurm/step6_node.sh). The scripts that
built a metric have been removed from the batch.

Usage:
    python3 make_batch_configs.py \
        --dir /global/cfs/cdirs/desicollab/science/td/pv/mocks/DR2/TF_mocks/full_mocks/v0.5.7 \
        --base configs/abacus_2color.json \
        --outdir configs/batch_v0.5.7 \
        --n-objects 5000
"""

import argparse
import glob
import json
import os
import re
import sys

from astropy.io import fits

# Dust-coefficient-error header keywords, in resolve_d_err_r()'s precedence
# order (color_predict.py). A mock config carries no dust_pickle, so one of
# these must be present or step 8 silently falls back to the iron value.
_DUST_KEYS = ("A_R_ERR", "DSTCFF_R_ERR")


def check_systematics_columns(fits_path, needs_dust):
    """Return a list of reasons ``fits_path`` cannot produce a valid run.

    Both checks catch failures that would otherwise surface only in step 8,
    after the run has already spent its step-6 GPU hours:

    * **dust** — with no ``dust_pickle`` in the config, ``resolve_d_err_r()``
      needs ``A_R_ERR`` or ``DSTCFF_R_ERR`` in the header. Absent both it logs
      one WARNING and uses the iron value 0.1768, changing every covariance in
      a way that is easy to miss across 125 run logs.
    * **photsys** — ``_systematic_offdiag_terms()`` needs either a
      ``PHOTSYS_ERR`` column or a string-typed ``PHOTSYS``. Mocks store PHOTSYS
      as a numeric offset, so without ``PHOTSYS_ERR`` it now raises rather than
      silently dropping the calibration systematic.

    Reads only the header and column descriptors — no row data.
    """
    problems = []
    with fits.open(fits_path) as hdul:
        hdu = hdul[1]
        header = hdu.header
        cols = {c.name: c.format for c in hdu.columns}  # type: ignore[union-attr]

    if needs_dust and not any(k in header for k in _DUST_KEYS):
        problems.append(f"no dust error keyword (need one of {', '.join(_DUST_KEYS)})")

    if "PHOTSYS_ERR" not in cols:
        ps = cols.get("PHOTSYS")
        if ps is None:
            problems.append("no PHOTSYS_ERR column and no PHOTSYS column")
        elif not ps.endswith("A"):
            problems.append(
                f"no PHOTSYS_ERR column and PHOTSYS is numeric (TFORM={ps!r}), "
                f"so the photsys calibration systematic cannot be resolved"
            )
    return problems

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
    parser.add_argument("--n-objects", type=int, default=5000,
                        help="Training sample size within the drawn subsample (default: 5000)")
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
    with open(args.base) as f:
        base_cfg = json.load(f)

    # Sample size lives in the base config, not in a flag here. Emitting configs
    # without it would make every run analyse the whole file's cut-passing
    # population, which step 8 cannot fit in memory or walltime.
    if base_cfg.get("target_main_count") is None:
        sys.exit(f"ERROR: {args.base} has no \"target_main_count\". The mock batch "
                 f"requires it (17234, matching DESI-DR2_TF_pv_cat_v5b MAIN); "
                 f"without it step 8's covariance is ~67 GB. See BATCH_MOCKS.md.")
    print(f"target_main_count from {args.base}: {base_cfg['target_main_count']}")

    fits_files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not fits_files:
        sys.exit(f"ERROR: no files matching '{args.pattern}' in {args.dir}")

    os.makedirs(args.outdir, exist_ok=True)

    n_written = 0
    n_skipped = 0
    seen_tokens = {}

    # --- Pass 1: resolve run names ---
    accepted = []          # (token, fits_path)
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
        accepted.append((token, fits_path))

    # --- Pass 2: validate the systematics inputs BEFORE writing anything ---
    # A file that cannot produce a correct covariance should cost seconds here,
    # not a step-6 GPU allocation followed by a step-8 failure. Nothing is
    # written if any file fails, so the batch is all-or-nothing rather than
    # silently partial.
    needs_dust = base_cfg.get("dust_pickle") is None
    if not needs_dust:
        print(f"dust: base config sets dust_pickle={base_cfg['dust_pickle']!r} — "
              f"per-file header keywords not required")
    bad = {}
    for token, fits_path in accepted:
        problems = check_systematics_columns(fits_path, needs_dust)
        if problems:
            bad[os.path.basename(fits_path)] = problems
    if bad:
        print(f"\nERROR: {len(bad)} of {len(accepted)} files cannot produce a "
              f"correct step-8 covariance. No configs written.", file=sys.stderr)
        for name, problems in sorted(bad.items()):
            for prob in problems:
                print(f"  {name}: {prob}", file=sys.stderr)
        sys.exit(f"\nFix the input files, or exclude them with --pattern. "
                 f"See BATCH_MOCKS.md decisions #2c and the photsys note.")
    print(f"systematics inputs OK for all {len(accepted)} files "
          f"(dust keyword{'' if needs_dust else ' not required'}, PHOTSYS_ERR/PHOTSYS)")

    # --- Pass 3: write ---
    for token, fits_path in accepted:
        run = token

        cfg = dict(base_cfg)
        cfg["run"] = run
        cfg["fits_file"] = os.path.abspath(fits_path)
        cfg["n_objects"] = args.n_objects

        cfg_path = os.path.join(args.outdir, f"{run}.json")
        if os.path.exists(cfg_path) and not args.overwrite:
            print(f"  skip (exists): {cfg_path}")
            n_skipped += 1
        else:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            n_written += 1

        # Create the run's output dir so step4 has somewhere to write.
        run_dir = os.path.join(args.output_root, run)
        os.makedirs(run_dir, exist_ok=True)

    print("---")
    print(f"FITS files found  : {len(fits_files)}")
    print(f"runs (1 per file) : {len(seen_tokens)}")
    print(f"target_main_count : {base_cfg['target_main_count']}")
    print(f"configs written   : {n_written}  (in {args.outdir})")
    print(f"skipped           : {n_skipped}")


if __name__ == "__main__":
    main()
