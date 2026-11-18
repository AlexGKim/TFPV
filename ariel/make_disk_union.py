#!/usr/bin/env python
"""Stack the DR2 disk catalogs into one FITS file, for a single-population fit.

The DR2 disk sample ships as two morphology files (spirals, irregulars) that
DR2_TWOPOP.md fits as two independent populations. This script builds the
un-split alternative: one catalog containing every galaxy from both files, fit
as a single population (see DR2_DISKS_SINGLE.md).

Like make_population_subsets.py, the only cut applied here is the population
predicate -- which for this run is just "not VI-rejected". That cut has to
happen here because nothing downstream applies it: desi_data.py and
color_predict.py re-derive validity and phase-space cuts, but JOHN_VI is not
among them, so a galaxy John rejected would otherwise be fit. Both the
morphology and the size chains drop these same rows, which is what keeps the
three runs comparable. Pass --keep-vi-rejected to fit all 32,488 rows instead.

Usage:
    python make_disk_union.py
    python make_disk_union.py --input a.fits b.fits --out data/other_all.fits --force
"""
import argparse
import datetime as _dt
import os
import subprocess

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

DEFAULT_INPUTS = ["data/DR2_TF_spirals_v5.fits", "data/DR2_TF_irrs_v5.fits"]
DEFAULT_OUT = "data/DR2_TF_disks_v5_all.fits"

VI_COL = "JOHN_VI"
ID_COL = "SGA_ID"


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def not_vi_rejected(table):
    """Rows that were not rejected by John's visual inspection.

    JOHN_VI is a masked column whose only unmasked value is 'reject', so
    mask == True means "no VI rejection recorded" and mask == False means the
    galaxy was rejected. An unmasked JOHN_VI column would mean every row carries
    a value, i.e. every row was rejected -- treat that as no rows kept rather
    than silently keeping everything.
    """
    if VI_COL not in table.colnames:
        raise KeyError(
            f"Column {VI_COL!r} not found in the catalog. This script is for the "
            f"visually-inspected SGA catalogs; mock catalogs do not carry it."
        )
    vi = table[VI_COL]
    return (np.asarray(vi.mask) if hasattr(vi, "mask")
            else np.zeros(len(table), dtype=bool))


def stack_inputs(paths):
    """Read and vstack several catalogs into one table.

    The inputs must share a schema and must not share galaxies: a duplicated
    galaxy would enter the single likelihood twice.
    """
    tables = []
    for p in paths:
        t = Table.read(p)
        print(f"  {p}: {len(t)} rows, {len(t.colnames)} columns")
        if tables and t.colnames != tables[0].colnames:
            only_a = set(tables[0].colnames) - set(t.colnames)
            only_b = set(t.colnames) - set(tables[0].colnames)
            raise ValueError(
                f"{p} does not share a schema with {paths[0]}: "
                f"missing {sorted(only_a)}, extra {sorted(only_b)}"
            )
        tables.append(t)
    stacked = vstack(tables, metadata_conflicts="silent") if len(tables) > 1 else tables[0]

    if ID_COL in stacked.colnames:
        ids = np.asarray(stacked[ID_COL])
        n_unique = len(np.unique(ids))
        if n_unique != len(stacked):
            raise ValueError(
                f"The inputs overlap: {len(stacked) - n_unique} duplicate {ID_COL} "
                f"values in the stacked catalog. Each galaxy must appear once."
            )
    return stacked


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=DEFAULT_INPUTS,
                   help=f"catalogs to stack, in order (default: {' '.join(DEFAULT_INPUTS)})")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"output FITS (default: {DEFAULT_OUT})")
    p.add_argument("--keep-vi-rejected", action="store_true",
                   help="do not drop VI-rejected galaxies (default: drop them, "
                        "matching the morphology and size chains)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise FileExistsError(f"{args.out} exists; pass --force to overwrite")

    print(f"Reading {len(args.input)} input catalog(s)")
    stacked = stack_inputs(args.input)
    print(f"  stacked: {len(stacked)} rows")

    if args.keep_vi_rejected:
        out_table = stacked
        predicate = "all rows (VI-rejected galaxies KEPT)"
    else:
        keep = not_vi_rejected(stacked)
        out_table = stacked[keep]
        n_drop = len(stacked) - len(out_table)
        predicate = f"{VI_COL} is masked (not VI-rejected)"
        print(f"  dropped {n_drop} VI-rejected rows")

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    out_table.write(args.out, overwrite=True)

    with fits.open(args.out, mode="update") as hdul:
        h = hdul[1].header  # type: ignore[union-attr]
        # Short comment: the joined basenames are long, and a long value plus a
        # long comment overflows the 80-character card and triggers a truncation
        # warning. The full provenance is in the HISTORY line below either way.
        h["PARENT"] = (";".join(os.path.basename(f) for f in args.input)[:68],
                       "stacked inputs")
        h["POPULATN"] = ("all disks", "single population, no split")
        h["NPARENT"] = (len(stacked), "rows in the stacked inputs")
        h["NROWS"] = (len(out_table), "rows in this catalog")
        h["MADEUTC"] = (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                        "UTC creation time")
        h["GITSHA"] = (_git_sha(), "ariel git revision")
        h.add_history(f"make_disk_union.py: {predicate}")

    print(f"\nWrote {args.out}  ({len(out_table)} rows)   [{predicate}]")


if __name__ == "__main__":
    main()
