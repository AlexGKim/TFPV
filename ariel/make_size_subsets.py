#!/usr/bin/env python
"""Split a DESI/SGA catalog into per-size FITS files, cut on physical diameter.

This is the size-split sibling of make_population_subsets.py: same principle
(apply the population predicate once, upstream, so every downstream stage is an
ordinary single-FITS run), different predicate. Instead of MORPHTYPE it cuts on
D26_kpc, the SGA r-band 26 mag/arcsec^2 isophotal diameter in kpc:

    large : D26_kpc >  30
    small : D26_kpc <= 30

The cut is on D26_kpc, not the angular D26 column -- D26 is in arcmin and tops
out near 22 for this catalog, so a threshold of 30 there would select nothing.

Unlike the morphology split, the DR2 disk sample arrives as two files (spirals
and irregulars), so --input takes several catalogs and stacks them into one
parent before cutting. The stacked parent is written out too, because the driver
(run_size_onepop.sh) derives its RUN/CONFIG names from a parent file that has to
exist on disk.

Only the size predicate (plus "not VI-rejected") is applied here. Validity and
phase-space cuts stay owned by the pipeline.

Usage:
    python make_size_subsets.py --input data/DR2_TF_spirals_v5.fits data/DR2_TF_irrs_v5.fits
    python make_size_subsets.py --input a.fits b.fits --parent-stem DR2_TF_disks_v5 --force
    python make_size_subsets.py --input a.fits --threshold 20 --column D26_kpc
"""
import argparse
import datetime as _dt
import os
import subprocess

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

DEFAULT_INPUTS = ["data/DR2_TF_spirals_v5.fits", "data/DR2_TF_irrs_v5.fits"]
DEFAULT_PARENT_STEM = "DR2_TF_disks_v5"
DEFAULT_COLUMN = "D26_kpc"
DEFAULT_THRESHOLD = 30.0

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
    """Read and vstack several catalogs into one parent table.

    The inputs must share a schema and must not share galaxies: the size
    populations are fit as independent likelihoods, so a galaxy appearing twice
    in the parent would appear twice in one of them.
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
    parent = vstack(tables, metadata_conflicts="silent") if len(tables) > 1 else tables[0]

    if ID_COL in parent.colnames:
        ids = np.asarray(parent[ID_COL])
        n_unique = len(np.unique(ids))
        if n_unique != len(parent):
            raise ValueError(
                f"The inputs overlap: {len(parent) - n_unique} duplicate {ID_COL} "
                f"values in the stacked parent. Each galaxy must appear once."
            )
    return parent


def size_masks(table, column, threshold):
    """(large, small) masks: not VI-rejected, and above / at-or-below threshold.

    A non-finite size is an error, not a silent drop: it would leave the two
    subsets failing to partition the parent, which is exactly the kind of
    quiet sample loss this pre-filter exists to make impossible.
    """
    if column not in table.colnames:
        raise KeyError(f"Column {column!r} not found in the catalog.")
    size = np.asarray(table[column], dtype=float)
    keep = not_vi_rejected(table)
    bad = keep & ~np.isfinite(size)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} kept rows have a non-finite {column}; refusing to "
            f"split a sample whose size cut is undefined for some galaxies."
        )
    return keep & (size > threshold), keep & (size <= threshold)


def write_subset(table, mask, path, parent_name, predicate, popvalue, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    sub = table[mask]
    sub.write(path, overwrite=True)

    with fits.open(path, mode="update") as hdul:
        h = hdul[1].header  # type: ignore[union-attr]
        h["PARENT"] = (os.path.basename(parent_name), "catalog this subset was cut from")
        h["POPULATN"] = (popvalue, "size population selected")
        h["NPARENT"] = (len(table), "rows in parent catalog")
        h["NROWS"] = (len(sub), "rows in this subset")
        h["MADEUTC"] = (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                        "UTC creation time")
        h["GITSHA"] = (_git_sha(), "ariel git revision")
        h.add_history(f"make_size_subsets.py: {predicate}")
    return len(sub)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=DEFAULT_INPUTS,
                   help=f"parent FITS file(s), stacked in order (default: {' '.join(DEFAULT_INPUTS)})")
    p.add_argument("--parent-stem", default=DEFAULT_PARENT_STEM,
                   help=f"basename stem of the stacked parent (default: {DEFAULT_PARENT_STEM})")
    p.add_argument("--outdir", default=None, help="output directory (default: alongside the first --input)")
    p.add_argument("--column", default=DEFAULT_COLUMN, help=f"size column (default: {DEFAULT_COLUMN})")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"size threshold, same units as --column (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--force", action="store_true", help="overwrite existing parent/subset files")
    args = p.parse_args()

    outdir = args.outdir or os.path.dirname(args.input[0]) or "."
    stem = args.parent_stem

    print(f"Reading {len(args.input)} input catalog(s)")
    parent = stack_inputs(args.input)
    print(f"  stacked parent: {len(parent)} rows")

    parent_path = os.path.join(outdir, f"{stem}.fits")
    if os.path.exists(parent_path) and not args.force:
        raise FileExistsError(f"{parent_path} exists; pass --force to overwrite")
    parent.write(parent_path, overwrite=True)
    print(f"  parent -> {parent_path}  ({len(parent)} rows)")

    large, small = size_masks(parent, args.column, args.threshold)
    populations = {
        "large": (large, f"{args.column} > {args.threshold:g}"),
        "small": (small, f"{args.column} <= {args.threshold:g}"),
    }

    masks = {}
    for suffix, (mask, size_predicate) in populations.items():
        masks[suffix] = mask
        predicate = f"{size_predicate} AND {VI_COL} is masked (not VI-rejected)"
        path = os.path.join(outdir, f"{stem}_{suffix}.fits")
        n = write_subset(parent, mask, path, parent_path, predicate,
                         size_predicate, args.force)
        print(f"  {suffix:<10} -> {path}  ({n} rows)   [{predicate}]")

    # The populations must not overlap -- a galaxy fit in both would appear in
    # two independent likelihoods -- and together they must account for every
    # non-VI-rejected row, or the split has quietly dropped galaxies.
    both = int((masks["large"] & masks["small"]).sum())
    if both:
        raise AssertionError(f"large and small overlap in {both} rows")
    n_kept = int(not_vi_rejected(parent).sum())
    n_split = int(masks["large"].sum() + masks["small"].sum())
    if n_split != n_kept:
        raise AssertionError(
            f"subsets do not partition the parent: {n_split} split rows vs "
            f"{n_kept} non-VI-rejected rows"
        )
    print(f"\nPopulations are disjoint and partition the {n_kept} non-VI-rejected "
          f"rows ({len(parent) - n_kept} VI-rejected dropped). Wrote 3 files.")


if __name__ == "__main__":
    main()
