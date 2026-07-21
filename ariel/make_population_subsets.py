#!/usr/bin/env python
"""Split a DESI/SGA catalog into per-population FITS files, one per morphology.

Each population is fit independently by the 2color pipeline. Rather than teach
every stage about the population cut, we pre-filter the catalog here so that each
population is an ordinary single-FITS run. The validity mask is re-derived
independently in desi_data.py, color_predict.py (three inline copies plus
_get_holdout_mask) and explore_residuals.py; a population predicate threaded
through all of them could be applied in one place and missed in another, silently
fitting one population and validating against another. A pre-filtered file cannot
disagree with itself.

Only the population predicate is applied here (morphology + not VI-rejected).
Validity and phase-space cuts stay owned by the pipeline.

Usage:
    python make_population_subsets.py                       # defaults
    python make_population_subsets.py --force               # overwrite existing
    python make_population_subsets.py --input data/other.fits --outdir data
"""
import argparse
import datetime as _dt
import os
import subprocess

import numpy as np
from astropy.io import fits
from astropy.table import Table

DEFAULT_INPUT = "data/SGA-2020_loa_Vrot_VI_v0.fits"

# suffix -> MORPHTYPE value. Both populations additionally require that the
# galaxy was NOT rejected by John's visual inspection (JOHN_VI is masked).
POPULATIONS = {
    "spiral": "Spiral",
    "irregular": "Irregular",
}

# Split on MORPHTYPE, not MORPHTYPE_AI: MORPHTYPE is the authoritative
# morphology label, whereas MORPHTYPE_AI is an AI classification that
# disagrees for a subset (e.g. it tags ~1600 MORPHTYPE=='Spiral' rows as
# Irregular/Undecided/Lenticular/Elliptical, which would wrongly drop them
# from the spiral population).
MORPH_COL = "MORPHTYPE"
VI_COL = "JOHN_VI"


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def population_mask(table, morphtype):
    """Rows with MORPHTYPE == morphtype that were not VI-rejected.

    JOHN_VI is a masked column whose only unmasked value is 'reject', so
    mask == True means "no VI rejection recorded" and mask == False means the
    galaxy was rejected. An unmasked JOHN_VI column would mean every row carries
    a value, i.e. every row was rejected -- treat that as no rows kept rather
    than silently keeping everything.
    """
    for col in (MORPH_COL, VI_COL):
        if col not in table.colnames:
            raise KeyError(
                f"Column {col!r} not found in the catalog. This script is for the "
                f"visually-inspected SGA catalogs; mock catalogs do not carry it."
            )
    morph = np.asarray(table[MORPH_COL]).astype(str)
    vi = table[VI_COL]
    not_rejected = (np.asarray(vi.mask) if hasattr(vi, "mask")
                    else np.zeros(len(table), dtype=bool))
    return (morph == morphtype) & not_rejected


def write_subset(table, mask, path, parent, morphtype, force):
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    sub = table[mask]
    sub.write(path, overwrite=True)

    predicate = f"{MORPH_COL} == '{morphtype}' AND {VI_COL} is masked (not VI-rejected)"
    with fits.open(path, mode="update") as hdul:
        h = hdul[1].header  # type: ignore[union-attr]
        h["PARENT"] = (os.path.basename(parent), "catalog this subset was cut from")
        h["POPULATN"] = (morphtype, "MORPHTYPE value selected")
        h["NPARENT"] = (len(table), "rows in parent catalog")
        h["NROWS"] = (len(sub), "rows in this subset")
        h["MADEUTC"] = (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                        "UTC creation time")
        h["GITSHA"] = (_git_sha(), "ariel git revision")
        h.add_history(f"make_population_subsets.py: {predicate}")
    return len(sub)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"parent FITS (default: {DEFAULT_INPUT})")
    p.add_argument("--outdir", default=None, help="output directory (default: alongside --input)")
    p.add_argument("--force", action="store_true", help="overwrite existing subset files")
    args = p.parse_args()

    outdir = args.outdir or os.path.dirname(args.input) or "."
    stem = os.path.splitext(os.path.basename(args.input))[0]

    print(f"Reading {args.input}")
    table = Table.read(args.input)
    print(f"  {len(table)} rows, {len(table.colnames)} columns")

    masks, written = {}, {}
    for suffix, morphtype in POPULATIONS.items():
        m = population_mask(table, morphtype)
        masks[suffix] = m
        path = os.path.join(outdir, f"{stem}_{suffix}.fits")
        n = write_subset(table, m, path, args.input, morphtype, args.force)
        written[suffix] = path
        print(f"  {morphtype:<10} -> {path}  ({n} rows)")

    # The populations must not overlap: a galaxy fit in both would appear in two
    # independent likelihoods, and MORPHTYPE is single-valued, so any overlap
    # means the predicate is wrong.
    keys = list(masks)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            both = int((masks[keys[i]] & masks[keys[j]]).sum())
            if both:
                raise AssertionError(f"{keys[i]} and {keys[j]} overlap in {both} rows")
    print(f"\nPopulations are disjoint. Wrote {len(written)} files.")


if __name__ == "__main__":
    main()
