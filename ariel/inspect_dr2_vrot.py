#!/usr/bin/env python3
"""Inspect DR2_TF_vrot_all_v5.fits -- plan section 6a.

Nothing in the repo references this file, so its contents are unknown. This
answers the questions that decide whether Step 0 (make_disk_union.py) is
needed and whether the config may keep "source": "DESI".

Run on NERSC (it reads a CFS path):
    python3 inspect_dr2_vrot.py
    python3 inspect_dr2_vrot.py --file <other.fits>
"""
import argparse
import os

import numpy as np
from astropy.io import fits

DEFAULT = ("/global/cfs/cdirs/desi/science/td/pv/tfgalaxies/Y3/"
           "DR2_TF_vrot_all_v5.fits")

# Columns the pipeline needs, by the name desi_data.py / color_predict.py use.
WANTED = [
    "SGA_ID", "ZOBS", "Z_DESI", "Z_DESI_CMB",
    "LOGVROT", "LOGVROT_ERR",
    "MORPHTYPE", "JOHN_VI", "MAIN", "ANALYSIS", "PHOTSYS",
    "G_MAG_SB26", "R_MAG_SB26", "Z_MAG_SB26",
    "G_ABSMAG_SB26", "R_ABSMAG_SB26", "Z_ABSMAG_SB26",
    "G_MAG_SB26_ERR", "R_MAG_SB26_ERR", "Z_MAG_SB26_ERR",
    "BA", "D26", "PGC",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT)
    args = ap.parse_args()
    path = args.file

    print(f"file : {path}")
    print(f"size : {os.path.getsize(path) / 1e6:.1f} MB")
    with fits.open(path) as hl:
        hl.info()
        d = hl[1].data
        names = list(d.columns.names)
        n = len(d)

    print(f"\nrows : {n}")
    print(f"cols : {len(names)}")

    print("\n-- pipeline columns --")
    for c in WANTED:
        print(f"  {'OK     ' if c in names else 'MISSING'} {c}")

    missing_z = [c for c in ("ZOBS", "Z_DESI", "Z", "ZHELIO", "Z_CMB") if c in names]
    print(f"\nredshift columns present: {missing_z or 'NONE -- desi_data.py will fail'}")

    print("\n-- all columns --")
    for i in range(0, len(names), 6):
        print("  " + "  ".join(f"{c:<22}" for c in names[i:i + 6]))

    def strs(col):
        return np.char.strip(np.asarray(d[col]).astype(str))

    # Is it disk-only, or does it still carry early types?
    if "MORPHTYPE" in names:
        m = strs("MORPHTYPE")
        vals, cnt = np.unique(m, return_counts=True)
        order = np.argsort(-cnt)
        print("\n-- MORPHTYPE --")
        for i in order[:25]:
            print(f"  {vals[i]:<12} {cnt[i]:7d}")
        early = np.isin(m, ["E", "S0", "PSF", "REX"])
        print(f"  early types (E/S0/PSF/REX): {int(early.sum())}")
        print("  -> if 0, this is already disk-only and Step 0 is unnecessary.")
    else:
        print("\n-- MORPHTYPE ABSENT --")
        print("  Cannot tell disks from early types here; whatever made this file")
        print("  must already have done the morphology selection.")

    # Has visual inspection already been applied?
    if "JOHN_VI" in names:
        v = strs("JOHN_VI")
        vals, cnt = np.unique(v, return_counts=True)
        print("\n-- JOHN_VI --")
        for a, b in zip(vals, cnt):
            print(f"  {a:<12} {b:7d}")
        print(f"  rejects: {int((v == 'reject').sum())}")
        print("  -> if 0, the file is already VI-filtered.")
    else:
        print("\n-- JOHN_VI ABSENT: cannot tell whether VI filtering was applied --")

    # MAIN: present means do NOT filter on it for a DESI-source run.
    if "MAIN" in names:
        mm = np.asarray(d["MAIN"]).astype(bool)
        print(f"\n-- MAIN present: {int(mm.sum())} of {n} true --")
        print('  Keep "source": "DESI" in the config. desi_data.py:232-239 gates the')
        print("  MAIN restriction on the SOURCE value precisely so that a DR2 run is")
        print("  never re-selected against a stale pipeline-written MAIN.")
    else:
        print("\n-- MAIN absent --")
    if "ANALYSIS" in names:
        aa = np.asarray(d["ANALYSIS"]).astype(bool)
        print(f"-- ANALYSIS present: {int(aa.sum())} of {n} true (hold-out exists) --")

    # Redshift distribution, and what a z>0.03 cut costs.
    zcol = next((c for c in ("ZOBS", "Z_DESI", "Z") if c in names), None)
    if zcol:
        z = np.asarray(d[zcol], float)
        ok = np.isfinite(z)
        print(f"\n-- {zcol} --")
        print(f"  finite {int(ok.sum())}  min {np.nanmin(z):.5f}  max {np.nanmax(z):.5f}")
        for lo in (0.01, 0.03):
            print(f"  z > {lo}: {int((ok & (z > lo)).sum())}")
        print(f"  0.01 < z <= 0.03: {int((ok & (z > 0.01) & (z <= 0.03)).sum())}")
        print(f"  0.03 < z <= 0.065: {int((ok & (z > 0.03) & (z <= 0.065)).sum())}")

    # The v2.0.8 defect: negative magnitude errors.
    print("\n-- negative magnitude errors (the v2.0.8 defect) --")
    for c in ("G_MAG_SB26_ERR", "R_MAG_SB26_ERR", "Z_MAG_SB26_ERR"):
        if c in names:
            e = np.asarray(d[c], float)
            print(f"  {c}: {int(np.sum(np.isfinite(e) & (e < 0)))} negative")

    # Comparison targets from the plan.
    print("\n-- comparison --")
    print("  DR2_TF_disks_v5_all.fits  31,368 rows (the VI-filtered union of the")
    print("                            two fitted halves -- the target for")
    print("                            comparability with v5b/v10b)")
    print("  DR2_TF_disks_v5.fits      32,488 rows (same union BEFORE VI filtering)")
    print(f"  this file                 {n} rows")


if __name__ == "__main__":
    main()
