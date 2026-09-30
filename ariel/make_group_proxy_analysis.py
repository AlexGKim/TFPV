#!/usr/bin/env python
"""Group-proxy ANALYSIS sample: keep one random member per FoF multiplet.

The fits are NOT rerun -- this is a post-hoc selection on an existing combined catalog.
Starting from the ANALYSIS==True rows, every FoF multiplet in
output/DR2_TF_v5_2color_grouped/multiplets_members.csv (1 h^-1 Mpc transverse,
1000 km/s) is reduced to a single randomly chosen member and its other members are
dropped.  Galaxies in no multiplet are kept untouched.

This is the discard-redundant-members counterpart to DESI-DR2_TF_pv_cat_v5b_grouped.fits,
which instead GLS-averages each system into one object.  Both break intra-group
correlation; this one costs information but assumes nothing about the covariance.

Multiplet membership was defined on the v5b ANALYSIS sample.  Membership is a property
of sky position and redshift, so it is reused as-is, but the train/analysis split differs
between fits -- members that landed in THIS fit's training set are simply not eligible.

Covariance is carried through as the corresponding sub-block, in mag^2 (distance modulus);
divide by 25 for logdist^2.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import h5py
from astropy.io import fits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="output/DR2_TF_v5_3_2color_combined")
    p.add_argument("--catalog", default="DESI-DR2_TF_pv_cat_v5_3.fits")
    p.add_argument("--members", default="output/DR2_TF_v5_2color_grouped/multiplets_members.csv")
    p.add_argument("--out-dir", default="output/DR2_TF_v5_3_gproxy")
    p.add_argument("--out-name", default="DESI-DR2_TF_pv_cat_v5_3_gproxy.fits")
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--chunk", type=int, default=2000)
    args = p.parse_args()

    cat_path = os.path.join(args.run_dir, args.catalog)
    cov_path = cat_path.replace(".fits", "_cov.h5")
    os.makedirs(args.out_dir, exist_ok=True)

    data = fits.open(cat_path)[1].data
    n = len(data)
    analysis = np.asarray(data["ANALYSIS"], dtype=bool)
    print(f"parent {cat_path}: {n} MAIN rows, {analysis.sum()} ANALYSIS")

    # --- multiplet membership, restricted to this fit's ANALYSIS rows -----------
    rows = list(csv.DictReader(open(args.members)))
    sys_rows = defaultdict(list)
    for r in rows:
        i = int(r["ROW_IN_Y3_H5"])
        if analysis[i]:
            sys_rows[int(r["GROUP_ID"])].append(i)
    # sanity: the row index must address the same galaxy
    idx_all = np.array([int(r["ROW_IN_Y3_H5"]) for r in rows])
    sga_all = np.array([int(r["SGA_ID"]) for r in rows])
    if not np.array_equal(np.asarray(data["SGA_ID"])[idx_all], sga_all):
        raise SystemExit("ROW_IN_Y3_H5 does not address the same galaxies in this catalog")

    n_elig = sum(len(v) for v in sys_rows.values())
    multi = {g: v for g, v in sys_rows.items() if len(v) > 1}
    print(f"multiplets: {len(rows)} members in {len(set(int(r['GROUP_ID']) for r in rows))} systems; "
          f"{n_elig} are in this fit's ANALYSIS, forming {len(sys_rows)} systems "
          f"({len(multi)} with >1 member)")

    rng = np.random.default_rng(args.seed)
    drop = set()
    group_of = {}
    nmem_of = {}
    for g in sorted(sys_rows):
        mem = sorted(sys_rows[g])
        pick = int(rng.choice(mem))
        for i in mem:
            group_of[i] = g
            nmem_of[i] = len(mem)
            if i != pick:
                drop.add(i)
    keep = analysis.copy()
    keep[list(drop)] = False
    print(f"dropped {len(drop)} redundant members -> {keep.sum()} rows "
          f"({100*len(drop)/analysis.sum():.2f}% of ANALYSIS)")

    rows_keep = np.flatnonzero(keep)

    # --- catalog ---------------------------------------------------------------
    gid = np.full(n, -1, dtype=np.int32)
    nmem = np.ones(n, dtype=np.int16)
    for i, g in group_of.items():
        gid[i] = g
        nmem[i] = nmem_of[i]
    # Subset first, then rebuild the columns from plain arrays. Passing the parent
    # ColDefs straight to from_columns hands astropy weakrefs into an HDUList that is
    # already collectable, which raises ReferenceError.
    src = data[rows_keep]
    cols = [fits.Column(name=nm, format=data.columns[nm].format,
                        array=np.asarray(src[nm]))
            for nm in data.columns.names]
    cols += [fits.Column(name="FOF_ID", format="J", array=gid[rows_keep]),
             fits.Column(name="FOF_N_MEM", format="I", array=nmem[rows_keep])]
    out_cat = os.path.join(args.out_dir, args.out_name)
    sub = fits.BinTableHDU.from_columns(cols)
    sub.header["GPROXY"] = (True, "one random member kept per FoF multiplet")
    sub.header["GPSEED"] = (args.seed, "RNG seed for member choice")
    sub.header["GPPARENT"] = (args.catalog, "parent catalog")
    sub.writeto(out_cat, overwrite=True)
    print(f"wrote {out_cat}  ({len(rows_keep)} rows)")

    # --- covariance sub-block (mag^2) -----------------------------------------
    out_cov = out_cat.replace(".fits", "_cov.h5")
    with h5py.File(cov_path, "r") as fin, h5py.File(out_cov, "w") as fout:
        src = fin["cov"]
        m = len(rows_keep)
        dst = fout.create_dataset("cov", shape=(m, m), dtype=src.dtype)
        for a in range(0, m, args.chunk):
            b = min(a + args.chunk, m)
            block = src[rows_keep[a:b], :]          # (chunk, n)
            dst[a:b, :] = block[:, rows_keep]
        fout.create_dataset("sga_id", data=np.asarray(data["SGA_ID"])[rows_keep])
        fout.create_dataset("fof_id", data=gid[rows_keep])
        fout.create_dataset("population", data=fin["population"][:][rows_keep]
                            if "population" in fin else np.zeros(m, np.int8))
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["row_order"] = (f"ANALYSIS rows of {args.out_name} in this directory, "
                                   "parent row order preserved; one random member per FoF multiplet")
        fout.attrs["units"] = "mag^2 (distance modulus); divide by 25 for logdist^2"
        fout.attrs["gproxy_seed"] = args.seed

    # --- checks ----------------------------------------------------------------
    with h5py.File(out_cov, "r") as f:
        d = np.array([f["cov"][i, i] for i in range(0, min(500, len(rows_keep)))])
    mu_err = np.asarray(data["MU_ERR"])[rows_keep][:len(d)]
    print(f"check: max|diag - MU_ERR^2| = {np.max(np.abs(d - mu_err**2)):.3e}")
    g_keep = gid[rows_keep]
    g_multi = g_keep[g_keep >= 0]
    print(f"check: duplicate groups remaining = {len(g_multi) - len(set(g_multi))} (must be 0)")
    print(f"wrote {out_cov}")


if __name__ == "__main__":
    main()
