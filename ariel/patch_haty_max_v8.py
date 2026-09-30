#!/usr/bin/env python
"""Replace the scalar haty_max in a run's input.json with the v8 per-galaxy vector.

    haty_max[n] = min( dwarf_cut , y_n + 5 log10(theta26_n / theta_r) )

The second term is the r-band diameter cut written as a bound on absolute magnitude:
theta26 > theta_r  <=>  y < SB_r - mu - 5 log10(theta_r), and that expression equals
y_n + 5 log10(theta26_n/theta_r) identically, so it is satisfied by construction for
every galaxy that passed the cut.  The dwarf cut is kept because it is a deliberate
population definition, not a survey selection.

Galaxies are matched to the parent FITS on train_sga_ids, which input.json carries in
the same order as its data arrays (verified exact on y).
"""
import argparse
import json
import shutil

import numpy as np
from astropy.io import fits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--fits", required=True, help="parent FITS the run was built from")
    p.add_argument("--theta-r", type=float, default=35.0)
    args = p.parse_args()

    ipath = f"output/{args.run}/input.json"
    d = json.load(open(ipath))
    if isinstance(d["haty_max"], list):
        raise SystemExit("haty_max is already a vector; nothing to do")
    dwarf = float(d["haty_max"])
    y = np.asarray(d["y"], float)
    ids = np.asarray(d["train_sga_ids"])

    f = fits.open(args.fits)[1].data
    lut = {int(s): i for i, s in enumerate(np.asarray(f["SGA_ID"]))}
    idx = np.array([lut[int(s)] for s in ids])
    th = np.asarray(f["D26"], float)[idx] * 60.0
    ycat = np.asarray(f["R_ABSMAG_SB26_CORR"], float)[idx]
    if np.max(np.abs(y - ycat)) > 1e-8:
        raise SystemExit("join failed: y does not match the matched catalogue rows")

    theta_bound = y + 5.0 * np.log10(th / args.theta_r)
    hm = np.minimum(dwarf, theta_bound)

    viol = int((y > hm).sum())
    print(f"run {args.run}:  N={len(y)}  theta_r={args.theta_r:g}\"")
    print(f"  dwarf cut {dwarf:+.2f}; theta bound: min {theta_bound.min():+.2f} "
          f"median {np.median(theta_bound):+.2f} max {theta_bound.max():+.2f}")
    print(f"  haty_max: min {hm.min():+.2f} median {np.median(hm):+.2f} max {hm.max():+.2f}")
    print(f"  theta bound binds (is brighter than the dwarf cut) for "
          f"{100*(theta_bound < dwarf).mean():.1f}% of galaxies")
    print(f"  violations (y > haty_max): {viol}  <-- must be 0")
    if viol:
        raise SystemExit("bound violated; refusing to write")
    if not (d["y_min"] < hm.min()):
        raise SystemExit(f"y_min={d['y_min']} must be below min(haty_max)={hm.min()}")
    if not (d["y_max"] > hm.max()):
        raise SystemExit(f"y_max={d['y_max']} must exceed max(haty_max)={hm.max()}")
    if not (hm.min() > d["haty_min"]):
        raise SystemExit(f"min(haty_max)={hm.min()} must exceed haty_min={d['haty_min']}")

    shutil.copy(ipath, ipath + ".bak_scalar_hatymax")
    d["haty_max"] = hm.tolist()
    json.dump(d, open(ipath, "w"))
    print(f"  wrote vector haty_max to {ipath} (scalar backed up)")


if __name__ == "__main__":
    main()
