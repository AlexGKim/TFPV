#!/usr/bin/env python
"""Add the surface-brightness state variable to a run's input.json (v8).

Adds three arrays, joined to the parent FITS on train_sga_ids (verified exact on y):

    s        = m_r + 5 log10(theta26)          r-band surface brightness, distance-free
    sigma_s  = hypot(sigma_mr, 5 sigma_logtheta)
    sel_c    = 5 log10(theta_r) + mu(z)        selection constant, so the r-band
                                               diameter cut is  s - yhat > sel_c

desi_data.py is left untouched: its selection logic threads z/g through three separate
append sites, and the join is exact, so post-processing is the lower-risk route.
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
    p.add_argument("--fits", required=True)
    p.add_argument("--theta-r", type=float, default=35.0)
    args = p.parse_args()

    ipath = f"output/{args.run}/input.json"
    d = json.load(open(ipath))
    y = np.asarray(d["y"], float)
    ids = np.asarray(d["train_sga_ids"])

    f = fits.open(args.fits)[1].data
    lut = {int(v): i for i, v in enumerate(np.asarray(f["SGA_ID"]))}
    idx = np.array([lut[int(v)] for v in ids])
    if np.max(np.abs(y - np.asarray(f["R_ABSMAG_SB26_CORR"], float)[idx])) > 1e-8:
        raise SystemExit("join failed against y")

    mr = np.asarray(f["R_MAG_SB26_CORR"], float)[idx]
    emr = np.asarray(f["R_MAG_SB26_ERR_CORR"], float)[idx]
    sma = np.asarray(f["SMA_SB26"], float)[idx]
    esma = np.asarray(f["SMA_SB26_ERR"], float)[idx]
    th = 2.0 * sma
    elogth = esma / sma / np.log(10.0)

    s = mr + 5.0 * np.log10(th)
    sigma_s = np.hypot(emr, 5.0 * elogth)
    mu = mr - y                                   # mu(z), exactly as the catalogue has it
    sel_c = 5.0 * np.log10(args.theta_r) + mu

    margin = (s - y) - sel_c                      # = 5 log10(theta26/theta_r)
    print(f"run {args.run}  N={len(y)}  theta_r={args.theta_r:g}\"")
    print(f"  s       : median {np.median(s):.3f}  sd {s.std():.3f}")
    print(f"  sigma_s : median {np.median(sigma_s):.4f}  (m_r {np.median(emr):.4f}, "
          f"size {np.median(5*elogth):.4f})")
    print(f"  margin (s - y - sel_c) = 5log10(theta26/theta_r): min {margin.min():+.4f} "
          f"median {np.median(margin):+.4f}")
    if margin.min() <= 0:
        raise SystemExit(f"{int((margin<=0).sum())} galaxies violate the theta_r cut")
    print("  all galaxies satisfy the cut (margin > 0), as required")

    shutil.copy(ipath, ipath + ".bak_nosb")
    d["s"] = s.tolist()
    d["sigma_s"] = sigma_s.tolist()
    d["sel_c"] = sel_c.tolist()
    d["theta_r"] = float(args.theta_r)
    json.dump(d, open(ipath, "w"))
    print(f"  wrote s, sigma_s, sel_c to {ipath}")


if __name__ == "__main__":
    main()
