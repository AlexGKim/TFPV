#!/usr/bin/env python
"""Build a v9 run directory from a v8 one: elliptical surface brightness + inclination.

Replaces the v8 surface-brightness variable

    s_v8 = m_r,CORR + 5 log10(theta26)                       [circular, dust-corrected]

with the SGA-2020 3.2.3 elliptical quantity built from the RAW magnitude

    s_v9 = m_r,RAW  + 2.5 log10(pi * a26^2 * b/a)            [elliptical, raw]

and adds the inclination covariate q = 2.5 log10(b/a).

The raw magnitude is used deliberately. R_INTDUST_CORR is 97% correlated with q
(slope -0.3005), so a dust-corrected s puts ~0.30q back into the variable and
corr(s, q) rises from +0.03 to ~+0.32. span{s_raw, q} == span{s_corr, q}, so the
model is the same either way -- only the conditioning differs, and raw is the
better-conditioned basis.
"""
import argparse
import json
import os

import numpy as np
from astropy.io import fits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="existing v8 run dir name under output/")
    p.add_argument("--dst", required=True, help="new v9 run dir name under output/")
    p.add_argument("--fits", default="data/DR2_TF_disks_v5.fits")
    args = p.parse_args()

    d = json.load(open(f"output/{args.src}/input.json"))
    y = np.asarray(d["y"], float)

    f = fits.open(args.fits)[1].data
    Y = np.asarray(f["R_ABSMAG_SB26_CORR"], float)
    if "train_sga_ids" in d:
        lut = {int(v): i for i, v in enumerate(np.asarray(f["SGA_ID"]))}
        idx = np.array([lut[int(v)] for v in d["train_sga_ids"]])
        how = "SGA_ID"
    else:
        # Older runs (the v5 irregular one) carry no train_sga_ids. y is
        # R_ABSMAG_SB26_CORR as a float64, so an exact value match identifies the
        # row; this is only accepted if every training galaxy matches EXACTLY ONE row.
        pos = {}
        for i, v in enumerate(Y):
            pos.setdefault(v, []).append(i)
        hits = [pos.get(v, []) for v in y]
        bad = sum(1 for h in hits if len(h) != 1)
        if bad:
            raise SystemExit(f"y-join ambiguous or missing for {bad} galaxies")
        idx = np.array([h[0] for h in hits])
        how = "exact y value (no train_sga_ids in source)"
    if np.max(np.abs(y - Y[idx])) > 1e-8:
        raise SystemExit("join failed against y")
    d["train_sga_ids"] = [int(v) for v in np.asarray(f["SGA_ID"])[idx]]
    print(f"  joined on {how}")

    mr = np.asarray(f["R_MAG_SB26"], float)[idx]            # RAW, deliberately
    emr = np.asarray(f["R_MAG_SB26_ERR"], float)[idx]
    sma = np.asarray(f["SMA_SB26"], float)[idx]             # arcsec
    esma = np.asarray(f["SMA_SB26_ERR"], float)[idx]
    ba = np.asarray(f["BA"], float)[idx]

    if not np.all(np.isfinite([mr, sma, ba]).all(axis=1) if False else
                  (np.isfinite(mr) & np.isfinite(sma) & np.isfinite(ba) & (ba > 0) & (sma > 0))):
        raise SystemExit("non-finite or non-positive SB ingredients in the training set")

    q = 2.5 * np.log10(ba)                                   # inclination covariate
    s = mr + 2.5 * np.log10(np.pi * sma ** 2 * ba)           # elliptical SB
    elogth = esma / sma / np.log(10.0)
    sigma_s = np.hypot(emr, 5.0 * elogth)

    print(f"src {args.src} -> dst {args.dst}   N = {len(y)}")
    print(f"  s_v9 (elliptical, raw) : median {np.median(s):+.3f}  sd {s.std():.3f}")
    print(f"  q = 2.5log10(b/a)      : median {np.median(q):+.3f}  sd {q.std():.3f}")
    print(f"  corr(s_v9, q) = {np.corrcoef(s, q)[0,1]:+.3f}")
    if "s" in d:                      # src was a v8 run: show what the change bought
        s_old = np.asarray(d["s"], float)
        print(f"  s_v8 (circular, corr)  : median {np.median(s_old):+.3f}  sd {s_old.std():.3f}")
        print(f"  corr(s_v8, q) = {np.corrcoef(s_old, q)[0,1]:+.3f}   <- the point of the change")
    print(f"  sigma_s: median {np.median(sigma_s):.4f}")

    d["s"] = s.tolist()
    d["sigma_s"] = sigma_s.tolist()
    d["q"] = q.tolist()
    d["s_bar_obs"] = float(np.mean(y - s))
    for k in ("z", "sigma_z", "g", "sigma_g", "c_bar_obs", "c_bar_g_obs",
              "sel_c", "theta_r"):
        d.pop(k, None)
    print(f"  s_bar_obs = {d['s_bar_obs']:.4f}")

    os.makedirs(f"output/{args.dst}", exist_ok=True)
    json.dump(d, open(f"output/{args.dst}/input.json", "w"))

    init = {
        "slope_std": -0.97323519,
        "intercept_std": [-19.901533],
        "sigma_int_x": 0.05,
        "L11": 0.43,          # sigma_int_y from v8
        "L21": 0.0,           # neutral: do not seed the coupling under test
        "L22": 0.49,          # SB mode norm from v8
        "delta_s": 0.0,
        "mu_s": d["s_bar_obs"],
        "gamma_inc": 0.0,
        "alpha_kcorr_r": -5.904099,
        "alpha_kcorr_s": 0.0,
    }
    json.dump(init, open(f"output/{args.dst}/init_v9.json", "w"), indent=2)
    print(f"  wrote output/{args.dst}/input.json and init_v9.json")


if __name__ == "__main__":
    main()
