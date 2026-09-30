#!/usr/bin/env python
"""Size-split null test: is a trend a genuine second parameter, or the selection?

The sample is selected on the HyperLeda B-band diameter D_L(25) > 20".  Galaxies far
above that threshold are essentially complete (S ~ 1); galaxies near it are strongly
incomplete.  So:

    a SELECTION-induced trend must be much stronger in the near-threshold subsample;
    a genuine PHYSICAL trend must be the same in both.

Two trends are tested:
  1. the surface-brightness dependence of the TF residual (the candidate second
     parameter: 11.4 sigma, 0.0187 dex per mag on the full sample);
  2. the shell offsets in eta (the Hubble-Bubble candidate).

Redshift is controlled inside each subsample separately, by subtracting the
inverse-variance mean residual in each redshift bin, so the comparison is of slopes
and shell contrasts, not of overall levels.  theta is the HyperLeda D25 -- the actual
selection variable -- not the SGA D26.
"""
import argparse

import numpy as np
from astropy.io import fits


def wls(x, y, w):
    """Weighted least squares slope with its error."""
    A = np.column_stack([np.ones_like(x), x - x.mean()])
    C = np.linalg.inv(A.T @ (A * w[:, None]))
    b = C @ (A.T @ (w * y))
    return b[1], np.sqrt(C[1, 1])


def zcontrol(z, eta, w, edges):
    out = eta.copy()
    for i in range(len(edges) - 1):
        s = (z >= edges[i]) & (z < edges[i + 1])
        if s.sum() > 20:
            out[s] -= np.sum(w[s] * eta[s]) / np.sum(w[s])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog",
                   default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    p.add_argument("--theta-complete", type=float, default=40.0,
                   help="theta above which selection is essentially complete, arcsec")
    p.add_argument("--theta-marginal", type=float, default=30.0,
                   help="theta below which selection bites, arcsec")
    args = p.parse_args()

    b = fits.open(args.catalog)[1].data
    d = b[b["ANALYSIS"].astype(bool)]
    z = np.asarray(d["Z_DESI_CMB"], float)
    eta = np.asarray(d["LOGDIST"], float)
    er = np.asarray(d["LOGDIST_ERR"], float)
    th = np.asarray(d["D25_LEDA"], float) * 60.0                    # selection variable
    sb = np.asarray(d["R_MAG_SB26_CORR"], float) + 5.0 * np.log10(
        np.asarray(d["D26"], float) * 60.0)

    g = (np.isfinite(z) & np.isfinite(eta) & np.isfinite(er) & (er > 0)
         & np.isfinite(th) & np.isfinite(sb) & (z > 1e-3))
    z, eta, er, th, sb = z[g], eta[g], er[g], th[g], sb[g]
    w = 1.0 / er ** 2
    zedges = np.linspace(z.min(), 0.065, 12)

    subs = {
        "complete  (theta > %g\")" % args.theta_complete: th > args.theta_complete,
        "marginal  (theta < %g\")" % args.theta_marginal: th < args.theta_marginal,
    }
    print(f"N = {len(z)} ANALYSIS galaxies;  theta = D25_LEDA (the selection variable)")
    print(f"full-sample sd(SB) = {sb.std():.3f} mag\n")

    print("1. SURFACE-BRIGHTNESS SLOPE  d<eta>/d(SB), z-controlled within each subsample")
    print(f"   {'subsample':>26} {'N':>6} {'<theta>':>8} {'<z>':>7} {'slope':>10} {'err':>8} {'sig':>6}")
    res = {}
    for lab, m in subs.items():
        y = zcontrol(z[m], eta[m], w[m], zedges)
        s, se = wls(sb[m], y, w[m])
        res[lab] = (s, se)
        print(f"   {lab:>26} {m.sum():6d} {th[m].mean():8.1f} {z[m].mean():7.4f} "
              f"{s:+10.5f} {se:8.5f} {s/se:+6.1f}")
    (s1, e1), (s2, e2) = res.values()
    dd, de = s1 - s2, np.hypot(e1, e2)
    print(f"   {'difference':>26} {'':6} {'':8} {'':7} {dd:+10.5f} {de:8.5f} {dd/de:+6.1f}")
    print("   -> selection would make the marginal subsample steeper; physical would match.\n")

    print("2. SHELL OFFSETS in eta, relative to the 0.032-0.040 shell, within each subsample")
    shells = [0.0, 0.008, 0.014, 0.020, 0.026, 0.032, 0.040]
    print(f"   {'shell':>14} " + " ".join(f"{k.split('(')[0].strip():>22}" for k in subs))
    ref = {}
    for lab, m in subs.items():
        s = m & (z >= shells[-2]) & (z < shells[-1])
        ref[lab] = np.sum(w[s] * eta[s]) / np.sum(w[s]) if s.sum() > 10 else np.nan
    for i in range(len(shells) - 2):
        row = f"   {shells[i]:6.3f}-{shells[i+1]:<7.3f}"
        for lab, m in subs.items():
            s = m & (z >= shells[i]) & (z < shells[i + 1])
            if s.sum() < 10:
                row += f" {'n/a':>22}"
                continue
            mu = np.sum(w[s] * eta[s]) / np.sum(w[s])
            se = np.sqrt(1.0 / np.sum(w[s]))
            row += f" {mu - ref[lab]:+11.5f}+/-{se:<8.5f}"
        print(row)
    print("   -> a bubble is a property of distance and must appear in both.")


if __name__ == "__main__":
    main()
