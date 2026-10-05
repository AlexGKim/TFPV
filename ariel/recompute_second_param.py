#!/usr/bin/env python
"""Second-parameter statistics for the TF relation, recomputed AT FIXED ROTATION VELOCITY.

A second parameter of the Tully-Fisher relation is by definition a dependence of the
distance residual on some quantity at FIXED logV. The statistics reported earlier in
this analysis (the "11.4 sigma" surface-brightness trend, the aperture decomposition,
the size-split null) were computed without controlling for logV. Since
corr(SB, logV) = -0.54 on the DR2 analysis sample, that conditioning matters: the
marginal and partial slopes have opposite signs.

Everything below uses the same machinery as the original analysis -- inverse-variance
weights 1/sigma_eta^2 and a redshift control that removes the weighted mean within
z bins -- and adds logV to the design matrix.
"""
import argparse

import numpy as np
from astropy.io import fits


def wls(X, y, w):
    """Weighted LS. Returns coefficients (excluding intercept) and their errors."""
    A = np.column_stack([np.ones(len(y))] + [c - np.average(c, weights=w) for c in X])
    XtW = A.T * w
    C = np.linalg.inv(XtW @ A)
    b = C @ (XtW @ y)
    r = y - A @ b
    dof = len(y) - A.shape[1]
    scale = (w * r * r).sum() / dof / np.mean(w)      # scale errors by the fit
    return b[1:], np.sqrt(np.diag(C)[1:] * np.mean(w) * scale)


def zcontrol(z, y, w, edges):
    out = y.copy()
    for i in range(len(edges) - 1):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if m.sum() > 20:
            out[m] -= np.sum(w[m] * out[m]) / np.sum(w[m])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat",
                   default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    args = p.parse_args()

    t = fits.open(args.cat)[1].data
    a = t["ANALYSIS"].astype(bool)
    mr = np.asarray(t["R_MAG_SB26"], float)
    sma = np.asarray(t["SMA_SB26"], float)
    ba = np.asarray(t["BA"], float)
    x = np.asarray(t["LOGVROT"], float)
    eta = np.asarray(t["LOGDIST"], float)
    err = np.asarray(t["LOGDIST_ERR"], float)
    z = np.asarray(t["Z_DESI_CMB"], float)
    d25 = np.asarray(t["D25_LEDA"], float) * 60.0

    s_ell = mr + 2.5 * np.log10(np.pi * sma ** 2 * ba)
    s_cir = mr + 5.0 * np.log10(2.0 * sma)
    q = 2.5 * np.log10(ba)

    g = (a & np.isfinite(eta) & np.isfinite(err) & (err > 0) & np.isfinite(z)
         & np.isfinite(x) & np.isfinite(s_ell) & np.isfinite(q) & np.isfinite(d25))
    dmu = 5.0 * eta[g]
    w = 1.0 / (5.0 * err[g]) ** 2
    s_ell, s_cir, q, x, z, d25 = s_ell[g], s_cir[g], q[g], x[g], z[g], d25[g]
    y = zcontrol(z, dmu, w, np.linspace(z.min(), 0.065, 12))

    print(f"N = {g.sum()} ANALYSIS galaxies;  corr(s_ell, logV) = "
          f"{np.corrcoef(s_ell, x)[0,1]:+.3f}\n")

    b, e = wls([x], y, w)
    print("0. Is the distance residual already independent of logV?")
    print(f"   d(dmu)/dlogV = {b[0]:+.4f} +/- {e[0]:.4f}  ({b[0]/e[0]:+.1f} sigma)")
    print("   (if this were 0 the conditioning would not matter; it is not)\n")

    print("1. SURFACE-BRIGHTNESS SLOPE  d(dmu)/ds  [elliptical, raw]")
    print(f"   {'conditioning':<34} {'slope':>9} {'err':>8} {'sigma':>7}")
    for lab, X in [("marginal (as reported earlier)", [s_ell]),
                   ("+ inclination", [s_ell, q]),
                   ("+ logV", [s_ell, x]),
                   ("+ logV + inclination", [s_ell, x, q])]:
        b, e = wls(X, y, w)
        print(f"   {lab:<34} {b[0]:+9.4f} {e[0]:8.4f} {b[0]/e[0]:+7.1f}")

    print("\n2. APERTURE DECOMPOSITION, now at fixed logV")
    for lab, X in [("circular,   at fixed logV", [s_cir, x]),
                   ("elliptical, at fixed logV", [s_ell, x]),
                   ("elliptical + incl, fixed logV", [s_ell, x, q])]:
        b, e = wls(X, y, w)
        extra = f"   d/dq = {b[2]:+.4f} +/- {e[2]:.4f}" if len(b) > 2 else ""
        print(f"   {lab:<32} d/ds = {b[0]:+.4f} +/- {e[0]:.4f}{extra}")

    print("\n3. SIZE-SPLIT NULL at fixed logV "
          "(selection acts through angular size; a real second parameter does not)")
    print(f"   {'subsample':<28} {'N':>6} {'d(dmu)/ds':>11} {'err':>8} {'sigma':>7}")
    res = {}
    for lab, m in [("complete  (D25 > 40\")", d25 > 40.0),
                   ("marginal  (D25 < 30\")", d25 < 30.0)]:
        yy = zcontrol(z[m], y[m], w[m], np.linspace(z[m].min(), 0.065, 10))
        b, e = wls([s_ell[m], x[m]], yy, w[m])
        res[lab] = (b[0], e[0])
        print(f"   {lab:<28} {m.sum():6d} {b[0]:+11.4f} {e[0]:8.4f} {b[0]/e[0]:+7.1f}")
    (s1, e1), (s2, e2) = res.values()
    d, de = s1 - s2, np.hypot(e1, e2)
    print(f"   {'difference':<28} {'':6} {d:+11.4f} {de:8.4f} {d/de:+7.1f}")
    print("   -> if selection-driven the marginal subsample is steeper; "
          "if physical they agree.")


if __name__ == "__main__":
    main()
