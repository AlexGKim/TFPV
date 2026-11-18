#!/usr/bin/env python3
"""Plot D26 and D25 against redshift, side by side, in physical kpc.

D26  : SGA r-band isophotal diameter at 26 mag/arcsec^2 (column D26, arcmin;
       D26_kpc is the same thing already in kpc).
D25  : LEDA B-band diameter at 25 mag/arcsec^2 (column D25_LEDA, arcmin).

The two come from different catalogs, different bands and different isophote
levels, so they are independent measures of size for the same galaxies. If both
show the same rise with redshift, the trend is a property of the sample selection
rather than of SGA photometry.

Unit handling: only D26 has a kpc column, so D25 is converted with the catalog's
OWN per-galaxy angular-to-physical factor,

    kpc_per_arcmin = D26_kpc / D26        (= C * D_A(z), whatever C and cosmology
                                            the catalog used)
    D25_kpc = D25_LEDA * kpc_per_arcmin

This keeps the two exactly on the same distance scale without having to reproduce
that convention -- which is worth doing explicitly, since D26_kpc / (D_A * theta)
comes out at ~0.732 for Planck18 rather than 1.

Usage:
    python3 plot_d25_d26_vs_z.py [FITS] [-o OUT.png] [--zcol Z_DESI_CMB] [--main-only]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

DEFAULT_FITS = "output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits"


def median_trend(z, y, good, nbins=20, mincount=20):
    edges = np.linspace(z[good].min(), z[good].max(), nbins + 1)
    cen, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = good & (z >= a) & (z < b)
        if m.sum() >= mincount:
            cen.append(0.5 * (a + b))
            med.append(np.median(y[m]))
    return np.array(cen), np.array(med)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fits", nargs="?", default=DEFAULT_FITS)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--zcol", default="Z_DESI_CMB")
    ap.add_argument("--main-only", action="store_true")
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.fits)),
                                   "d25_d26_vs_redshift.png")
    out = os.path.abspath(out)

    with fits.open(args.fits) as h:
        data = h[1].data
        names = set(data.dtype.names or ())
        need = ("D26", "D26_kpc", "D25_LEDA", args.zcol)
        missing = [c for c in need if c not in names]
        if missing:
            raise SystemExit(f"ERROR: missing column(s) {missing} in {args.fits}")
        g_ = lambda c: np.asarray(data[c], float)
        d26_am, d26_kpc, d25_am = g_("D26"), g_("D26_kpc"), g_("D25_LEDA")
        z = g_(args.zcol)
        main = np.asarray(data["MAIN"], bool) if "MAIN" in names else np.ones(len(z), bool)

    kpc_per_arcmin = np.where(d26_am > 0, d26_kpc / d26_am, np.nan)
    d25_kpc = d25_am * kpc_per_arcmin

    good = (np.isfinite(z) & (z > 0) & np.isfinite(d26_kpc) & (d26_kpc > 0)
            & np.isfinite(d25_kpc) & (d25_kpc > 0))
    if args.main_only:
        good &= main

    print(f"file      : {args.fits}")
    print(f"rows      : {len(z)}   plotted: {int(good.sum())}")
    print(f"kpc/arcmin: median {np.nanmedian(kpc_per_arcmin[good]):.3f}")
    print(f"D25/D26 (angular): median {np.median(d25_am[good]/d26_am[good]):.4f}\n")

    lz = np.log1p(z[good])
    print("               median@z<0.010  median@z>0.058   growth   dlnD/dln(1+z)")
    series = (("D26 (SGA r, 26)", d26_kpc, "#3366cc"),
              ("D25 (LEDA B, 25)", d25_kpc, "#cc3366"))
    for lbl, y, _c in series:
        lo = np.median(y[good & (z < 0.010)])
        hi = np.median(y[good & (z > 0.058)])
        s = np.polyfit(lz, np.log(y[good]), 1)[0]
        print(f"  {lbl:16s} {lo:9.2f}      {hi:9.2f}     {hi/lo:6.2f}x   {s:+8.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), constrained_layout=True)
    for ax, (lbl, y, c) in zip(axes[:2], series):
        ax.scatter(z[good], y[good], s=4, alpha=0.25, linewidths=0, color=c)
        cen, med = median_trend(z, y, good)
        ax.plot(cen, med, color="black", lw=2.2, label="median in $z$ bins")
        ax.set_yscale("log")
        ax.set_xlabel(rf"$z$   ({args.zcol})")
        ax.set_ylabel("diameter  [kpc]")
        ax.set_title(lbl)
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="lower right", fontsize=9)

    # direct comparison of the two trends, normalised to their own low-z value
    ax = axes[2]
    for lbl, y, c in series:
        cen, med = median_trend(z, y, good)
        ax.plot(cen, med / med[0], color=c, lw=2.2, label=lbl)
    ax.set_xlabel(rf"$z$   ({args.zcol})")
    ax.set_ylabel("median diameter / its value in the first $z$ bin")
    ax.set_title("relative growth with redshift")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(os.path.basename(args.fits)
                 + ("   (MAIN only)" if args.main_only else ""))
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
