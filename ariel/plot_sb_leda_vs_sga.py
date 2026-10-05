#!/usr/bin/env python
"""LEDA B-band mean surface brightness against SGA r-band mean surface brightness.

Both use the ELLIPTICAL isophotal area, following SGA-2020 section 3.2.3:

    <mu_r>_26 = R_MAG_SB26 + 2.5 log10( pi * SMA_SB26^2 * BA )
    <mu_B>_25 = MAG_LEDA   + 2.5 log10( pi * a25^2 * BA_LEDA ),  a25 = D25_LEDA/2

Note the catalogue's own SB_D25_LEDA column is the CIRCULAR-aperture version
(bt + 2.5 log10(pi a25^2), i.e. HyperLeda's bri25); it is 2.5 log10(BA_LEDA) = 0.78 mag
fainter in the median and carries 0.44 mag of pure axis-ratio spread, so it is not used.

Magnitudes are raw (not MW-dust corrected) on both axes, since MAG_LEDA is HyperLeda's
observed bt.  The r-band MW correction is only 0.074 mag in the median.

The offset between the axes is physical and expected: a B-r colour of ~1 mag plus the
25-vs-26 mag/arcsec^2 isophote difference, which enlarges the r-band aperture.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fits", default="data/DR2_TF_disks_v5.fits")
    p.add_argument("--corrected", action="store_true",
                   help="use R_MAG_SB26_CORR instead of the raw R_MAG_SB26")
    p.add_argument("--out", default="output/sb_leda_vs_sga.png")
    args = p.parse_args()

    t = fits.open(args.fits)[1].data

    # ---- SGA r-band, elliptical area inside the 26 mag/arcsec^2 isophote -----
    sma = np.asarray(t["SMA_SB26"], float)                   # arcsec, = D26*60/2
    ba = np.asarray(t["BA"], float)
    mr = np.asarray(t["R_MAG_SB26_CORR" if args.corrected else "R_MAG_SB26"], float)
    mu_r = mr + 2.5 * np.log10(np.pi * sma ** 2 * ba)

    # ---- LEDA B-band, elliptical area inside the 25 mag/arcsec^2 isophote ----
    a25 = np.asarray(t["D25_LEDA"], float) * 60.0 / 2.0      # arcmin -> arcsec
    ba_l = np.asarray(t["BA_LEDA"], float)
    bt = np.asarray(t["MAG_LEDA"], float)
    mu_b = bt + 2.5 * np.log10(np.pi * a25 ** 2 * ba_l)

    g = (np.isfinite(mu_r) & np.isfinite(mu_b) & (sma > 0) & (ba > 0)
         & (a25 > 0) & (ba_l > 0))
    mu_r, mu_b = mu_r[g], mu_b[g]
    d = mu_b - mu_r

    r = np.corrcoef(mu_r, mu_b)[0, 1]
    sl = np.polyfit(mu_r, mu_b, 1)[0]
    print(f"N = {g.sum()}")
    print(f"<mu_r>_26 : med {np.median(mu_r):.3f}  sd {mu_r.std():.3f}")
    print(f"<mu_B>_25 : med {np.median(mu_b):.3f}  sd {mu_b.std():.3f}")
    print(f"corr = {r:+.3f}   OLS slope = {sl:+.3f}")
    print(f"mu_B - mu_r : med {np.median(d):+.3f}  sd {d.std():.3f}  "
          f"NMAD {1.4826*np.median(np.abs(d-np.median(d))):.3f}")

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.scatter(mu_r, mu_b, s=2.2, c="#1f6fb2", alpha=0.10, lw=0,
               rasterized=True, zorder=2)

    lo = min(np.percentile(mu_r, 0.1), np.percentile(mu_b, 0.1)) - 0.25
    hi = max(np.percentile(mu_r, 99.9), np.percentile(mu_b, 99.9)) + 0.25
    lim = (lo, hi)
    xx = np.linspace(*lim, 50)
    ax.plot(xx, xx, color="#555555", lw=1.3, ls=(0, (5, 3)), zorder=4, label="1:1")
    off = np.median(d)
    ax.plot(xx, xx + off, color="#d1670b", lw=2.0, zorder=5,
            label=rf"1:1 $+\ {off:.2f}$ mag (median offset)")

    # binned median of mu_B in bins of mu_r
    edges = np.percentile(mu_r, np.linspace(0.5, 99.5, 22))
    cen, med = [], []
    for i in range(len(edges) - 1):
        m = (mu_r >= edges[i]) & (mu_r < edges[i + 1])
        if m.sum() < 30:
            continue
        cen.append(np.median(mu_r[m]))
        med.append(np.median(mu_b[m]))
    ax.plot(cen, med, color="#111111", lw=2.2, marker="o", ms=4.0, zorder=6,
            label="binned median")

    band = "dust-corrected" if args.corrected else "raw"
    ax.set_xlabel(r"SGA $\langle\mu_r\rangle_{26}"
                  r" = m_{r,26} + 2.5\log_{10}(\pi\,a_{26}^2\,b/a)$"
                  f"   [mag arcsec$^{{-2}}$, {band}]")
    ax.set_ylabel(r"HyperLeda $\langle\mu_B\rangle_{25}"
                  r" = b_T + 2.5\log_{10}(\pi\,a_{25}^2\,(b/a)_{\rm LEDA})$"
                  r"   [mag arcsec$^{-2}$]")
    ax.set_title("Mean surface brightness inside the elliptical isophote:\n"
                 f"HyperLeda B(25) vs SGA r(26)   "
                 f"$N$ = {g.sum():,}, $r$ = {r:+.2f}, "
                 rf"sd($\mu_B-\mu_r$) = {d.std():.2f} mag")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
