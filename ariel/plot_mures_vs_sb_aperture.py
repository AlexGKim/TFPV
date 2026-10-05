#!/usr/bin/env python
"""Distance-modulus residual against r-band surface brightness: circular vs elliptical.

SUPERSEDED -- see recompute_second_param.py.  The slopes this script reports are
MARGINAL in rotation velocity: they control for redshift but not for logV.  Since
corr(SB, logV) = -0.54, and the estimated residual carries a 54-sigma artificial
dependence on logV from the truncated-normal shrinkage in mu_TF, the marginal slope is
dominated by that artifact.  An AbacusSummit mock with a synthetic variable having the
same logV correlation and NO real signal returns +0.297 +/- 0.010 marginally and
-0.003 +/- 0.014 at fixed logV.  The +0.105 this script reports is therefore smaller
than the pure artifact; at fixed logV the slope is -0.195 +/- 0.010.  The 59/41
surface-brightness/inclination decomposition below does not survive the correct
conditioning either.  Kept for the figures and the aperture comparison only.

Left:   mu_circ = m_r,26 + 2.5 log10( pi * a26^2      )   -- circular aperture
Right:  mu_ell  = m_r,26 + 2.5 log10( pi * a26^2 * b/a )   -- elliptical aperture (SGA-2020 3.2.3)

The two differ by EXACTLY 2.5 log10(b/a), a pure inclination term, so the change in slope
between the panels is the part of the apparent surface-brightness trend that is really
inclination.

Both panels use the RAW R_MAG_SB26.  The dust-corrected magnitude cannot be used here:
R_INTDUST_CORR is 97% correlated with 2.5 log10(b/a) (slope -0.30), so it would put part
of the inclination term into the "circular" definition and destroy the comparison.

y is the distance-modulus residual mu_zcmb - mu_TF = 5*LOGDIST, in magnitudes.  The
redshift trend is removed inside z bins first (inverse-variance weighted), so what is
plotted is the surface-brightness dependence alone and not the z drift.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wls(x, y, w):
    A = np.column_stack([np.ones_like(x), x - x.mean()])
    C = np.linalg.inv(A.T @ (A * w[:, None]))
    b = C @ (A.T @ (w * y))
    return b[1], np.sqrt(C[1, 1]), b[0]


def binned(x, y, w, edges, n_min=40):
    cen, mu, se = [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() < n_min:
            continue
        cen.append(np.median(x[m]))
        mu.append(np.sum(w[m] * y[m]) / np.sum(w[m]))
        se.append(np.sqrt(1.0 / np.sum(w[m])))
    return map(np.asarray, (cen, mu, se))


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
    p.add_argument("--corrected", action="store_true",
                   help="use R_MAG_SB26_CORR (NOT recommended -- see module docstring)")
    p.add_argument("--style", choices=("new", "old"), default="new",
                   help="'old' reproduces the axes of plot_sb_hypothesis.py: y in dex, "
                        "binned points only, autoscaled")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    old = args.style == "old"
    if args.out is None:
        args.out = ("output/mures_vs_sb_aperture_oldaxes.png" if old
                    else "output/mures_vs_sb_aperture.png")
    scale = 1.0 if old else 5.0          # eta [dex]  vs  5*eta = dmu [mag]
    unit = "dex" if old else "mag"

    t = fits.open(args.cat)[1].data
    mr = np.asarray(t["R_MAG_SB26_CORR" if args.corrected else "R_MAG_SB26"], float)
    sma = np.asarray(t["SMA_SB26"], float)          # arcsec
    ba = np.asarray(t["BA"], float)
    eta = np.asarray(t["LOGDIST"], float)
    err = np.asarray(t["LOGDIST_ERR"], float)
    z = np.asarray(t["Z_DESI_CMB"], float)
    anal = t["ANALYSIS"].astype(bool)

    area_c = np.pi * sma ** 2
    sb_c = mr + 2.5 * np.log10(area_c)
    sb_e = mr + 2.5 * np.log10(area_c * ba)

    g = (anal & np.isfinite(eta) & np.isfinite(err) & (err > 0) & np.isfinite(z)
         & np.isfinite(sb_c) & np.isfinite(sb_e) & (sma > 0) & (ba > 0))
    dmu = scale * eta[g]                     # distance-modulus residual
    w = 1.0 / (scale * err[g]) ** 2
    sb_c, sb_e, zz = sb_c[g], sb_e[g], z[g]

    zedges = np.linspace(zz.min(), 0.065, 12)
    y = zcontrol(zz, dmu, w, zedges)

    print(f"N = {g.sum()} ANALYSIS galaxies   "
          f"({'dust-corrected' if args.corrected else 'raw'} R_MAG_SB26)")
    print(f"mu_ell - mu_circ = 2.5log10(b/a): med {np.median(sb_e-sb_c):+.3f}  "
          f"sd {(sb_e-sb_c).std():.3f} mag")
    print()

    panels = [("circular aperture", sb_c, r"$\mu_{r,26}^{\rm circ}"
               r" = m_{r,26} + 2.5\log_{10}(\pi a_{26}^2)$", "#1f6fb2"),
              ("elliptical aperture", sb_e, r"$\langle\mu_r\rangle_{26}"
               r" = m_{r,26} + 2.5\log_{10}(\pi a_{26}^2\,b/a)$", "#0b8043")]

    width = 2.6
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6) if old else (14.2, 6.0),
                             sharey=True)
    out = {}
    for ax, (lab, x, xlab, col) in zip(axes, panels):
        sl, se, c0 = wls(x, y, w)
        out[lab] = (sl, se)
        print(f"{lab:>20}:  sd = {x.std():.3f} mag   "
              f"slope = {sl:+.5f} +/- {se:.5f} {unit}/mag   ({sl/se:+.1f} sigma)")

        ax.axhline(0.0, color="#9a9a96", lw=1.1, zorder=1)

        if old:
            # exactly the earlier figure's rendering: 13 percentile bins, points only
            e = np.percentile(x, np.linspace(1, 99, 13))
            cen, mu_, se_ = binned(x, y, w, e)
            ax.errorbar(cen, mu_, yerr=se_, fmt="o", color=col, ms=5, lw=0,
                        elinewidth=1.5, capsize=2.5, zorder=3,
                        label=rf"{lab}:  ${sl:+.4f}\pm{se:.4f}$ dex/mag  "
                              rf"(${sl/se:+.1f}\sigma$)")
            xs = np.linspace(e[0], e[-1], 40)
            ax.plot(xs, sl * (xs - x.mean()), color=col, lw=1.8, zorder=2)
        else:
            ax.scatter(x, y, s=2.2, c=col, alpha=0.09, lw=0, rasterized=True, zorder=2)
            e = np.percentile(x, np.linspace(1, 99, 15))
            cen, mu_, se_ = binned(x, y, w, e)
            ax.errorbar(cen, mu_, yerr=se_, fmt="o", ms=6.0, color="#111111",
                        ecolor="#111111", elinewidth=1.4, capsize=3.0, zorder=5,
                        label="inverse-variance binned mean")
            xc = np.median(x)
            xx = np.linspace(xc - width / 2, xc + width / 2, 20)
            ax.plot(xx, c0 + sl * (xx - x.mean()), color="#d1670b", lw=2.4, zorder=6,
                    label=rf"WLS slope = ${sl:+.3f}\pm{se:.3f}$  (${sl/se:+.1f}\sigma$)")
            ax.set_xlim(xc - width / 2, xc + width / 2)

        ax.set_xlabel(xlab + "   [mag arcsec$^{-2}$]")
        ax.set_title(lab, fontsize=12)
        ax.grid(alpha=0.18, lw=0.6)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    if old:
        axes[0].set_ylabel(r"$\langle\eta\rangle$, redshift-controlled  [dex]")
    else:
        axes[0].set_ylim(-0.85, 0.85)
        axes[0].set_ylabel(
            r"distance-modulus residual   $\mu_{z} - \mu_{\rm TF}$   [mag]")
    # --- decomposition: mu_circ = mu_ell - 2.5log10(b/a), the two near-orthogonal ---
    inc = 2.5 * np.log10(ba[g])
    A = np.column_stack([np.ones_like(y), sb_e - sb_e.mean(), inc - inc.mean()])
    C = np.linalg.inv(A.T @ (A * w[:, None]))
    b = C @ (A.T @ (w * y))
    se_ = np.sqrt(np.diag(C))
    print(f"\ncorr(mu_ell, 2.5log10(b/a)) = {np.corrcoef(sb_e, inc)[0,1]:+.3f}  "
          f"-> the two carry nearly independent information")
    print(f"joint fit:  d/d(mu_ell) = {b[1]:+.4f} +/- {se_[1]:.4f} "
          f"({b[1]/se_[1]:+.1f} sigma)   "
          f"d/d(2.5log10 b/a) = {b[2]:+.4f} +/- {se_[2]:.4f} ({b[2]/se_[2]:+.1f} sigma)")
    pred = (b[1] * sb_e.var() - b[2] * inc.var()) / sb_c.var()
    print(f"circular slope reconstructed from the two: {pred:+.4f}  "
          f"vs {out['circular aperture'][0]:+.4f} measured")
    frac = b[1] * sb_e.var() / (b[1] * sb_e.var() - b[2] * inc.var())
    print(f"  -> {100*frac:.0f}% of the circular-aperture trend is surface brightness, "
          f"{100*(1-frac):.0f}% is inclination")

    axes[0].text(0.97, 0.05,
                 "blends two effects:\n"
                 rf"${100*frac:.0f}\%$ surface brightness, ${100*(1-frac):.0f}\%$ inclination",
                 transform=axes[0].transAxes, fontsize=9.5, color="#8a3ffc",
                 va="bottom", ha="right")
    axes[1].text(0.97, 0.05,
                 "surface brightness alone;\n"
                 rf"inclination enters separately at ${b[2]:+.3f}\pm{se_[2]:.3f}$",
                 transform=axes[1].transAxes, fontsize=9.5, color="#8a3ffc",
                 va="bottom", ha="right")
    head = ("TF residual" if old else "Distance-modulus residual")
    fig.suptitle(f"{head} vs r-band surface brightness, v5b ANALYSIS "
                 f"($N$ = {g.sum():,}); the panels differ only by the "
                 r"$2.5\log_{10}(b/a)$ inclination term",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
