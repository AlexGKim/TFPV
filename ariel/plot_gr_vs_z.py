#!/usr/bin/env python
"""Observed SGA g - r colour against redshift for the DR2 TF sample.

Uses the Legacy Survey isophotal magnitudes inside D26: G_MAG_SB26 - R_MAG_SB26,
OBSERVED, i.e. no k-correction and no Galactic or internal extinction correction.
Pass --corrected to use the _CORR columns instead.

This is the same g - r that carries a -34 sigma correlation with the TF residual in
the x-only predictions, so its behaviour with redshift matters: the sample is selected
on the B-band diameter D_L(25) > 20", so a colour trend with z mixes k-correction,
evolution, and selection. The internal-dust correction is inclination-dependent, so
observed and corrected colours can trend differently with z.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POP_COLOR = {"spiral": "#1f6fb2", "irregular": "#d1670b"}


def binned(x, y, edges, n_min=25):
    cen, med, lo, hi = [], [], [], []
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1])
        if s.sum() < n_min:
            continue
        cen.append(0.5 * (edges[i] + edges[i + 1]))
        med.append(np.median(y[s]))
        lo.append(np.percentile(y[s], 16))
        hi.append(np.percentile(y[s], 84))
    return map(np.asarray, (cen, med, lo, hi))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",
                   default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    p.add_argument("--corrected", action="store_true",
                   help="use G/R_MAG_SB26_CORR instead of the observed magnitudes")
    p.add_argument("--out", default="output/gr_color_vs_z.png")
    args = p.parse_args()

    suf = "_CORR" if args.corrected else ""
    d = fits.open(args.catalog)[1].data
    z = np.asarray(d["Z_DESI_CMB"], float)
    gr = np.asarray(d["G_MAG_SB26" + suf], float) - np.asarray(d["R_MAG_SB26" + suf], float)
    pop = np.asarray(d["POPULATION"])
    good = np.isfinite(z) & np.isfinite(gr) & (z > 1e-3)
    kind = "extinction-corrected" if args.corrected else "observed"
    print(f"{kind} g-r: N={good.sum()} median {np.median(gr[good]):.3f} sd {np.std(gr[good]):.3f}")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    edges = np.linspace(z[good].min(), z[good].max(), 22)

    for name in ("spiral", "irregular"):
        s = good & (pop == name)
        c = POP_COLOR[name]
        ax.scatter(z[s], gr[s], s=2.5, c=c, alpha=0.10, lw=0, rasterized=True, zorder=2)
        cen, med, lo, hi = binned(z[s], gr[s], edges)
        ax.fill_between(cen, lo, hi, color=c, alpha=0.18, lw=0, zorder=3)
        ax.plot(cen, med, color="white", lw=4.0, zorder=3.5, solid_capstyle="round")
        ax.plot(cen, med, color=c, lw=2.2, zorder=4, solid_capstyle="round",
                label=f"{name}  (N={int(s.sum()):,})")
        ax.annotate(name, xy=(cen[-1], med[-1]), xytext=(5, 0),
                    textcoords="offset points", color=c, fontsize=9.5, va="center")
        A = np.column_stack([np.ones(int(s.sum())), z[s] - z[s].mean()])
        co, *_ = np.linalg.lstsq(A, gr[s], rcond=None)
        print(f"  {name:10s} N={int(s.sum()):6d} median {np.median(gr[s]):.3f} "
              f"d(g-r)/dz = {co[1]:+.3f} per unit z "
              f"({co[1]*0.055:+.3f} mag across 0.01<z<0.065)")

    cen, med, _, _ = binned(z[good], gr[good], edges)
    ax.plot(cen, med, color="#555", lw=1.4, ls=(0, (5, 3)), zorder=5, label="all")
    A = np.column_stack([np.ones(int(good.sum())), z[good] - z[good].mean()])
    co, *_ = np.linalg.lstsq(A, gr[good], rcond=None)
    print(f"  {'all':10s} N={int(good.sum()):6d} median {np.median(gr[good]):.3f} "
          f"d(g-r)/dz = {co[1]:+.3f} per unit z ({co[1]*0.055:+.3f} mag across the range)")

    ax.set_xlabel("redshift  $z_{\\rm CMB}$")
    ax.set_ylabel(f"{kind} $g - r$  (SGA, inside $D_{{26}}$)  [mag]")
    ax.set_title(f"{kind.capitalize()} SGA $g-r$ colour vs redshift\n"
                 "sample selected on the B-band $D_{25}>20''$, so k-correction, "
                 "evolution and selection all enter",
                 fontsize=10.5, loc="left")
    ax.set_ylim(np.percentile(gr[good], 0.5), np.percentile(gr[good], 99.5))
    ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c8c8c4")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
