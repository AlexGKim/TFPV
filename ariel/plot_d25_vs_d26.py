#!/usr/bin/env python
"""LEDA B-band angular size (D25) against SGA r-band angular size (D26).

D25 is the HyperLeda B-band isophotal diameter the sample is selected on (> 20"); D26
is the SGA r-band diameter, measured far more precisely (0.0034 dex vs 0.080 dex). The
two correlate at 0.87 in the log, which is why an r-band diameter cut can supersede the
B-band one -- and the scatter about that relation is what sets how stringent the r-band
cut has to be.
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
    p.add_argument("--theta-r", type=float, default=35.0)
    p.add_argument("--out", default="output/d25_vs_d26.png")
    args = p.parse_args()

    d = fits.open(args.fits)[1].data
    d25 = np.asarray(d["D25_LEDA"], float) * 60.0
    d26 = np.asarray(d["D26"], float) * 60.0
    g = np.isfinite(d25) & np.isfinite(d26) & (d25 > 0) & (d26 > 0)
    d25, d26 = d25[g], d26[g]
    r = np.corrcoef(np.log10(d26), np.log10(d25))[0, 1]
    print(f"N = {g.sum()}   corr(log D26, log D25) = {r:+.3f}")
    print(f"ratio D26/D25: median {np.median(d26/d25):.3f}  "
          f"sd(log) {np.std(np.log10(d26/d25)):.4f}")

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    ax.scatter(d26, d25, s=2.0, c="#1f6fb2", alpha=0.08, lw=0, rasterized=True, zorder=2)

    lim = (18, 320)
    xx = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 50)
    ax.plot(xx, xx, color="#555555", lw=1.3, ls=(0, (5, 3)), zorder=4, label="1:1")
    med = np.median(d26 / d25)
    ax.plot(xx, xx / med, color="#d1670b", lw=2.0, zorder=5,
            label=f"median ratio $D_{{26}}/D_{{25}}$ = {med:.2f}")

    ax.axhline(20.0, color="#9a9a96", lw=1.6, zorder=3)
    ax.text(lim[1] * 0.95, 20.6, "LEDA selection:  $D_{25} > 20''$",
            ha="right", va="bottom", fontsize=9, color="#6b6b67")
    ax.axvline(args.theta_r, color="#8a3ffc", lw=1.4, ls=(0, (3, 2)), zorder=3)
    ax.text(args.theta_r * 1.05, lim[1] * 0.80,
            f"candidate r-band cut\n$\\theta_r = {args.theta_r:g}''$",
            fontsize=9, color="#8a3ffc", rotation=90, va="top")

    # binned median of D25 in bins of D26
    edges = np.logspace(np.log10(20), np.log10(300), 20)
    cen, mu = [], []
    for i in range(len(edges) - 1):
        m = (d26 >= edges[i]) & (d26 < edges[i + 1])
        if m.sum() < 25:
            continue
        cen.append(np.sqrt(edges[i] * edges[i + 1]))
        mu.append(np.median(d25[m]))
    ax.plot(cen, mu, color="white", lw=4.0, zorder=5.5)
    ax.plot(cen, mu, color="#1f6fb2", lw=2.2, zorder=6, label="binned median")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SGA r-band angular size  $D_{26}$  [arcsec]")
    ax.set_ylabel("LEDA B-band angular size  $D_{25}$  [arcsec]")
    ax.set_title(f"B-band vs r-band angular size   (N = {g.sum():,},  "
                 f"corr in log = {r:+.3f})\n"
                 "the pile-up along $D_{25}=20''$ is the selection boundary",
                 fontsize=10.5, loc="left")
    ax.grid(True, which="both", color="#e6e6e3", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color("#c8c8c4")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
