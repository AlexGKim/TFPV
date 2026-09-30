#!/usr/bin/env python
"""Change in TF-predicted absolute magnitude between two combined catalogs, vs redshift.

Default comparison is v5_3 (z_obs_min = 0.03) against v5b (z_obs_min = 0.01).
Both fit the same galaxies, so rows match on SGA_ID and difference directly.

The predicted absolute magnitude is recomputed as R_MAG_SB26_CORR - MU_TF rather
than read from R_ABSMAG_SB26_TF: that column is carried through from the input
catalog untouched by color_predict.py and disagrees with the model prediction by
up to ~1.9 mag.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POP_COLOR = {"spiral": "#1f6fb2", "irregular": "#d1670b"}


def predicted_absmag(data):
    return data["R_MAG_SB26_CORR"] - data["MU_TF"]


def binned_trend(z, d, edges):
    """Median and 16/84 percentiles of d in bins of z; empty bins give NaN."""
    idx = np.digitize(z, edges) - 1
    mid, lo, hi, cen = [], [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 20:
            continue
        mid.append(np.median(d[sel]))
        lo.append(np.percentile(d[sel], 16))
        hi.append(np.percentile(d[sel], 84))
        cen.append(0.5 * (edges[b] + edges[b + 1]))
    return map(np.asarray, (cen, mid, lo, hi))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    p.add_argument("--new", default="output/DR2_TF_v5_3_2color_combined/DESI-DR2_TF_pv_cat_v5_3.fits")
    p.add_argument("--ref-label", default="v5b  ($z_{\\rm obs,min}=0.01$)")
    p.add_argument("--new-label", default="v5_3  ($z_{\\rm obs,min}=0.03$)")
    p.add_argument("--zcut", type=float, default=0.03, help="new z_obs_min, marked on the plot")
    p.add_argument("--out", default="output/DR2_TF_v5_3_2color_combined/delta_predmag_vs_z.png")
    args = p.parse_args()

    ref = fits.open(args.ref)[1].data
    new = fits.open(args.new)[1].data
    if not np.array_equal(ref["SGA_ID"], new["SGA_ID"]):
        raise SystemExit("catalogs do not share row order; SGA_ID mismatch")

    z = new["Z_DESI"]
    delta = predicted_absmag(new) - predicted_absmag(ref)
    good = np.isfinite(delta) & np.isfinite(z)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.axhline(0.0, color="#9a9a96", lw=1.0, zorder=1)
    ax.axvline(args.zcut, color="#9a9a96", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(
        args.zcut, 0.985, f"  new cut $z={args.zcut:g}$",
        transform=ax.get_xaxis_transform(), va="top", ha="left",
        fontsize=8.5, color="#6b6b67",
    )

    edges = np.linspace(z[good].min(), z[good].max(), 26)
    for pop in ("spiral", "irregular"):
        sel = good & (new["POPULATION"] == pop)
        c = POP_COLOR[pop]
        ax.scatter(z[sel], delta[sel], s=2.5, c=c, alpha=0.10, lw=0, zorder=2, rasterized=True)
        cen, mid, lo, hi = binned_trend(z[sel], delta[sel], edges)
        ax.fill_between(cen, lo, hi, color=c, alpha=0.18, lw=0, zorder=3)
        ax.plot(cen, mid, color=c, lw=2.0, zorder=4,
                solid_capstyle="round", path_effects=None)
        ax.plot(cen, mid, color="white", lw=4.0, zorder=3.5, solid_capstyle="round")
        ax.plot(cen, mid, color=c, lw=2.0, zorder=4, solid_capstyle="round",
                label=f"{pop}  (N={sel.sum():,})")
        ax.annotate(
            pop, xy=(cen[-1], mid[-1]), xytext=(4, 0), textcoords="offset points",
            color=c, fontsize=9.5, va="center", ha="left", fontweight="medium",
        )

    ax.set_xlabel("redshift  $z_{\\rm DESI}$")
    ax.set_ylabel("$\\Delta M_r^{\\rm TF}$  (new $-$ reference)  [mag]")
    ax.set_title(
        f"Change in TF-predicted absolute magnitude\n{args.new_label}  $-$  {args.ref_label}",
        fontsize=11, loc="left",
    )
    ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c8c8c4")
    ax.set_xlim(0, edges[-1] * 1.06)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    d = delta[good]
    ax.text(
        0.015, 0.03,
        f"all: mean {d.mean():+.4f}, sd {d.std():.4f} mag  (N={good.sum():,})",
        transform=ax.transAxes, fontsize=8.5, color="#6b6b67",
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    for pop in ("spiral", "irregular"):
        sel = good & (new["POPULATION"] == pop)
        print(f"  {pop:10s} N={sel.sum():6d}  mean {delta[sel].mean():+.4f}  sd {delta[sel].std():.4f}")


if __name__ == "__main__":
    main()
