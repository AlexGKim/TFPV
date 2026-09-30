#!/usr/bin/env python
"""TF residuals (LOGDIST) against redshift for the two fits that differ only in z_obs_min.

If the TFR is universal and the truncation is correctly modelled, <logdist> must be
flat in z.  v5b (fitted z>0.01) is flat; v5_3 (fitted z>0.03) is flat only inside its
own fitted range and pulls away below it, which is extrapolation rather than evidence
of a redshift-dependent TFR.

Both catalogs cover the identical galaxies, so the two curves are paired point by point.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("v5b",  "output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits",
     0.01, "#1f6fb2", "o", "-"),
    ("v5_3", "output/DR2_TF_v5_3_2color_combined/DESI-DR2_TF_pv_cat_v5_3.fits",
     0.03, "#d1670b", "s", "--"),
]


def load(path):
    b = fits.open(path)[1].data
    z, ld, e = b["Z_DESI_CMB"], b["LOGDIST"], b["LOGDIST_ERR"]
    g = np.isfinite(z) & np.isfinite(ld) & np.isfinite(e) & (e > 0)
    return z[g], ld[g], e[g]


def binned(z, ld, e, edges, n_min=30):
    c, m, s = [], [], []
    for i in range(len(edges) - 1):
        k = (z >= edges[i]) & (z < edges[i + 1])
        if k.sum() < n_min:
            continue
        w = 1.0 / e[k] ** 2
        c.append(0.5 * (edges[i] + edges[i + 1]))
        m.append(np.sum(w * ld[k]) / np.sum(w))
        s.append(np.sqrt(1.0 / np.sum(w)))
    return map(np.asarray, (c, m, s))


def trend(z, ld, e):
    w = 1.0 / e ** 2
    A = np.column_stack([np.ones_like(z), z - z.mean()])
    C = np.linalg.inv(A.T @ (A * w[:, None]))
    b = C @ (A.T @ (w * ld))
    return b, np.sqrt(np.diag(C)), z.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--z-lo", type=float, default=0.01)
    p.add_argument("--z-hi", type=float, default=0.065)
    p.add_argument("--nbins", type=int, default=12)
    p.add_argument("--out", default="output/logdist_vs_z_v5_v5_3.png")
    args = p.parse_args()

    edges = np.linspace(args.z_lo, args.z_hi, args.nbins)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    # region where v5_3 has no training data
    ax.axvspan(args.z_lo, 0.03, color="#f0ede6", zorder=0)
    ax.text(0.0105, 0.975, "v5_3 extrapolates here\n(no training data below $z=0.03$)",
            transform=ax.get_xaxis_transform(), va="top", ha="left",
            fontsize=8.5, color="#8a8a84")
    ax.axhline(0.0, color="#9a9a96", lw=1.2, zorder=1)

    zgrid = np.linspace(args.z_lo, args.z_hi, 100)
    dodge = 0.18 * (edges[1] - edges[0])   # so the two sets of error bars stay legible
    for j, (lab, path, zmin, col, mk, ls) in enumerate(RUNS):
        z, ld, e = load(path)
        c, m, s = binned(z, ld, e, edges)
        c = c + (j - 0.5) * dodge
        b, be, zbar = trend(z, ld, e)
        sig = b[1] / be[1]
        ax.plot(zgrid, b[0] + b[1] * (zgrid - zbar), ls=ls, color=col, lw=1.4,
                alpha=0.85, zorder=3)
        ax.errorbar(c, m, yerr=s, fmt=mk, color=col, ms=5.5, lw=0, elinewidth=1.6,
                    capsize=2.5, zorder=4,
                    label=f"{lab}  ($z_{{\\rm obs,min}}={zmin:g}$):  "
                          f"$d\\langle\\eta\\rangle/dz = {b[1]:+.3f}\\pm{be[1]:.3f}$  "
                          f"(${sig:+.1f}\\sigma$)")
        print(f"{lab:5s} slope {b[1]:+.4f} +/- {be[1]:.4f}  ({sig:+.1f} sigma)")

    ax.set_xlabel("redshift  $z_{\\rm CMB}$")
    ax.set_ylabel("$\\langle \\eta \\rangle = \\langle \\log_{10}(d_z/d_{\\rm TF}) \\rangle$")
    ax.set_title("TF residuals vs redshift: identical galaxies, two fitting windows\n"
                 "flat $\\Rightarrow$ the TFR and its truncation are consistent across the range",
                 fontsize=10.5, loc="left")
    ax.set_xlim(args.z_lo - 0.001, args.z_hi + 0.001)
    ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c8c8c4")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
