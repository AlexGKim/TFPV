#!/usr/bin/env python
"""Size dependence of the TF residual: is it angular size, physical size, or surface brightness?

The residual eta = log10(d_z/d_TF) correlates with the HyperLeda diameter at 6 sigma, and a
joint fit shows the driver is log D25 itself rather than its measurement error, so this is a
second parameter of the TFR and not an Eddington selection artifact.

The four panels separate variables that are easy to confuse:

  1. angular D25        -- what the sample is selected on, but distance-dependent
  2. physical D25 (kpc) -- angular x d_A(z), so it inherits the redshift distance
  3. B-band surface brightness  SB_D25_LEDA = bt + 5log10(D25) + const
  4. r-band surface brightness  m_r + 5log10(D26)

Only 3 and 4 are DISTANCE-INDEPENDENT and can therefore be used to correct the TFR without
feeding the distance back into its own predictor.  Angular size must never be used that way.

Redshift is controlled throughout by subtracting the inverse-variance mean residual in each
redshift bin, since all four quantities correlate with distance through the selection.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARC = 206264.806
C_PT = "#1f6fb2"
C_FIT = "#d1670b"


def wmean(y, w):
    return float(np.sum(w * y) / np.sum(w))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",
                   default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    p.add_argument("--label", default="", help="fit name shown in the figure title")
    p.add_argument("--rows", choices=["analysis", "main", "common"], default="analysis",
                   help="'common' = ANALYSIS in BOTH fits, so the two versions use the "
                        "same galaxies and only the fit differs")
    p.add_argument("--other",
                   default="output/DR2_TF_v5_3_2color_combined/DESI-DR2_TF_pv_cat_v5_3.fits",
                   help="the other fit, used to build the common-row mask")
    p.add_argument("--out", default="output/resid_vs_size.png")
    args = p.parse_args()

    b = fits.open(args.catalog)[1].data
    if args.rows == "main":
        sel = np.ones(len(b), bool)
    elif args.rows == "common":
        o = fits.open(args.other)[1].data
        if not np.array_equal(np.asarray(o["SGA_ID"]), np.asarray(b["SGA_ID"])):
            raise SystemExit("catalogs are not row-aligned; cannot build a common mask")
        sel = b["ANALYSIS"].astype(bool) & o["ANALYSIS"].astype(bool)
    else:
        sel = b["ANALYSIS"].astype(bool)
    d = b[sel]
    print(f"catalog {args.catalog}\n  rows mode '{args.rows}': {sel.sum()} galaxies")

    z = np.asarray(d["Z_DESI_CMB"], float)
    mu = np.asarray(d["MU_ZCMB"], float)
    d25 = np.asarray(d["D25_LEDA"], float) * 60.0          # arcsec
    d26 = np.asarray(d["D26"], float) * 60.0               # arcsec
    sb_b = np.asarray(d["SB_D25_LEDA"], float)
    m_r = np.asarray(d["R_MAG_SB26_CORR"], float)
    ld = np.asarray(d["LOGDIST"], float)
    er = np.asarray(d["LOGDIST_ERR"], float)

    dL = 10.0 ** (mu / 5.0 - 5.0)
    dA = dL / (1.0 + z) ** 2
    d25kpc = d25 / ARC * dA * 1e3
    sb_r = m_r + 5.0 * np.log10(d26)

    g = (np.isfinite(z) & np.isfinite(ld) & np.isfinite(er) & (er > 0) & (z > 1e-3)
         & np.isfinite(d25) & np.isfinite(sb_b) & np.isfinite(sb_r) & np.isfinite(d25kpc))
    z, ld, er, d25, d25kpc, sb_b, sb_r = [a[g] for a in (z, ld, er, d25, d25kpc, sb_b, sb_r)]
    w = 1.0 / er ** 2
    print(f"ANALYSIS rows used: {g.sum()}")

    # remove the redshift trend
    zb = np.linspace(z.min(), 0.065, 12)
    zi = np.clip(np.digitize(z, zb) - 1, 0, len(zb) - 2)
    y = ld.copy()
    for k in range(len(zb) - 1):
        s = zi == k
        if s.sum() > 20:
            y[s] -= wmean(ld[s], w[s])

    panels = [
        (np.log10(d25), "$\\log_{10}(D_{25}\\,/\\,{\\rm arcsec})$",
         "1. angular size\ndistance-dependent - NOT usable as a TFR term", "dex"),
        (np.log10(d25kpc), "$\\log_{10}(D_{25}\\,/\\,{\\rm kpc})$",
         "2. physical size\ninherits the redshift distance", "dex"),
        (sb_b, "$\\mu_B$ = SB_D25_LEDA  [mag arcsec$^{-2}$]",
         "3. B-band surface brightness\ndistance-independent", "mag"),
        (sb_r, "$m_r + 5\\log_{10} D_{26}$  [mag]",
         "4. r-band surface brightness\ndistance-independent, same band as the TFR", "mag"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.4))
    for ax, (x, xlab, title, unit) in zip(axes, panels):
        A = np.column_stack([np.ones_like(x), x - x.mean()])
        C = np.linalg.inv(A.T @ (A * w[:, None]))
        bb = C @ (A.T @ (w * y))
        se = np.sqrt(np.diag(C))
        sig = bb[1] / se[1]
        print(f"  {title.splitlines()[0]:24s} slope {bb[1]:+.5f} +/- {se[1]:.5f} "
              f"({sig:+5.1f} sigma) per {unit}")

        qs = np.percentile(x, np.linspace(2, 98, 11))
        cen, m, s = [], [], []
        for i in range(len(qs) - 1):
            k = (x >= qs[i]) & (x < qs[i + 1])
            if k.sum() < 30:
                continue
            cen.append(np.median(x[k]))
            m.append(wmean(y[k], w[k]))
            s.append(float(np.sqrt(1.0 / np.sum(w[k]))))
        cen, m, s = map(np.asarray, (cen, m, s))

        ax.axhline(0.0, color="#9a9a96", lw=1.2, zorder=1)
        xs = np.linspace(qs[0], qs[-1], 50)
        ax.plot(xs, bb[0] + bb[1] * (xs - x.mean()), color=C_FIT, lw=1.8, zorder=2)
        ax.errorbar(cen, m, yerr=s, fmt="o", color=C_PT, ms=5, lw=0, elinewidth=1.5,
                    capsize=2.5, zorder=3)
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.text(0.03, 0.04, f"slope {bb[1]:+.4f}$\\pm${se[1]:.4f}\n(${sig:+.1f}\\sigma$)",
                transform=ax.transAxes, fontsize=8.5, color="#6b6b67")
        ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c8c8c4")
    axes[0].set_ylabel("$\\langle\\eta\\rangle$ (redshift-controlled)")

    # joint fit: does r-band SB explain the angular-size trend?
    print("\njoint fit, z-controlled residual on log10(D25 ang) and r-band SB:")
    X = np.column_stack([np.ones_like(y), np.log10(d25) - np.log10(d25).mean(),
                         sb_r - sb_r.mean()])
    C = np.linalg.inv(X.T @ (X * w[:, None]))
    bb = C @ (X.T @ (w * y))
    se = np.sqrt(np.diag(C))
    for nm, i in (("log D25 (ang)", 1), ("r-band SB", 2)):
        print(f"   {nm:16s} {bb[i]:+.5f} +/- {se[i]:.5f}  ({bb[i]/se[i]:+.1f} sigma)")

    tag = f"{args.label}   |   " if args.label else ""
    fig.suptitle(f"{tag}Size dependence of the TF residual  "
                 "$\\eta=\\log_{10}(d_z/d_{\\rm TF})$   "
                 f"— positive $\\eta$ means brighter than the TFR predicts   "
                 f"[{args.rows} rows, N={len(d)}]",
                 fontsize=11, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
