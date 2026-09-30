#!/usr/bin/env python
"""TF residual against how many sigma a galaxy sits above the D25 = 20" selection cut.

The SGA-2020 parent is selected on the HyperLeda B-band isophotal diameter D_L(25) > 20".
HyperLeda quotes an uncertainty e_logD25 on that diameter, which SGA-2020 does not carry
through, so it is joined here by PGC from VizieR VII/237 (Paturel+ 2003).

    n_sigma = (log10(D25 / 0.1') - log10(20" / 0.1')) / e_logD25

A galaxy at small n_sigma may be in the sample only because its diameter scattered up
across the threshold.  Since diameter correlates with luminosity (M_r = -13.58 - 5.19
log10 D25), such a galaxy is intrinsically fainter than its measured size implies, so an
Eddington-type selection bias would show up as a trend of TF residual with n_sigma.

Redshift is a confound -- angular diameter shrinks with distance, so low n_sigma is
preferentially high z, which carries its own residual trend.  The z-controlled series
subtracts the weighted mean residual in each redshift bin before binning in n_sigma.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG20 = np.log10(20.0 / 6.0)      # 20 arcsec expressed in log10(0.1 arcmin)
C_RAW = "#1f6fb2"
C_CTL = "#d1670b"


def load_hyperleda(path):
    pgc, eld = [], []
    for line in open(path):
        if line.startswith(("#", "-")) or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 3 or not f[0].strip().lstrip("-").isdigit():
            continue
        pgc.append(int(f[0]))
        eld.append(float(f[2]) if f[2].strip() else np.nan)
    return {p: e for p, e in zip(pgc, eld)}


def wmean(y, w):
    return float(np.sum(w * y) / np.sum(w))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",
                   default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    p.add_argument("--hyperleda", default="data/external/hyperleda_VII237_logd25.tsv")
    p.add_argument("--out", default="output/resid_vs_d25_sigma.png")
    args = p.parse_args()

    b = fits.open(args.catalog)[1].data
    d = b[b["ANALYSIS"].astype(bool)]
    lut = load_hyperleda(args.hyperleda)
    e = np.array([lut.get(int(p), np.nan) for p in d["PGC"]])

    lg = np.log10(np.asarray(d["D25_LEDA"], float) * 10.0)     # arcmin -> 0.1 arcmin
    ns = (lg - LOG20) / e
    ld = np.asarray(d["LOGDIST"], float)
    er = np.asarray(d["LOGDIST_ERR"], float)
    z = np.asarray(d["Z_DESI_CMB"], float)

    g = np.isfinite(ns) & np.isfinite(ld) & np.isfinite(er) & (er > 0) & np.isfinite(z)
    ns, ld, er, z = ns[g], ld[g], er[g], z[g]
    w = 1.0 / er ** 2
    print(f"ANALYSIS rows {len(d)}; with HyperLeda e_logD25 {g.sum()} ({100*g.mean():.1f}%)")
    print(f"n_sigma: p10 {np.percentile(ns,10):.2f}  median {np.median(ns):.2f} "
          f" p90 {np.percentile(ns,90):.2f}")

    # remove the known redshift trend
    zb = np.linspace(z.min(), 0.065, 12)
    zi = np.clip(np.digitize(z, zb) - 1, 0, len(zb) - 2)
    ldc = ld.copy()
    for k in range(len(zb) - 1):
        s = zi == k
        if s.sum() > 20:
            ldc[s] -= wmean(ld[s], w[s])

    edges = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 40.0])
    cen, m_raw, m_ctl, se, nb = [], [], [], [], []
    for i in range(len(edges) - 1):
        s = (ns >= edges[i]) & (ns < edges[i + 1])
        if s.sum() < 30:
            continue
        cen.append(np.median(ns[s]))
        m_raw.append(wmean(ld[s], w[s]))
        m_ctl.append(wmean(ldc[s], w[s]))
        se.append(float(np.sqrt(1.0 / np.sum(w[s]))))
        nb.append(int(s.sum()))
    cen, m_raw, m_ctl, se = map(np.asarray, (cen, m_raw, m_ctl, se))

    print(f"\n{'n_sig':>7} {'N':>6} {'<eta> raw':>11} {'z-controlled':>13} {'err':>8}")
    for i in range(len(cen)):
        print(f"{cen[i]:7.2f} {nb[i]:6d} {m_raw[i]:+11.5f} {m_ctl[i]:+13.5f} {se[i]:8.5f}")

    for nm, y in (("raw", ld), ("z-controlled", ldc)):
        x = np.log10(np.clip(ns, 0.05, None))
        A = np.column_stack([np.ones_like(x), x - x.mean()])
        C = np.linalg.inv(A.T @ (A * w[:, None]))
        bb = C @ (A.T @ (w * y))
        print(f"  slope d<eta>/dlog10(n_sigma) [{nm:12s}] = {bb[1]:+.5f} +/- "
              f"{np.sqrt(C[1,1]):.5f}  ({bb[1]/np.sqrt(C[1,1]):+.1f} sigma)")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.axhline(0.0, color="#9a9a96", lw=1.2, zorder=1)
    ax.errorbar(cen, m_raw, yerr=se, fmt="o", color=C_RAW, ms=5.5, lw=0,
                elinewidth=1.6, capsize=2.5, zorder=3, label="raw")
    ax.plot(cen, m_raw, color=C_RAW, lw=1.5, alpha=0.8, zorder=2)
    ax.errorbar(cen, m_ctl, yerr=se, fmt="s", color=C_CTL, ms=5.5, lw=0,
                elinewidth=1.6, capsize=2.5, zorder=3, label="redshift-controlled")
    ax.plot(cen, m_ctl, color=C_CTL, lw=1.5, ls="--", alpha=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("$n_\\sigma$ above the $D_{25}=20''$ cut   "
                  "$=(\\log D_{25}-\\log 20'')\\,/\\,e_{\\log D25}$")
    ax.set_ylabel("$\\langle \\eta \\rangle = \\langle\\log_{10}(d_z/d_{\\rm TF})\\rangle$")
    ax.set_title("TF residual vs distance above the HyperLeda selection threshold\n"
                 "an Eddington-type selection bias would trend at small $n_\\sigma$",
                 fontsize=10.5, loc="left")
    ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c8c8c4")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
