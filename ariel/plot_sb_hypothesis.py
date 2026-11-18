#!/usr/bin/env python
"""Does the Tully-Fisher relation need a surface-brightness term?

Left:  TF distance residual vs r-band surface brightness, SB_r = m_r + 5log10(theta26),
       for v5b (x-only predictions).  A non-zero slope is the missing second parameter.
       v6 is deliberately not shown: its residuals are suppressed 126-fold by the
       distance leak, so its slope is an artefact of that cancellation, not a measurement.
Middle: SB_r against redshift as a 2-D histogram.  An unmodelled second parameter only
       biases distances if it drifts with redshift -- which is what makes this panel the
       relevant one.
Right: the same for the HyperLeda B-band surface brightness inside D25.  That is the
       band the sample is actually selected in (D_L(25) > 20"), so its redshift drift is
       the one tied directly to the selection rather than to the TF band.
"""
import argparse

import numpy as np
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_V5 = "#1f6fb2"
C_V6 = "#d1670b"


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


def wls(x, y, w):
    A = np.column_stack([np.ones_like(x), x - x.mean()])
    C = np.linalg.inv(A.T @ (A * w[:, None]))
    b = C @ (A.T @ (w * y))
    return b[1], np.sqrt(C[1, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v5", default="output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
    p.add_argument("--v6", default="output/DR2_TF_v6_2color_combined/DESI-DR2_TF_pv_cat_v6.fits")
    p.add_argument("--out", default="output/sb_hypothesis.png")
    args = p.parse_args()

    d5 = fits.open(args.v5)[1].data
    d6 = fits.open(args.v6)[1].data
    a = d5["ANALYSIS"].astype(bool)
    mr = np.asarray(d5["R_MAG_SB26_CORR"], float)
    sb = mr + 5.0 * np.log10(np.asarray(d5["D26"], float) * 60.0)
    z = np.asarray(d5["Z_DESI_CMB"], float)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.2))

    # ---- left: residual vs SB --------------------------------------------
    ax = axes[0]
    ax.axhline(0.0, color="#9a9a96", lw=1.1, zorder=1)
    edges = np.percentile(sb[a & np.isfinite(sb)], np.linspace(1, 99, 13))
    for d, c, lab in ((d5, C_V5, "v5b  (x-only)"),):
        eta = np.asarray(d["LOGDIST"], float)
        er = np.asarray(d["LOGDIST_ERR"], float)
        g = a & np.isfinite(eta) & np.isfinite(er) & (er > 0) & np.isfinite(sb) & np.isfinite(z)
        w = 1.0 / er[g] ** 2
        # remove the redshift trend so this is the SB dependence alone
        y = eta[g].copy()
        zb = np.linspace(z[g].min(), 0.065, 12)
        for i in range(len(zb) - 1):
            m = (z[g] >= zb[i]) & (z[g] < zb[i + 1])
            if m.sum() > 20:
                y[m] -= np.sum(w[m] * y[m]) / np.sum(w[m])
        sl, se_ = wls(sb[g], y, w)
        cen, mu, se = binned(sb[g], y, w, edges)
        ax.errorbar(cen, mu, yerr=se, fmt="o", color=c, ms=5, lw=0, elinewidth=1.5,
                    capsize=2.5, zorder=3,
                    label=f"{lab}:  {sl:+.4f}$\\pm${se_:.4f} dex/mag  (${sl/se_:+.1f}\\sigma$)")
        xs = np.linspace(edges[0], edges[-1], 40)
        ax.plot(xs, sl * (xs - sb[g].mean()), color=c, lw=1.8, zorder=2)
        print(f"{lab:28s} slope {sl:+.5f} +/- {se_:.5f}  ({sl/se_:+.1f} sigma)")
    ax.set_xlabel("$SB_r = m_r + 5\\log_{10}\\theta_{26}$  [mag]")
    ax.set_ylabel("$\\langle\\eta\\rangle$, redshift-controlled  [dex]")
    ax.set_title("TF residual vs surface brightness\n"
                 "a non-zero slope is the missing second parameter",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # ---- density panels --------------------------------------------------
    def density_panel(ax, val, g, ylab, title_what, note):
        zlo, zhi = 0.0, 0.065
        slo, shi = np.percentile(val[g], 0.5), np.percentile(val[g], 99.5)
        zedge = np.linspace(zlo, zhi, 41)
        sedge = np.linspace(slo, shi, 33)
        H, _, _ = np.histogram2d(z[g], val[g], bins=[zedge, sedge])
        H = np.ma.masked_where(H == 0, H)
        pcm = ax.pcolormesh(zedge, sedge, H.T, cmap="Blues",
                            norm=matplotlib.colors.LogNorm(vmin=1, vmax=H.max()),
                            zorder=2, edgecolors="none")
        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label("galaxies per cell", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ze = np.linspace(z[g].min(), 0.065, 18)
        cen, mu, _ = binned(z[g], val[g], np.ones(int(g.sum())), ze)
        ax.plot(cen, mu, color="white", lw=4.5, zorder=3)
        ax.plot(cen, mu, color="#d1670b", lw=2.4, zorder=4, label="median")
        r = np.corrcoef(z[g], val[g])[0, 1]
        ax.set_xlabel("redshift  $z_{\\rm CMB}$")
        ax.set_ylabel(ylab)
        ax.set_ylim(slo, shi)
        ax.set_title(f"{title_what}\ncorr = {r:+.3f}: {note}", fontsize=10.5, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor="#333333")
        print(f"  corr({ylab.split('$')[1] if '$' in ylab else ylab}, z) = {r:+.4f}"
              f"   median {mu[0]:.3f} -> {mu[-1]:.3f}  (drift {mu[-1]-mu[0]:+.3f} mag)")
        return r

    g = a & np.isfinite(sb) & np.isfinite(z)
    density_panel(axes[1], sb, g, "$SB_r$  [mag]",
                  "r-band surface brightness vs redshift",
                  "the TF-band second parameter drifts with distance")

    sb_b = np.asarray(d5["SB_D25_LEDA"], float)
    gb = a & np.isfinite(sb_b) & np.isfinite(z)
    density_panel(axes[2], sb_b, gb, "$\\mu_B$ = SB_D25_LEDA  [mag arcsec$^{-2}$]",
                  "B-band surface brightness vs redshift",
                  "the band the sample is selected in")

    for ax in axes:
        ax.grid(True, color="#e6e6e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#c8c8c4")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
