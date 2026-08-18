#!/usr/bin/env python3
"""Plot D26 vs redshift, raw and corrected for cosmological surface-brightness dimming.

Why a correction is needed
--------------------------
D26 is an *isophotal* diameter: the size at which surface brightness reaches
26 mag/arcsec^2. Surface brightness dims as (1+z)^-4 with redshift, so at higher z
the observed 26 mag/arcsec^2 isophote falls at a smaller radius than the rest-frame
one. The measured D26 therefore shrinks with redshift for reasons that have nothing
to do with the galaxies, biasing any D26-vs-z trend.

The correction
--------------
Taken from the DESI PV selection-function code, which treats the isophotal size as
shrinking by the factor (1 + 4 ln(1+z)) -- see e.g.
DESI_SGA/TF/Y1/systematic_tests/TF_Y1_alternate_v10_sigma2D-SB_corr.ipynb and the
Ariel mock generators, where the limiting redshift solves

    (1 + 4 ln(1+x)) * D_A(x)  =  D_A(z_obs) * theta_obs * (1 + 4 ln(1+z_obs)) / theta_min

The quantity held fixed across redshift there is D_A(z) * theta_obs * (1+4 ln(1+z)).
Since the uncorrected physical diameter is D_A(z) * theta_obs, the rest-frame
(dimming-corrected) diameter is simply

    D26_corrected = D26_kpc * (1 + 4 ln(1+z))

The factor exceeds 1 for z > 0, i.e. the correction makes galaxies larger, which is
the right sign: dimming causes the observed isophote to underestimate the
rest-frame one. 4 ln(1+z) ~ 4z at low z, so it is a ~20% effect by z = 0.05.

Which definition of "D26"?  (this determines the sign)
------------------------------------------------------
The sign above holds because D26 is ISOPHOTAL -- the diameter at which surface
brightness reaches 26 mag/arcsec^2. Under the other plausible reading, an aperture
enclosing 26 mag of total flux, dimming would require a LARGER aperture and the
correction would divide rather than multiply. Two checks on this catalog settle it
as isophotal:

  * R_MAG_SB26 has median 16.07, not ~26 -- it is the total magnitude enclosed
    WITHIN the SB26 isophote, so "26" is not a flux threshold.
  * SMA_SB22..SMA_SB26 rise monotonically (6.36, 10.31, 13.77, 17.04, 20.88
    arcsec): isophotal radii at successively fainter SB levels.
  * D26 * 60 == 2 * SMA_SB26 exactly.

The third panel plots the opposite convention anyway, for comparison.

What actually drives the D26-vs-z trend (it is NOT dimming)
-----------------------------------------------------------
Median D26_kpc rises 1.82x across 0.002 < z < 0.066 (17.9 -> 32.6 kpc). The
correction's entire dynamic range over that span is only 1.245x, and it acts in the
same direction, steepening the trend to 2.18x rather than flattening it. The cause
is the diameter-limited selection: theta_obs floors at 20.04 arcsec in the highest
z bins versus 55.7 arcsec in the lowest, so at higher z only intrinsically larger
galaxies enter the sample. That edge is drawn in red. For an unbiased size
distribution, weight by 1/V_max using the catalog's DIST_MAX / MAX_VOL_FRAC
columns -- cf. DESI_SGA/TF/Y1/Y1_logdist_correct_v2.ipynb.

Usage:
    python3 plot_d26_vs_z.py [FITS] [-o OUT.png] [--zcol Z_DESI_CMB] [--main-only]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

DEFAULT_FITS = "output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits"
COLORS = {"spiral": "#3366cc", "irregular": "#dd6633"}


def sb_dimming_factor(z):
    """(1 + 4 ln(1+z)) -- the isophotal-size correction for (1+z)^-4 SB dimming."""
    return 1.0 + 4.0 * np.log1p(z)


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
    ap.add_argument("--zcol", default="Z_DESI_CMB",
                    help="redshift for the correction (default Z_DESI_CMB: dimming "
                         "depends on cosmological redshift, not heliocentric)")
    ap.add_argument("--main-only", action="store_true")
    args = ap.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.fits)),
                                   "d26_kpc_corrected_vs_redshift.png")
    out = os.path.abspath(out)

    with fits.open(args.fits) as h:
        data = h[1].data
        names = set(data.dtype.names or ())
        for col in ("D26_kpc", args.zcol):
            if col not in names:
                raise SystemExit(f"ERROR: column {col!r} not in {args.fits}")
        d26 = np.asarray(data["D26_kpc"], float)
        d26_arcmin = (np.asarray(data["D26"], float) if "D26" in names
                      else np.full(len(d26), np.nan))
        names_all = names
        z = np.asarray(data[args.zcol], float)
        pop = (np.asarray(data["POPULATION"]).astype(str)
               if "POPULATION" in names else np.full(len(d26), "all"))
        main = (np.asarray(data["MAIN"], bool)
                if "MAIN" in names else np.ones(len(d26), bool))

    good = np.isfinite(d26) & np.isfinite(z) & (d26 > 0) & (z > 0)
    if args.main_only:
        good &= main

    d26c = d26 * sb_dimming_factor(z)

    print(f"file            : {args.fits}")
    print(f"rows            : {len(d26)}")
    print(f"plotted         : {int(good.sum())}  (excluded {int((~good).sum())})")
    print(f"redshift column : {args.zcol}, range "
          f"{z[good].min():.5f} to {z[good].max():.5f}")
    print(f"correction      : (1 + 4 ln(1+z)) = "
          f"{sb_dimming_factor(z[good]).min():.4f} to "
          f"{sb_dimming_factor(z[good]).max():.4f}")
    print(f"D26 raw         : median {np.median(d26[good]):7.3f} kpc")
    print(f"D26 corrected   : median {np.median(d26c[good]):7.3f} kpc")

    # Quantify the bias the correction is meant to remove: fit log D26 vs log(1+z)
    # in each case. A flat slope means no residual redshift dependence.
    lz = np.log1p(z[good])
    for lbl, y in (("raw", d26[good]), ("x factor", d26c[good]),
                   ("/ factor", (d26 / sb_dimming_factor(z))[good])):
        s, b = np.polyfit(lz, np.log(y), 1)
        print(f"  d ln D26 / d ln(1+z)  {lbl:9s} = {s:+.3f}")

    # The diameter-limited selection boundary. The sample is cut at theta >= ~20",
    # so at each z only galaxies above D26_min(z) can enter. D26_kpc/theta is
    # proportional to the angular diameter distance, so estimate it empirically in
    # fine z bins rather than assuming the catalog's cosmology and normalization
    # (which we could not reproduce exactly: D26_kpc / (D_A*theta) ~ 0.73 for
    # Planck18, a constant offset that cancels out of this ratio).
    theta_arcsec = None
    if "D26" in names_all:
        theta_arcsec = d26_arcmin * 60.0
        theta_min = np.nanmin(theta_arcsec[good])
        zb = np.linspace(z[good].min(), z[good].max(), 40)
        bcen, bmin = [], []
        for a, b in zip(zb[:-1], zb[1:]):
            m = good & (z >= a) & (z < b)
            if m.sum() >= 10:
                bcen.append(0.5 * (a + b))
                bmin.append(np.median(d26[m] / theta_arcsec[m]) * theta_min)
        bcen, bmin = np.array(bcen), np.array(bmin)
        print(f"selection edge  : theta_min = {theta_min:.2f} arcsec")
    else:
        bcen = bmin = np.array([])

    d26d = d26 / sb_dimming_factor(z)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), sharey=True,
                             constrained_layout=True)
    order = [p for p in ("spiral", "irregular") if p in set(pop)] or sorted(set(pop))
    panels = (
        (axes[0], d26,  "raw  $D_{26}$"),
        (axes[1], d26c, r"$\times\,(1+4\ln(1+z))$   rest-frame larger"),
        (axes[2], d26d, r"$\div\,(1+4\ln(1+z))$   rest-frame smaller"),
    )
    for ax, y, title in panels:
        for p in order:
            m = good & (pop == p)
            if m.any():
                ax.scatter(z[m], y[m], s=4, alpha=0.28, linewidths=0,
                           color=COLORS.get(p, "grey"),
                           label=f"{p} (n={int(m.sum()):,})")
        cen, med = median_trend(z, y, good)
        if len(cen):
            ax.plot(cen, med, color="black", lw=2.2, label="median in $z$ bins")
        # the raw median trend on both panels, so the change is visible
        cen0, med0 = median_trend(z, d26, good)
        if len(cen0):
            ax.plot(cen0, med0, color="black", lw=1.2, ls="--", alpha=0.55,
                    label="raw median (reference)")
        if len(bcen):
            if y is d26c:
                edge = bmin * sb_dimming_factor(bcen)
            elif y is d26d:
                edge = bmin / sb_dimming_factor(bcen)
            else:
                edge = bmin
            ax.plot(bcen, edge, color="#cc2222", lw=2.0,
                    label=rf"selection edge $\theta={theta_min:.0f}''$")
        ax.set_yscale("log")
        ax.set_xlabel(rf"$z$   ({args.zcol})")
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel(r"$D_{26}$   [kpc]")
    axes[2].legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.suptitle(os.path.basename(args.fits)
                 + ("   (MAIN only)" if args.main_only else ""))
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
