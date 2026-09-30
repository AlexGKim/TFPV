#!/usr/bin/env python
"""Apply the r-band diameter cut that supersedes the LEDA B-band selection.

The survey selects on the HyperLeda B-band D25 > 20", which is soft in absolute
magnitude (the size-luminosity scatter is 0.63 mag, so a hard bound derived from it
excludes 10.6% of the observed galaxies).  Imposing our own, more stringent cut on the
precisely measured r-band diameter instead makes the selection exact:

    theta26 > theta_r   <=>   y < SB_r - mu(z) - 5 log10(theta_r)

and the right-hand side is an identity for y + 5 log10(theta26/theta_r), so no galaxy
in the cut sample can violate it.

Residual incompleteness -- galaxies with theta26 > theta_r whose catalogued D25 fell
below 20" and so are missing entirely -- is estimated from a truncated-Gaussian fit to
log10(D26/D25): about 20% at 35", 5.9% at 40", 1.4% at 45", 0.36% at 50".
"""
import argparse
import os

import numpy as np
from astropy.io import fits

DEFAULT_INPUTS = ["data/DR2_TF_spirals_v5_spiral.fits",
                  "data/DR2_TF_irrs_v5_irregular.fits"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    p.add_argument("--theta-r", type=float, default=35.0, help="r-band diameter cut, arcsec")
    p.add_argument("--suffix", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    suf = args.suffix or f"_r{int(round(args.theta_r))}"

    tot_in = tot_out = 0
    for path in args.input:
        stem, ext = os.path.splitext(path)
        out = f"{stem}{suf}{ext}"
        if os.path.exists(out) and not args.force:
            raise SystemExit(f"{out} exists; pass --force")
        with fits.open(path) as hl:
            d = hl[1].data
            th = np.asarray(d["D26"], float) * 60.0          # arcsec
            keep = np.isfinite(th) & (th > args.theta_r)
            hdu = fits.BinTableHDU(data=d[keep], header=hl[1].header.copy())
            hdu.header["RBANDCUT"] = (args.theta_r, "r-band D26 cut [arcsec]")
            hdu.writeto(out, overwrite=True)
        tot_in += len(th)
        tot_out += int(keep.sum())
        print(f"  {path:42s} {len(th):6d} -> {int(keep.sum()):6d} -> {out}")
    print(f"  {'TOTAL':42s} {tot_in:6d} -> {tot_out:6d}  "
          f"({100*tot_out/tot_in:.1f}% kept at theta_r = {args.theta_r:g}\")")


if __name__ == "__main__":
    main()
