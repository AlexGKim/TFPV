#!/usr/bin/env python
"""Pre-filter a DESI/SGA catalog to a volume-limited angular-size sample.

WHY THIS EXISTS
---------------
SGA/HyperLeda selects on angular size: every galaxy in the parent catalog has
D26 > 20 arcsec (verified: the minimum is 20.0106 arcsec, nothing below 20).
The apparent "magnitude edge" near m_r ~ 17.3 is that angular cut convolved
with the size-luminosity relation, not a flux limit -- at fixed theta there is
no common magnitude wall, and the smallest galaxies reach m = 18.5.

An angular cut is distance-dependent, so in the model's variables it is
redshift-dependent, and the selection integral -- whose geometry is fixed in
(x, y) -- cannot represent it. That misspecification is the suspected cause of
the tophat MLE runaway at z > 0.03 and of the risk that a redshift-dependent
slope absorbs the deficit.

Requiring instead that a galaxy would have passed the 20 arcsec floor at ANY
redshift in the window makes the sample volume-limited in physical size. The
selection is then distance-independent and the existing fixed-geometry
machinery is correct as written, with no new Stan math.

    theta_i > theta_min * d_A(z_max) / d_A(z_i)

WHY A RATIO AND NOT A kpc THRESHOLD
-----------------------------------
The ratio is independent of the distance scale. That matters here: MU_ZCMB in
these catalogs differs from a Planck18 distance modulus by a constant offset
(~ -0.829 mag), so a kpc threshold means different things on different scales.
This script derives d_A from the catalog's OWN MU_ZCMB, and reports the implied
kpc threshold in the header purely for reference.

Note D26_kpc is NOT used: it is on a third scale again (theta * d_A(MU_ZCMB)
over D26_kpc has median 0.903 with 6.5% scatter), so cutting on it would
silently mix distance scales.

The cut is applied once, upstream, writing a new FITS file, so the rest of the
pipeline is an ordinary single-FITS run -- the pattern DR2_TWOPOP.md states and
make_population_subsets.py follows. It does NOT touch haty_max: the dwarf cut
is a population boundary in absolute magnitude and keeps doing its own job.

Usage:
    python make_size_limited.py                   # defaults
    python make_size_limited.py --force
"""
import argparse
import datetime as _dt
import os
import subprocess

import numpy as np
from astropy.io import fits
from astropy.table import Table

DEFAULT_INPUT = "data/DR2_TF_disks_v5_all.fits"
DEFAULT_OUTPUT = "data/DR2_TF_disks_v5_all_sizelim.fits"
THETA_COL = "D26"          # arcmin -- multiply by 60 for arcsec
Z_COL = "Z_DESI_CMB"
MU_COL = "MU_ZCMB"
ARCSEC_PER_RAD = 206264.806247


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _d_A_pc(z, mu):
    """Angular-diameter distance in pc from a distance modulus."""
    return 10.0 ** (mu / 5.0 + 1.0) / (1.0 + z) ** 2


def _mu_at(z_target, z, mu, n_bins=400):
    """Interpolate the catalog's own mu(z) relation at z_target.

    Uses the data rather than an imported cosmology so the threshold sits on
    the same distance scale as every per-galaxy d_A computed here.
    """
    good = np.isfinite(z) & np.isfinite(mu) & (z > 0.001) & (z < 0.2)
    zs = z[good]
    ms = mu[good]
    order = np.argsort(zs)
    zs, ms = zs[order], ms[order]
    k = max(len(zs) // n_bins, 1)
    zb = zs[::k]
    mb = np.array([np.median(ms[i:i + k]) for i in range(0, len(ms), k)])[:len(zb)]
    return float(np.interp(z_target, zb, mb))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--theta_min", type=float, default=20.0,
                   help="parent angular-size floor in arcsec (default 20.0)")
    p.add_argument("--z_max", type=float, default=0.065,
                   help="top of the redshift window the sample must be complete to")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to overwrite")

    t = Table.read(args.input)
    z = np.asarray(t[Z_COL], float)
    mu = np.asarray(t[MU_COL], float)
    theta = np.asarray(t[THETA_COL], float) * 60.0        # arcmin -> arcsec

    mu_zmax = _mu_at(args.z_max, z, mu)
    dA_zmax = _d_A_pc(args.z_max, mu_zmax)
    thresh_kpc = args.theta_min / ARCSEC_PER_RAD * dA_zmax / 1000.0

    finite = np.isfinite(theta) & np.isfinite(z) & np.isfinite(mu) & (z > 0)
    # theta_i > theta_min * d_A(z_max)/d_A(z_i)
    with np.errstate(invalid="ignore", divide="ignore"):
        need = args.theta_min * dA_zmax / _d_A_pc(z, mu)
    keep = finite & (theta > need)

    sub = t[keep]
    sub.write(args.output, overwrite=True)

    window = finite & (z > 0.03) & (z <= args.z_max)
    retained = int((keep & window).sum())
    print(f"parent              : {args.input}  ({len(t)} rows)")
    print(f"theta floor         : {args.theta_min} arcsec  (min in parent: {np.nanmin(theta):.4f})")
    print(f"z_max               : {args.z_max}   mu(z_max) = {mu_zmax:.4f}  (catalog scale)")
    print(f"implied threshold   : {thresh_kpc:.2f} kpc  [reference only; the cut is the ratio]")
    print(f"written             : {args.output}  ({len(sub)} rows)")
    print(f"0.03 < z <= {args.z_max}  : {retained}/{int(window.sum())} "
          f"= {100.0 * retained / max(int(window.sum()), 1):.1f}% retained")

    predicate = (f"{THETA_COL}*60 > {args.theta_min} arcsec * "
                 f"d_A({args.z_max})/d_A(z)  [volume-limited angular size]")
    with fits.open(args.output, mode="update") as hdul:
        h = hdul[1].header
        h["PARENT"] = (os.path.basename(args.input), "catalog this subset was cut from")
        h["NPARENT"] = (len(t), "rows in parent catalog")
        h["NROWS"] = (len(sub), "rows in this subset")
        h["THETAMIN"] = (args.theta_min, "parent angular-size floor [arcsec]")
        h["SIZEZMAX"] = (args.z_max, "redshift the sample is complete to")
        h["MUZMAX"] = (mu_zmax, "mu(z_max) on the catalog MU_ZCMB scale")
        h["DTHRKPC"] = (thresh_kpc, "implied physical size threshold [kpc]")
        h["NFITWIN"] = (retained, "retained with 0.03 < z <= z_max")
        h["MADEUTC"] = (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                        "UTC creation time")
        h["GITSHA"] = (_git_sha(), "ariel git revision")
        h.add_history(f"make_size_limited.py: {predicate}")


if __name__ == "__main__":
    main()
