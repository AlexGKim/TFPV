#!/usr/bin/env python
"""Exact eta <-> peculiar-velocity transform, and why the linearised one must not be used.

doc/model3.tex sets 1 + z_obs = (1 + z_cos)(1 + z_PV) with z_PV = v_r/c.  So a galaxy with
radial peculiar velocity v_r observed at CMB redshift z_obs sits at cosmological redshift

    z_cos = (1 + z_obs)/(1 + v_r/c) - 1

and its log-distance ratio is

    eta_model(z_obs, v_r) = log10[ d_L(z_obs) / d_L(z_cos) ]

Two things this script establishes, both needed by b1_fit.py:

1. The cosmology.  MU_ZCMB in the catalogues is FlatLambdaCDM(H0=100, Om0=0.3137).distmod
   to ~0.002 mag rms, and the mock's LOGDIST_TRUE is exactly the expression above.

2. Why we fit FORWARD in eta and never convert data to velocities.  eta is convex in v_r,
   so the inverse map applied to noisy eta gives a one-signed bias that grows toward low z
   -- i.e. it mimics a Hubble Bubble, the very thing we are trying to measure.  The design
   matrix therefore uses K = d eta / d v_r evaluated numerically on the exact map, not the
   analytic ln10*c*z/(1+z) kernel.

Run:  python b0_transform.py
"""
import argparse
import os

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

C_KMS = 299792.458
COSMO = FlatLambdaCDM(H0=100.0, Om0=0.3137)
ARIEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MOCK = "output/abacus_spec_c000_ph000_r001_2color/color_xonly_catalog.fits"
GROUPED = "output/DR2_TF_v5_2color_grouped/DESI-DR2_TF_pv_cat_v5b_grouped.fits"


def dl(z):
    """Luminosity distance, Mpc/h, vectorised and safe for tiny/negative z."""
    z = np.asarray(z, float)
    return COSMO.luminosity_distance(np.clip(z, 1e-8, None)).value


def eta_model(z_obs, v_r):
    """Exact log-distance ratio produced by radial peculiar velocity v_r [km/s]."""
    z_obs = np.asarray(z_obs, float)
    z_cos = (1.0 + z_obs) / (1.0 + np.asarray(v_r, float) / C_KMS) - 1.0
    return np.log10(dl(z_obs) / dl(z_cos))


def kernel(z_obs, dv=100.0):
    """K = d eta / d v_r at v_r = 0, by central difference on the exact map.

    Carries the d_L curvature and the (1+z) factors automatically; differs from the
    analytic (1+z)/(ln10 c z) kernel at the few-percent level.
    """
    return (eta_model(z_obs, +dv) - eta_model(z_obs, -dv)) / (2.0 * dv)


def kernel_linear(z_obs):
    """The naive kernel, for comparison only: eta = v /(ln10 * c * z/(1+z))."""
    z = np.asarray(z_obs, float)
    return (1.0 + z) / (np.log(10.0) * C_KMS * z)


def _shells():
    return [(0.001, 0.003), (0.003, 0.005), (0.005, 0.010),
            (0.010, 0.020), (0.020, 0.035), (0.035, 0.065)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", default=os.path.join(ARIEL, MOCK))
    p.add_argument("--grouped", default=os.path.join(ARIEL, GROUPED))
    args = p.parse_args()

    # ---- 1. cosmology check against MU_ZCMB -----------------------------------
    d = fits.open(args.grouped)["GROUPS"].data
    z = np.asarray(d["Z_DESI_CMB"], float)
    mu = np.asarray(d["MU_ZCMB"], float)
    g = np.isfinite(z) & np.isfinite(mu) & (z > 1e-3)
    pred = 5.0 * np.log10(dl(z[g])) + 25.0
    res = mu[g] - pred
    print("1. cosmology")
    print(f"   MU_ZCMB - FlatLambdaCDM(H0=100, Om0=0.3137).distmod : "
          f"median {np.median(res):+.5f}  rms {np.std(res):.5f}  max|.| {np.max(np.abs(res)):.5f} mag")

    # ---- 2. exact map reproduces the mock's LOGDIST_TRUE ----------------------
    print("\n2. exact map vs the mock's LOGDIST_TRUE")
    m = fits.open(args.mock)[1].data
    names = m.columns.names
    if "LOGDIST_TRUE" in names and all(c in names for c in ("vx", "vy", "vz", "RA", "DEC", "ZOBS")):
        ra = np.radians(np.asarray(m["RA"], float))
        dec = np.radians(np.asarray(m["DEC"], float))
        nh = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        v = np.array([np.asarray(m[c], float) for c in ("vx", "vy", "vz")])
        vr = (nh * v).sum(0)
        zo = np.asarray(m["ZOBS"], float)
        lt = np.asarray(m["LOGDIST_TRUE"], float)
        k = np.isfinite(zo) & np.isfinite(lt) & np.isfinite(vr) & (zo > 1e-3)
        pred = eta_model(zo[k], vr[k])
        A = np.column_stack([np.ones(k.sum()), pred])
        coef, *_ = np.linalg.lstsq(A, lt[k], rcond=None)
        r = np.corrcoef(pred, lt[k])[0, 1]
        print(f"   N={k.sum()}  slope {coef[1]:.6f}  intercept {coef[0]:+.2e}  r={r:.9f}")
        print(f"   max|eta_exact - LOGDIST_TRUE| = {np.max(np.abs(pred - lt[k])):.3e}")
    else:
        print("   mock lacks LOGDIST_TRUE / vx,vy,vz -- skipped")

    # ---- 3. the linearisation bias, which mimics a bubble ---------------------
    print("\n3. linearisation bias  <eta_linear - eta_exact>  (one-signed, grows toward low z)")
    print(f"   {'z range':>16} {'N':>7} {'mean bias':>11} {'rms':>9} {'median frac':>12}")
    if "LOGDIST_TRUE" in names:
        for lo, hi in _shells():
            s = k & (zo >= lo) & (zo < hi)
            if s.sum() < 20:
                continue
            ex = eta_model(zo[s], vr[s])
            lin = vr[s] * kernel_linear(zo[s])
            b = lin - ex
            frac = np.median(b / np.where(np.abs(ex) > 1e-6, ex, np.nan))
            print(f"   {lo:6.3f}-{hi:<6.3f} {s.sum():7d} {np.mean(b):+11.5f} "
                  f"{np.std(b):9.5f} {100*frac:11.1f}%")

    # ---- 4. the kernel actually used, vs the naive one ------------------------
    print("\n4. design-matrix kernel:  1/K  (km/s per unit eta), exact vs naive")
    print(f"   {'z':>8} {'1/K exact':>12} {'1/K naive':>12} {'diff':>8}")
    for zz in (0.005, 0.01, 0.02, 0.03, 0.05, 0.065):
        ke, kl = kernel(zz), kernel_linear(zz)
        print(f"   {zz:8.3f} {1/ke:12.1f} {1/kl:12.1f} {100*(kl/ke-1):7.2f}%")
    print("\n   b1_fit.py uses the exact K. A 350 km/s velocity at z=0.01 is "
          f"eta = {350*kernel(0.01):.4f}.")


if __name__ == "__main__":
    main()
