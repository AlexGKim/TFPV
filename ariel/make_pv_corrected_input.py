#!/usr/bin/env python
"""Build a peculiar-velocity-corrected copy of a DR2 TF input catalogue (the "2m" inputs).

For every galaxy the radial peculiar velocity v predicted by a 2M++ reconstruction (pvhub,
default TwoMPP_SDSS = Said+2020 / Carr+2021) at (RA, DEC, Z_DESI_CMB) is turned into the
cosmological redshift and a distance-modulus shift

    z_cos = (1 + z_cmb) / (1 + v/c) - 1                 (doc/model3.tex convention; same map as
                                                         ariel/b0_transform.py, formerly under output/DR2_TF_v5_2color_bulkflow/)
    dmu   = 5 log10[ d_L(z_cos) / d_L(z_cmb) ]          astropy Planck18; H0 cancels in the ratio

and the absolute magnitudes are moved to the PV-corrected distance while apparent magnitudes,
errors, LOGVROT and redshifts are left untouched:

    {R,G,Z}_ABSMAG_SB26[_CORR]  ->  M - dmu          (infall, v<0  =>  dmu>0  =>  brighter)
    MU_ZCMB                     ->  MU_ZCMB + dmu    (keeps  app - M == MU_ZCMB  exactly)

Galaxies with no finite prediction (pvhub leaves the undefined central cells NaN below
z=0.01) or with z_cmb <= 1e-3 get dmu = 0 and PV_CORR_OK = False.

Added columns: V_PRED_2MPP [km/s], Z_COS_2MPP, DMU_2MPP [mag], PV_CORR_OK.

Usage:
    python make_pv_corrected_input.py --input data/DR2_TF_spirals_v5_spiral.fits \
        --output data/DR2_TF_spirals_v5_2m_spiral.fits
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys

import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits

C_KMS = 299792.458
ARIEL = os.path.abspath(os.path.dirname(__file__))
ABS_COLS = ["R_ABSMAG_SB26", "G_ABSMAG_SB26", "Z_ABSMAG_SB26",
            "R_ABSMAG_SB26_CORR", "G_ABSMAG_SB26_CORR", "Z_ABSMAG_SB26_CORR"]


def pvhub_model(name: str):
    sys.path.insert(0, os.path.join(ARIEL, "data", "external"))
    from pvhub import pvhub as P
    return getattr(P, name)()


def dmu_from_velocity(z_cmb, v_kms, cosmo=Planck18):
    """Distance-modulus shift mu(z_cos) - mu(z_cmb) for radial peculiar velocity v [km/s]."""
    z_cmb = np.asarray(z_cmb, float); v = np.asarray(v_kms, float)
    z_cos = (1.0 + z_cmb) / (1.0 + v / C_KMS) - 1.0
    ok = np.isfinite(v) & np.isfinite(z_cmb) & (z_cmb > 1e-3) & (z_cos > 1e-4)
    dmu = np.zeros_like(z_cmb)
    dl_obs = cosmo.luminosity_distance(z_cmb[ok]).value
    dl_cos = cosmo.luminosity_distance(z_cos[ok]).value
    dmu[ok] = 5.0 * np.log10(dl_cos / dl_obs)
    z_cos = np.where(ok, z_cos, z_cmb)
    return dmu, z_cos, ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="TwoMPP_SDSS", help="pvhub reconstruction class (TwoMPP_SDSS | TwoMPP_redshift | ...)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if os.path.exists(args.output) and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite")

    with fits.open(args.input) as hl:
        hdr0 = hl[0].header.copy()
        hdu = hl[1]
        tbl = hdu.data
        names = list(tbl.columns.names)
        hdr1 = hdu.header.copy()
        ra = np.asarray(tbl["RA"], float); dec = np.asarray(tbl["DEC"], float)
        z_cmb = np.asarray(tbl["Z_DESI_CMB"], float)

        rec = pvhub_model(args.model)
        v = np.asarray(rec.calculate_pv(ra, dec, np.clip(z_cmb, 1e-4, None), extrapolation=True), float)
        dmu, z_cos, ok = dmu_from_velocity(z_cmb, v)
        v_out = np.where(ok, v, np.nan)

        # modified copies of the magnitude columns
        new_cols = []
        for col in tbl.columns:
            arr = np.array(tbl[col.name])
            if col.name in ABS_COLS:
                arr = arr - dmu
            elif col.name == "MU_ZCMB":
                arr = arr + dmu
            new_cols.append(fits.Column(name=col.name, format=col.format, unit=col.unit, array=arr))
        new_cols += [fits.Column(name="V_PRED_2MPP", format="D", unit="km/s", array=v_out),
                     fits.Column(name="Z_COS_2MPP", format="D", array=z_cos),
                     fits.Column(name="DMU_2MPP", format="D", unit="mag", array=dmu),
                     fits.Column(name="PV_CORR_OK", format="L", array=ok)]
        out = fits.BinTableHDU.from_columns(new_cols, name=hdr1.get("EXTNAME", "DATA"))
        for k in ("PARENT", "POPULATN", "NPARENT", "NROWS", "MADEUTC", "GITSHA"):
            if k in hdr1:
                out.header[k] = hdr1[k]
        out.header["PVPARENT"] = (os.path.basename(args.input), "file the PV correction was applied to")
        out.header["PVMODEL"] = (f"pvhub.{args.model}", "2M++ reconstruction used for v_pred")
        out.header["PVCOSMO"] = ("Planck18", "cosmology for d_L(z_cos)/d_L(z_cmb)")
        out.header["PVNOK"] = (int((~ok).sum()), "rows left uncorrected (no finite v_pred or z<=1e-3)")
        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ARIEL, text=True).strip()
        except Exception:
            sha = "unknown"
        out.header["PVGITSHA"] = sha
        out.header["PVMADE"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        out.header["HISTORY"] = ("make_pv_corrected_input.py: ABSMAG -= dmu, MU_ZCMB += dmu, "
                                 "dmu = 5 log10 dL(z_cos)/dL(z_cmb), z_cos=(1+z_cmb)/(1+v/c)-1")
        fits.HDUList([fits.PrimaryHDU(header=hdr0), out]).writeto(args.output, overwrite=args.overwrite)

    # ---- checks and summary -----------------------------------------------------------
    with fits.open(args.output) as hl, fits.open(args.input) as hi:
        d = hl[1].data; o = hi[1].data
        app = np.asarray(d["R_MAG_SB26_CORR"], float)
        assert np.array_equal(np.asarray(d["R_MAG_SB26_CORR"]), np.asarray(o["R_MAG_SB26_CORR"])), "apparent mags changed"
        assert np.array_equal(np.asarray(d["LOGVROT"]), np.asarray(o["LOGVROT"])), "LOGVROT changed"
        assert np.array_equal(np.asarray(d["Z_DESI"]), np.asarray(o["Z_DESI"])), "Z_DESI changed"
        for c in ABS_COLS:
            diff = np.asarray(d[c], float) - np.asarray(o[c], float) + np.asarray(d["DMU_2MPP"], float)
            assert np.nanmax(np.abs(diff)) < 1e-6, f"{c} shift != -dmu"
        ident = app - np.asarray(d["R_ABSMAG_SB26_CORR"], float) - np.asarray(d["MU_ZCMB"], float)
        assert np.nanmax(np.abs(ident)) < 1e-6, "app - M_new != MU_ZCMB_new"
        dmu = np.asarray(d["DMU_2MPP"], float); z = np.asarray(d["Z_DESI_CMB"], float); okk = np.asarray(d["PV_CORR_OK"], bool)
        v = np.asarray(d["V_PRED_2MPP"], float)
        print(f"{args.output}: {len(d)} rows; corrected {okk.sum()}, uncorrected {(~okk).sum()}; model pvhub.{args.model}")
        print(f"  v_pred   : median {np.nanmedian(v):+.0f} km/s, 16-84% [{np.nanpercentile(v,16):+.0f}, {np.nanpercentile(v,84):+.0f}], min/max {np.nanmin(v):+.0f}/{np.nanmax(v):+.0f}")
        print(f"  dmu [mag]: median {np.median(dmu[okk]):+.4f}, 16-84% [{np.percentile(dmu[okk],16):+.4f}, {np.percentile(dmu[okk],84):+.4f}], max|dmu| {np.max(np.abs(dmu)):.3f}")
        print(f"  {'z range':>13} {'n':>6} {'<v_pred>':>9} {'<dmu>':>8} {'rms dmu':>8}")
        for lo, hi in zip([0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1], [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 1.0]):
            s = okk & (z >= lo) & (z < hi)
            if s.sum():
                print(f"  {lo:6.3f}-{hi:<6.3f} {s.sum():6d} {np.mean(v[s]):+9.0f} {np.mean(dmu[s]):+8.4f} {np.std(dmu[s]):8.4f}")
        print("  checks passed: apparent mags/LOGVROT/Z_DESI unchanged; all six ABSMAG columns shifted by -dmu; app - M_new == MU_ZCMB_new")


if __name__ == "__main__":
    main()
