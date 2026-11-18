#!/usr/bin/env python
"""Convert the raw v5b_2m combined catalogue to the delivered convention, and compare with v5b.

The v5b_2m fit was trained on absolute magnitudes moved to 2M++ peculiar-velocity-corrected
distances (make_pv_corrected_input.py). Because color_predict.py rebuilds
MU_ZCMB = app - abs_mag from the input, the RAW combined catalogue has

    LOGDIST_raw = 0.2 (mu(z_cos) - MU_TF) = log10 d(z_cos)/d_TF        (PV removed)

This script writes the delivered file with LOGDIST against the OBSERVED redshift, as in every
other catalogue, and keeps the PV-corrected quantities as extra columns:

    LOGDIST          = LOGDIST_raw - DMU_2MPP/5      = log10 d(z_obs)/d_TF   (v5b convention)
    LOGDIST_PVCORR   = LOGDIST_raw                   = log10 d(z_cos)/d_TF
    MU_ZCMB          = MU_ZCMB_raw - DMU_2MPP        (observed; identical to v5b's column)
    MU_ZCOS          = MU_ZCMB_raw
    {R,G,Z}_ABSMAG_SB26[_CORR]         restored to observed values (+ DMU_2MPP)
    {R,G,Z}_ABSMAG_SB26[_CORR]_PVCORR  the PV-corrected values the fit and MAIN cut used

LOGDIST_ERR, MU_TF, MU_ERR, MAIN, ANALYSIS, POPULATION and the covariance are unchanged.

    python finalize_v5b_2m.py              # write the delivered catalogue (+ copy to catalogs/)
    python finalize_v5b_2m.py --compare    # also compare with catalogs/DESI-DR2_TF_pv_cat_v5b.fits
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from astropy.io import fits

ARIEL = os.path.abspath(os.path.dirname(__file__))
RUN_DIR = os.path.join(ARIEL, "output", "DR2_TF_v5b_2m_2color_combined")
RAW = os.path.join(RUN_DIR, "DESI-DR2_TF_pv_cat_v5b_2m_raw.fits")
OUT = os.path.join(RUN_DIR, "DESI-DR2_TF_pv_cat_v5b_2m.fits")
CAT_DIR = os.path.join(ARIEL, "catalogs")
V5B = os.path.join(CAT_DIR, "DESI-DR2_TF_pv_cat_v5b.fits")
ABS_COLS = ["R_ABSMAG_SB26", "G_ABSMAG_SB26", "Z_ABSMAG_SB26",
            "R_ABSMAG_SB26_CORR", "G_ABSMAG_SB26_CORR", "Z_ABSMAG_SB26_CORR"]
C_KMS = 299792.458


def finalize(raw=RAW, out=OUT, deliver=True):
    with fits.open(raw) as hl:
        hdr0 = hl[0].header.copy()
        t = hl[1].data
        hdr1 = hl[1].header.copy()
        dmu = np.asarray(t["DMU_2MPP"], float)
        cols = []
        for col in t.columns:
            arr = np.array(t[col.name])
            if col.name == "LOGDIST":
                cols.append(fits.Column(name="LOGDIST", format=col.format, array=arr - dmu / 5.0))
                cols.append(fits.Column(name="LOGDIST_PVCORR", format=col.format, array=arr))
            elif col.name == "MU_ZCMB":
                cols.append(fits.Column(name="MU_ZCMB", format=col.format, array=arr - dmu))
                cols.append(fits.Column(name="MU_ZCOS", format=col.format, array=arr))
            elif col.name in ABS_COLS:
                cols.append(fits.Column(name=col.name, format=col.format, unit=col.unit, array=arr + dmu))
                cols.append(fits.Column(name=col.name + "_PVCORR", format=col.format, unit=col.unit, array=arr))
            else:
                cols.append(fits.Column(name=col.name, format=col.format, unit=col.unit, array=arr))
        hdu = fits.BinTableHDU.from_columns(cols, name=hdr1.get("EXTNAME", "DATA"))
        for k in hdr1:
            if k.startswith(("PV", "PARENT", "POPULATN", "GITSHA", "MADEUTC")) and k not in hdu.header:
                hdu.header[k] = hdr1[k]
        hdu.header["LDCONV"] = ("z_obs", "LOGDIST = log10 d(z_obs)/d_TF; LOGDIST_PVCORR uses z_cos (2M++)")
        hdu.header["HISTORY"] = "finalize_v5b_2m.py: LOGDIST=raw-DMU_2MPP/5, MU_ZCMB=raw-DMU_2MPP, ABSMAG restored (+DMU), *_PVCORR kept"
        fits.HDUList([fits.PrimaryHDU(header=hdr0), hdu]).writeto(out, overwrite=True)

    with fits.open(out) as hl:
        d = hl[1].data
        chk = np.asarray(d["LOGDIST"], float) - np.asarray(d["LOGDIST_PVCORR"], float) + np.asarray(d["DMU_2MPP"], float) / 5.0
        assert np.nanmax(np.abs(chk)) < 1e-6
        ident = np.asarray(d["R_MAG_SB26_CORR"], float) - np.asarray(d["R_ABSMAG_SB26_CORR"], float) - np.asarray(d["MU_ZCMB"], float)
        assert np.nanmax(np.abs(ident)) < 1e-5, "app - M_obs != MU_ZCMB after restoration"
        print(f"wrote {os.path.relpath(out, ARIEL)}: {len(d)} rows; MAIN={int(np.asarray(d['MAIN']).sum())}, "
              f"ANALYSIS={int(np.asarray(d['ANALYSIS']).sum())}; populations: "
              + ", ".join(f"{k}={v}" for k, v in zip(*np.unique(np.char.strip(np.asarray(d['POPULATION']).astype(str)), return_counts=True))))
    if not deliver:
        return
    # deliver: catalogue copy + covariance under the delivered name (hard link to avoid a 1 GB copy)
    shutil.copy2(out, os.path.join(CAT_DIR, os.path.basename(out)))
    raw_cov = raw.replace(".fits", "_cov.h5"); out_cov = out.replace(".fits", "_cov.h5")
    if os.path.exists(raw_cov) and not os.path.exists(out_cov):
        try:
            os.link(raw_cov, out_cov)
        except OSError:
            shutil.copy2(raw_cov, out_cov)
    print(f"copied to {os.path.relpath(os.path.join(CAT_DIR, os.path.basename(out)), ARIEL)}"
          + (f"; covariance -> {os.path.relpath(out_cov, ARIEL)}" if os.path.exists(out_cov) else ""))


def _stansummary_scalars(path, keep=("slope", "intercept", "sigma", "alpha", "beta", "mu_", "lambda", "theta", "frac")):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 10 or "[" in parts[0] or parts[0].endswith("__"):
                continue
            try:
                mean, sd, rhat = float(parts[1]), float(parts[3]), float(parts[-1])
            except ValueError:
                continue
            if any(parts[0].startswith(k) for k in keep):
                out[parts[0]] = (mean, sd, rhat)
    return out


def compare(out=OUT, v5b=V5B):
    import lowz_common as L
    a = fits.open(v5b)[1].data; b = fits.open(out)[1].data
    ia = {s: i for i, s in enumerate(np.asarray(a["SGA_ID"], np.int64))}
    pairs = [(ia[s], j) for j, s in enumerate(np.asarray(b["SGA_ID"], np.int64)) if s in ia]
    ja, jb = map(np.array, zip(*pairs))
    only_v5b = len(a) - len(ja); only_2m = len(b) - len(jb)
    print(f"\nmatched {len(ja)} galaxies; MAIN only in v5b: {only_v5b}; only in v5b_2m: {only_2m}")
    z = np.asarray(a["Z_DESI_CMB"], float)[ja]; K = L.b0.kernel(z)
    dmu_tf = np.asarray(b["MU_TF"], float)[jb] - np.asarray(a["MU_TF"], float)[ja]
    ld_a = np.asarray(a["LOGDIST"], float)[ja]; ld_b = np.asarray(b["LOGDIST"], float)[jb]; ld_pv = np.asarray(b["LOGDIST_PVCORR"], float)[jb]
    logV = np.asarray(a["LOGVROT"], float)[ja]
    print(f"\n{'z range':>13} {'n':>6} {'<dMU_TF> mag':>13} {'<eta> v5b':>10} {'<eta> 2m':>9} {'<eta_PVCORR> 2m':>16} {'km/s v5b':>9} {'km/s 2m':>8} {'km/s PVCORR':>12} {'+/-':>5}")
    rows = []
    for lo, hi in zip(L.DEFAULT_ZBINS[:-1], L.DEFAULT_ZBINS[1:]):
        s = (z >= lo) & (z < hi)
        if s.sum() < 10:
            continue
        k = K[s].mean(); se = ld_pv[s].std(ddof=1) / np.sqrt(s.sum())
        rows.append(dict(zlo=lo, zhi=hi, n=int(s.sum()), dmu_tf=float(dmu_tf[s].mean()), eta_v5b=float(ld_a[s].mean()),
                         eta_2m=float(ld_b[s].mean()), eta_pv=float(ld_pv[s].mean()), se=float(se), K=float(k)))
        r = rows[-1]
        print(f"{lo:6.3f}-{hi:<6.3f} {r['n']:6d} {r['dmu_tf']:+13.4f} {r['eta_v5b']:+10.4f} {r['eta_2m']:+9.4f} {r['eta_pv']:+16.4f} "
              f"{r['eta_v5b']/k:+9.0f} {r['eta_2m']/k:+8.0f} {r['eta_pv']/k:+12.0f} {se/k:5.0f}")
    X = np.column_stack([np.ones(len(z)), logV - 2.1, z - 0.03])
    beta, cov, _, _ = L.wls(X, dmu_tf, np.ones(len(z)))
    print(f"\ncalibration change dMU_TF = {beta[0]:+.4f} {beta[1]:+.4f}(logV-2.1) {beta[2]:+.3f}(z-0.03) mag  (rms about fit {np.std(dmu_tf - X @ beta):.4f})")
    sh = L.fit_shape(z, ld_pv, np.asarray(b["LOGDIST_ERR"], float)[jb])
    m1 = sh["models"]["M1"]
    print(f"shape of <LOGDIST_PVCORR>(z): 1/z amplitude b = {m1['coef']['b_kms']:+.0f} +/- {m1['err']['b_kms']:.0f} km/s (dAIC vs const {m1['dAIC_vs_M0']:+.1f})")
    sh0 = L.fit_shape(z, ld_a, np.asarray(a["LOGDIST_ERR"], float)[ja])
    print(f"  same for v5b LOGDIST:          b = {sh0['models']['M1']['coef']['b_kms']:+.0f} +/- {sh0['models']['M1']['err']['b_kms']:.0f} km/s")
    # posterior summaries side by side
    for pop, r5, r2 in (("spiral", "DR2_TF_spirals_v5_2color_spiral", "DR2_TF_spirals_v5b_2m_2color_spiral"),
                        ("irregular", "DR2_TF_irrs_v5_2color_irregular", "DR2_TF_irrs_v5b_2m_2color_irregular")):
        s5 = _stansummary_scalars(os.path.join(ARIEL, "output", r5, "stansummary.txt"))
        s2 = _stansummary_scalars(os.path.join(ARIEL, "output", r2, "stansummary.txt"))
        if not s2:
            continue
        print(f"\nposterior scalars, {pop}: v5b mean±sd | v5b_2m mean±sd (Rhat) | difference in sigma")
        for k in s2:
            if k in s5:
                m5, sd5, _ = s5[k]; m2, sd2, rh = s2[k]
                print(f"  {k:28s} {m5:+10.4f}±{sd5:.4f} | {m2:+10.4f}±{sd2:.4f} ({rh:.3f}) | {(m2-m5)/np.hypot(sd5,sd2):+5.1f}")
    L.save_json("v5b_2m_compare", dict(bins=rows, dmu_tf_fit=dict(a=float(beta[0]), logV=float(beta[1]), z=float(beta[2])),
                                       shape_pvcorr=sh, shape_v5b=sh0, only_v5b=only_v5b, only_2m=only_2m))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=RAW); p.add_argument("--out", default=OUT)
    p.add_argument("--compare", action="store_true"); p.add_argument("--no-finalize", action="store_true")
    p.add_argument("--no-deliver", action="store_true", help="do not copy to catalogs/ (testing)")
    a = p.parse_args()
    if not a.no_finalize:
        finalize(a.raw, a.out, deliver=not a.no_deliver)
    if a.compare:
        compare(a.out)


if __name__ == "__main__":
    main()
