#!/usr/bin/env python
"""GLS estimator for a monopole, dipole and shell offsets in the TF log-distance residuals.

Model, all columns in eta units (eta = LOGDIST = log10(d_z/d_TF)):

    eta_i = Z                          one common TF zeropoint (+ any uniform bubble)
          + K_i * U                    constant-velocity radial monopole  [km/s]
          + K_i * (n_i . V)            bulk flow                          [km/s]
          + sum_s S_s * 1{i in shell s}   shell offsets, outermost shell dropped
          + NS * 1{PHOTSYS == 'N'}     North/South calibration
          + NSZ * 1{N} * (z_i - zbar)  drift of that offset

K_i is the exact d(eta)/d(v_r) kernel from b0_transform.py, not the linearised one.

The zeropoint Z is ALWAYS free and common to every shell.  That is what makes the shell
offsets -- which are differences relative to the outermost shell -- the observable, and it
is also why the dense near-uniform mode in the covariance costs almost nothing here.

Noise:  Sigma = C_cat/25 + diag(K_i^2 * sigma_v_int^2), with sigma_v_int profiled.

Importable: use load_grouped(), build_design(), gls().
"""
import argparse
import os

import numpy as np
import h5py
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve

import b0_transform as b0

C_KMS = b0.C_KMS
ARIEL = b0.ARIEL
GROUPED = os.path.join(ARIEL, "output/DR2_TF_v5_2color_grouped/DESI-DR2_TF_pv_cat_v5b_grouped.fits")
GROUPED_COV = GROUPED.replace(".fits", "_cov.h5")
UNGROUPED = os.path.join(ARIEL, "output/DR2_TF_v5_2color_combined/DESI-DR2_TF_pv_cat_v5b.fits")
UNGROUPED_COV = UNGROUPED.replace(".fits", "_cov.h5")


def unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def load_grouped(cat=GROUPED, cov=GROUPED_COV, hdu="GROUPS", analysis_only=False):
    """Load catalogue + covariance. Returns a dict; cov is in eta^2 (already /25)."""
    with fits.open(cat) as hl:
        d = hl[hdu].data if hdu in [h.name for h in hl] else hl[1].data
        d = np.asarray(d)
    keep = np.ones(len(d), bool)
    if analysis_only and "ANALYSIS" in d.dtype.names:
        keep &= d["ANALYSIS"].astype(bool)
    z = np.asarray(d["Z_DESI_CMB"], float)
    eta = np.asarray(d["LOGDIST"], float)
    err = np.asarray(d["LOGDIST_ERR"], float)
    keep &= np.isfinite(z) & np.isfinite(eta) & np.isfinite(err) & (err > 0) & (z > 1e-3)

    with h5py.File(cov, "r") as f:
        n_cov = f["cov"].shape[0]
        if n_cov != len(d):
            raise SystemExit(f"cov has {n_cov} rows, catalogue {len(d)} -- row map unresolved")
        idx = np.where(keep)[0]
        # h5py has no 2-D fancy indexing: take the rows (idx is sorted), then the columns
        C = np.asarray(f["cov"][idx, :], dtype=np.float64)[:, idx] / 25.0
    C = 0.5 * (C + C.T)

    # the assertion g4_check.py makes: the covariance must describe LOGDIST_ERR
    dev = np.max(np.abs(np.sqrt(np.diag(C)) - err[keep]))
    if dev > 1e-6:
        raise SystemExit(f"sqrt(diag(cov))/5 != LOGDIST_ERR (max dev {dev:.2e})")

    ps = np.asarray(d["PHOTSYS"]).astype(str)[keep] if "PHOTSYS" in d.dtype.names else \
        np.full(keep.sum(), "S")
    return dict(z=z[keep], eta=eta[keep], err=err[keep], C=C,
                nhat=unit(np.asarray(d["RA"], float)[keep], np.asarray(d["DEC"], float)[keep]),
                photsys=ps, idx=np.where(keep)[0], n=int(keep.sum()))


def build_design(z, nhat, photsys, shells=None, monopole=True, dipole=True,
                 ns=True, nsz=True, zeropoint=True, extra=None):
    """Returns (A, names). Shell offsets are relative to the LAST shell, which is dropped."""
    K = b0.kernel(z)
    cols, names = [], []
    if zeropoint:
        cols.append(np.ones_like(z)); names.append("Z")
    if monopole:
        cols.append(K); names.append("U")
    if dipole:
        for j, lab in enumerate("xyz"):
            cols.append(K * nhat[:, j]); names.append(f"V{lab}")
    if shells is not None:
        edges = np.asarray(shells, float)
        which = np.clip(np.digitize(z, edges) - 1, 0, len(edges) - 2)
        for s in range(len(edges) - 2):          # drop the outermost => reference
            cols.append((which == s).astype(float))
            names.append(f"S{s}[{edges[s]:.3f}-{edges[s+1]:.3f}]")
    if ns:
        isn = (photsys == "N").astype(float)
        cols.append(isn); names.append("NS")
        if nsz:
            cols.append(isn * (z - z.mean())); names.append("NSz")
    if extra:
        for nm, v in extra.items():
            cols.append(np.asarray(v, float)); names.append(nm)
    return np.column_stack(cols), names


def gls(eta, A, C, K, sigma_v, alpha=1.0):
    """One GLS solve at fixed (sigma_v, alpha). Returns theta, P, chi2, logdet.

    alpha rescales the catalogue covariance. The v5b errors are known to be ~20%
    conservative (pull rms 0.861, i.e. variance factor 0.74), which drives chi2/dof
    below 1 and makes sigma_v_int unidentifiable if alpha is held at 1.
    """
    S = (alpha ** 2) * C + np.diag((K * sigma_v) ** 2)
    cf = cho_factor(S, lower=True)
    Ai = cho_solve(cf, A)
    F = A.T @ Ai
    P = np.linalg.inv(F)
    theta = P @ (A.T @ cho_solve(cf, eta))
    r = eta - A @ theta
    chi2 = float(r @ cho_solve(cf, r))
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return theta, P, chi2, logdet


def fit(data, shells=None, sigma_v=None, sigma_grid=None, iterate=True, verbose=True,
        **design_kw):
    """Full fit with sigma_v profiled (unless given) and one exact-map iteration."""
    z, eta, C = data["z"], data["eta"], data["C"]
    K = b0.kernel(z)
    A, names = build_design(z, data["nhat"], data["photsys"], shells=shells, **design_kw)

    alpha = 1.0
    if sigma_v is None:
        # joint profile over the error scale and the intrinsic velocity dispersion
        grid = np.arange(0.0, 601.0, 25.0) if sigma_grid is None else np.asarray(sigma_grid)
        agrid = np.arange(0.60, 1.41, 0.05)
        best, bl = (0.0, 1.0), -np.inf
        for a in agrid:
            for sv in grid:
                _, _, chi2, ld = gls(eta, A, C, K, sv, a)
                ll = -0.5 * chi2 - 0.5 * ld
                if ll > bl:
                    bl, best = ll, (sv, a)
        # refine both
        sv0, a0 = best
        for a in np.linspace(max(0.3, a0 - 0.05), a0 + 0.05, 11):
            for sv in np.linspace(max(0.0, sv0 - 25.0), sv0 + 25.0, 11):
                _, _, chi2, ld = gls(eta, A, C, K, sv, a)
                ll = -0.5 * chi2 - 0.5 * ld
                if ll > bl:
                    bl, best = ll, (sv, a)
        sigma_v, alpha = float(best[0]), float(best[1])

    theta, P, chi2, _ = gls(eta, A, C, K, sigma_v, alpha)

    shift = None
    if iterate:
        # replace the linear velocity terms by the exact map at the best fit and refit
        for _ in range(3):
            iU = names.index("U") if "U" in names else None
            iV = [names.index(f"V{l}") for l in "xyz" if f"V{l}" in names]
            vr = np.zeros_like(z)
            if iU is not None:
                vr += theta[iU]
            if iV:
                vr += data["nhat"] @ np.array([theta[i] for i in iV])
            exact = b0.eta_model(z, vr)
            linear = K * vr
            eta_corr = eta - (exact - linear)      # move the nonlinearity to the data side
            new, P, chi2, _ = gls(eta_corr, A, C, K, sigma_v, alpha)
            shift = np.max(np.abs((new - theta) / np.sqrt(np.diag(P))))
            theta = new
            if shift < 0.1:
                break

    out = dict(theta=theta, P=P, names=names, chi2=chi2, sigma_v=sigma_v, alpha=alpha,
               dof=len(eta) - A.shape[1], n=len(eta), shift=shift, A=A, K=K)
    if verbose:
        report(out)
    return out


def report(r):
    se = np.sqrt(np.diag(r["P"]))
    print(f"  N={r['n']}  npar={len(r['names'])}  chi2/dof={r['chi2']/r['dof']:.3f}  "
          f"sigma_v_int={r['sigma_v']:.0f} km/s  err_scale={r.get('alpha',1.0):.2f}" +
          (f"  exact-iter shift={r['shift']:.2f} sigma" if r["shift"] is not None else ""))
    for nm, t, s in zip(r["names"], r["theta"], se):
        unit_ = "km/s" if (nm == "U" or nm.startswith("V")) else "dex"
        print(f"    {nm:22s} {t:+12.5f} +/- {s:10.5f} {unit_:5s} ({t/s:+5.1f} sigma)")
    iV = [r["names"].index(f"V{l}") for l in "xyz" if f"V{l}" in r["names"]]
    if len(iV) == 3:
        V = r["theta"][iV]
        FV = np.linalg.inv(r["P"][np.ix_(iV, iV)])
        amp = float(np.linalg.norm(V))
        dchi2 = float(V @ FV @ V)
        samp = float(np.sqrt(V @ r["P"][np.ix_(iV, iV)] @ V) / amp) if amp > 0 else np.nan
        ra = np.degrees(np.arctan2(V[1], V[0])) % 360
        dec = np.degrees(np.arcsin(V[2] / amp))
        print(f"    |V| = {amp:.1f} +/- {samp:.1f} km/s  toward RA={ra:.1f} Dec={dec:+.1f}"
              f"   dchi2(V=0) = {dchi2:.1f} on 3 dof")
    if "U" in r["names"] and len(iV) == 3:
        iU = r["names"].index("U")
        cc = r["P"][iU, iV] / (np.sqrt(r["P"][iU, iU]) * np.sqrt(np.diag(r["P"])[iV]))
        print(f"    corr(U, Vx,Vy,Vz) = {cc[0]:+.2f} {cc[1]:+.2f} {cc[2]:+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zmax", type=float, default=0.03)
    p.add_argument("--ungrouped", action="store_true")
    p.add_argument("--no-ns", action="store_true")
    args = p.parse_args()

    cat, cov = (UNGROUPED, UNGROUPED_COV) if args.ungrouped else (GROUPED, GROUPED_COV)
    d = load_grouped(cat, cov, hdu="GROUPS", analysis_only=args.ungrouped)
    m = d["z"] < args.zmax
    sub = dict(z=d["z"][m], eta=d["eta"][m], err=d["err"][m],
               C=d["C"][np.ix_(np.where(m)[0], np.where(m)[0])],
               nhat=d["nhat"][m], photsys=d["photsys"][m], n=int(m.sum()))
    print(f"z < {args.zmax}  ({'ungrouped' if args.ungrouped else 'grouped'})")
    fit(sub, ns=not args.no_ns)


if __name__ == "__main__":
    main()
