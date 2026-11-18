#!/usr/bin/env python
"""Diagnose the low-redshift LOGDIST offset in the DR2 Tully-Fisher PV catalogues.

Every catalogue in catalogs/ gives a negative mean eta = LOGDIST = log10(d_z/d_TF) below
z ~ 0.02. The subcommands below test whether that is produced by (A) galaxies whose
intrinsic properties do not follow the TFR, (B) unaccounted peculiar velocities, (C)
selection truncation in the training set, or (D) framework-specific model choices.

    python diagnose_lowz.py t6_mock          mock control: pipeline bias vs true PV
    python diagnose_lowz.py t1_shape         mean eta(z); 1/z vs linear shape; joint flow fit
    python diagnose_lowz.py t3_lever         property regressions, transfer, inverse TFR
    python diagnose_lowz.py t4_spatial       monopole / dipole / shells (GLS), groups, Virgo
    python diagnose_lowz.py t5_external      pvhub predicted velocities / CF4 distances
    python diagnose_lowz.py t2_scatter       sigma_v from scatter growth (consistency only)
    python diagnose_lowz.py t8_size          D26 size-residual ratio test
    python diagnose_lowz.py t7_pairs         catalogue-pair decomposition
    python diagnose_lowz.py report           assemble LOWZ_DIAGNOSIS.md from results/*.json

Run inside the `tfpv` conda env. Outputs: output/lowz_diagnosis/{figs,results}/.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import lowz_common as L

b0, b1 = L.b0, L.b1


def _cats(args):
    return [c.strip() for c in args.catalogs.split(",") if c.strip()]


def _zbins(args):
    if args.zbins:
        return [float(x) for x in args.zbins.split(",")]
    return L.DEFAULT_ZBINS


def _print_bins(rows, label=""):
    print(f"  {label:>8} {'z range':>13} {'n':>6} {'<eta>':>8} {'+/-':>7} {'km/s':>7} {'+/-':>6} {'std':>7} {'err_rms':>8}")
    for r in rows:
        er = f"{r['err_rms']:8.4f}" if r.get("err_rms") is not None else "        "
        print(f"  {'':>8} {r['zlo']:6.3f}-{r['zhi']:<6.3f} {r['n']:6d} {r['mean']:+8.4f} {r['se']:7.4f} "
              f"{r['kms']:+7.0f} {r['kms_se']:6.0f} {r['std']:7.4f} {er}")


def _plot_bins(ax, rows, color, label, ls="-", key="mean", ekey="se", marker="o"):
    zm = [r["zmean"] for r in rows]
    ax.errorbar(zm, [r[key] for r in rows], yerr=[r[ekey] for r in rows], color=color, ls=ls,
                marker=marker, ms=5, lw=1.6, capsize=2, label=label)


# ============================================================================
# t6_mock
# ============================================================================
def cmd_t6_mock(args):
    """Mock control. The Hier pipeline applied to an AbacusSummit mock whose galaxies follow
    a single TFR with no curvature or non-TFR population, but have realistic peculiar
    velocities and the DR2-like selection. Decomposes eta = eta_true + (eta - eta_true):
    the first term is pure PV (plus footprint leakage of the mock's own bulk flow), the
    second is the pipeline's model bias b_model(z) (selection, extrapolation)."""
    import matplotlib.pyplot as plt
    L.style()
    bins = _zbins(args)
    m = L.load_mock(main_only=True)
    print(f"mock prediction catalogue: {m['n']} MAIN rows, z {m['z'].min():.4f}-{m['z'].max():.4f}")
    rows_eta = L.zbin_stats(m["z"], m["eta"], m["err"], bins, K=m["K"])
    rows_true = L.zbin_stats(m["z"], m["eta_true"], None, bins, K=m["K"])
    rows_bias = L.zbin_stats(m["z"], m["eta"] - m["eta_true"], None, bins, K=m["K"])
    # mock's own bulk motion of the observer frame + footprint leak: <v_r> and <n.V_mock>
    Vmock = m["v_vec"].mean(0)
    rows_vr = L.zbin_stats(m["z"], m["K"] * m["v_r"], None, bins, K=m["K"])
    print("\n  eta (pipeline prediction)"); _print_bins(rows_eta)
    print("\n  eta_true (pure PV)"); _print_bins(rows_true)
    print("\n  eta - eta_true (model bias b_model(z))"); _print_bins(rows_bias)
    print(f"\n  mean 3-D velocity of MAIN mock galaxies: |V|={np.linalg.norm(Vmock):.0f} km/s "
          f"(the mock sample's bulk flow; footprint leak <n.V> per bin below)")
    for r in rows_vr:
        s = (m["z"] >= r["zlo"]) & (m["z"] < r["zhi"])
        r["nV_leak_kms"] = float((m["nhat"][s] @ Vmock).mean())
        print(f"    {r['zlo']:.3f}-{r['zhi']:.3f}  <v_r>={r['kms']:+6.0f} km/s   <n.V_bulk>={r['nV_leak_kms']:+6.0f} km/s")

    # shape of the pipeline bias: is it linear in z (selection-like)?
    sh_bias = L.fit_shape(m["z"], m["eta"] - m["eta_true"], m["err"], K=m["K"])
    sh_eta = L.fit_shape(m["z"], m["eta"], m["err"], K=m["K"])
    for lab, sh in (("eta-eta_true", sh_bias), ("eta", sh_eta)):
        c3 = sh["models"]["M3"]
        print(f"  shape fit {lab:12s}: M3  b={c3['coef']['b_kms']:+.0f}+/-{c3['err']['b_kms']:.0f} km/s  "
              f"c={c3['coef']['c_dex_per_z']:+.3f}+/-{c3['err']['c_dex_per_z']:.3f} dex/z   "
              f"dAIC(M1,M2,M3 vs M0) = {sh['models']['M1']['dAIC_vs_M0']:+.1f}, "
              f"{sh['models']['M2']['dAIC_vs_M0']:+.1f}, {sh['models']['M3']['dAIC_vs_M0']:+.1f}")

    # population medians vs z, mock vs data (v10b), for the comparability check
    d = L.load("v10b")
    comp = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sm = (m["z"] >= lo) & (m["z"] < hi); sd = (d["z"] >= lo) & (d["z"] < hi)
        if sm.sum() < 10 or sd.sum() < 10:
            continue
        comp.append(dict(zlo=lo, zhi=hi, n_mock=int(sm.sum()), n_data=int(sd.sum()),
                         logV_mock=float(np.median(m["logV"][sm])), logV_data=float(np.median(d["logV"][sd])),
                         Mr_mock=float(np.median(m["M_r"][sm])), Mr_data=float(np.median(d["M_r"][sd])),
                         std_eta_mock=float(m["eta"][sm].std()), std_eta_data=float(d["eta"][sd].std())))
    print("\n  population comparability (medians): z bin | n mock/data | logV mock/data | M_r mock/data | std eta mock/data")
    for c in comp:
        print(f"    {c['zlo']:.3f}-{c['zhi']:.3f} | {c['n_mock']:5d}/{c['n_data']:5d} | {c['logV_mock']:.3f}/{c['logV_data']:.3f} "
              f"| {c['Mr_mock']:.2f}/{c['Mr_data']:.2f} | {c['std_eta_mock']:.3f}/{c['std_eta_data']:.3f}")

    # figure: three panels
    f, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    rows_data = L.zbin_stats(d["z"], d["eta"], d["err"], bins, K=d["K"])
    _plot_bins(axs[0], rows_data, L.COLORS["v10b"], "data v10b (Hier)")
    _plot_bins(axs[0], rows_eta, L.COLORS["mock"], "mock, pipeline eta")
    _plot_bins(axs[0], rows_true, L.MUTED, "mock, eta_true (pure PV)", marker="s")
    axs[0].set_title("Mean η vs redshift"); axs[0].set_ylabel("⟨η⟩ = ⟨log d_z/d_TF⟩  [dex]")
    _plot_bins(axs[1], rows_bias, L.COLORS["mock"], "mock: η − η_true")
    axs[1].set_title("Pipeline model bias b_model(z)"); axs[1].set_ylabel("⟨η − η_true⟩  [dex]")
    _plot_bins(axs[2], rows_data, L.COLORS["v10b"], "data v10b", key="kms", ekey="kms_se")
    _plot_bins(axs[2], rows_eta, L.COLORS["mock"], "mock pipeline", key="kms", ekey="kms_se")
    _plot_bins(axs[2], rows_true, L.MUTED, "mock true PV", key="kms", ekey="kms_se", marker="s")
    axs[2].set_title("Same, in velocity units ⟨η⟩/⟨K⟩"); axs[2].set_ylabel("km/s")
    for ax in axs:
        L.zero_line(ax); ax.set_xlabel("z_CMB"); ax.legend(loc="best")
    f.suptitle("T6 — Mock control: the Hier pipeline on an AbacusSummit mock (MAIN rows)", x=0.01, ha="left")
    L.fig("t6_mock")

    L.save_json("t6_mock", dict(n=m["n"], bins=bins, eta=rows_eta, eta_true=rows_true, bias=rows_bias,
                                vr=rows_vr, V_bulk_mock=Vmock, shape_bias=sh_bias, shape_eta=sh_eta,
                                comparability=comp))


# ============================================================================
# t1_shape
# ============================================================================
def _onepergroup(d, zmax):
    """Group-averaged catalogue (one row per GROUP_ID) below zmax, for shape fits whose
    noise model has no group covariance."""
    return L.group_average(d, zmax=zmax)


def cmd_t1_shape(args):
    """Baseline mean eta(z) per catalogue in dex and km/s, nested shape fits (constant / 1/z /
    linear / both) on group-averaged data, and a joint fit over catalogues with one shared
    radial-velocity amplitude U and per-catalogue slope/colour terms."""
    import matplotlib.pyplot as plt
    L.style()
    bins = _zbins(args)
    cats = _cats(args)
    res = {"bins": bins, "subset": args.subset, "zmax_shape": args.zmax, "catalogs": {}}
    data = {}
    f, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for c in cats:
        d = L.load(c, args.subset); data[c] = d
        rows = L.zbin_stats(d["z"], d["eta"], d["err"], bins, K=d["K"])
        print(f"\n== {c} ({d['info'].framework}, {d['info'].note}; subset={args.subset}, n={d['n']})")
        _print_bins(rows)
        g = _onepergroup(d, args.zmax)
        sh = L.fit_shape(g["z"], g["eta"], g["err"], K=g["K"])
        m1, m2, m3 = (sh["models"][k] for k in ("M1", "M2", "M3"))
        print(f"  shape (group-averaged, z<{args.zmax}, n={g['n']}, alpha={sh['alpha']:.2f}):")
        print(f"    M1 1/z only : b = {m1['coef']['b_kms']:+.0f} +/- {m1['err']['b_kms']:.0f} km/s     dAIC vs const = {m1['dAIC_vs_M0']:+.1f}")
        print(f"    M2 linear   : c = {m2['coef']['c_dex_per_z']:+.3f} +/- {m2['err']['c_dex_per_z']:.3f} dex/z  dAIC vs const = {m2['dAIC_vs_M0']:+.1f}")
        print(f"    M3 both     : b = {m3['coef']['b_kms']:+.0f} +/- {m3['err']['b_kms']:.0f} km/s, "
              f"c = {m3['coef']['c_dex_per_z']:+.3f} +/- {m3['err']['c_dex_per_z']:.3f}  dAIC vs const = {m3['dAIC_vs_M0']:+.1f}")
        res["catalogs"][c] = dict(n=d["n"], framework=d["info"].framework, bins=rows, shape=sh,
                                  n_groups=g["n"])
        col, ls = L.COLORS[c], L.LINESTYLE[d["info"].framework]
        _plot_bins(axs[0], rows, col, f"{c} ({d['info'].framework})", ls=ls)
        _plot_bins(axs[1], rows, col, f"{c}", ls=ls, key="kms", ekey="kms_se")
    axs[0].set_ylabel("⟨η⟩  [dex]"); axs[0].set_title("Mean log-distance ratio vs redshift")
    axs[1].set_ylabel("⟨η⟩/⟨K⟩  [km/s]"); axs[1].set_title("Same, as a mean radial velocity")
    axs[1].set_ylim(-400, 700)
    for ax in axs:
        L.zero_line(ax); ax.set_xlabel("z_CMB")
    axs[0].legend(loc="lower right", ncol=2)
    f.suptitle(f"T1 — Low-z offset in every catalogue ({args.subset} rows); solid = Hier, dashed = Hyper",
               x=0.01, ha="left")
    L.fig("t1_shape_bins")

    # ---- joint fit: shared U*K(z) + per-catalogue (a_c + c_c*(z - zref)) ------------------
    # One radial-velocity amplitude common to every catalogue (a PV is catalogue-independent),
    # with a catalogue-specific zero-point and linear-in-z drift (the pipeline/selection
    # trend that T6 shows is ~linear). Galaxy-property terms are deliberately NOT included:
    # the within-bin eta-logV slope is a pipeline feature present identically in the mock
    # (see t3_lever) and logV drifts with z, which makes such terms degenerate with K(z).
    # Uses the galaxies common to all requested catalogues, group-averaged.
    common = None
    for c in cats:
        s = set(data[c]["sga_id"].tolist())
        common = s if common is None else common & s
    common = np.array(sorted(common))
    zref = 0.03
    Xs, ys, ws = [], [], []
    ncat = len(cats)
    for i, c in enumerate(cats):
        d = data[c]
        sel = np.isin(d["sga_id"], common)
        sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == (d["n"],) else v) for k, v in d.items()}
        sub["n"] = int(sel.sum())
        g = L.group_average(sub, zmax=args.zmax)
        n = g["n"]
        X = np.zeros((n, 1 + 2 * ncat))
        X[:, 0] = g["K"]
        X[:, 1 + 2 * i] = 1.0
        X[:, 2 + 2 * i] = g["z"] - zref
        Xs.append(X); ys.append(g["eta"]); ws.append(1.0 / g["err"] ** 2)
    X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    alpha = L.profile_alpha(X, y, 1 / np.sqrt(w))
    beta, cov, chi2, _ = L.wls(X, y, w / alpha ** 2)
    se = np.sqrt(np.diag(cov))
    beta0, cov0, chi20, _ = L.wls(X[:, 1:], y, w / alpha ** 2)
    joint = dict(n_common=int(len(common)), n_rows=int(len(y)), zref=zref, alpha=alpha,
                 U_kms=float(beta[0]), U_err=float(se[0]), dchi2_U=float(chi20 - chi2),
                 per_catalog={c: dict(a=float(beta[1 + 2 * i]), a_err=float(se[1 + 2 * i]),
                                      c_dex_per_z=float(beta[2 + 2 * i]), c_err=float(se[2 + 2 * i]))
                              for i, c in enumerate(cats)},
                 caveat="rows of different catalogues are the same galaxies and are correlated; "
                        "the formal error on U is understated by up to sqrt(n_catalogs)")
    print(f"\n== joint fit over {ncat} catalogues, {len(common)} common galaxies (group-averaged), z<{args.zmax}, alpha={alpha:.2f}")
    print(f"  shared radial velocity U = {joint['U_kms']:+.0f} +/- {joint['U_err']:.0f} km/s (formal)   dchi2 for U = {joint['dchi2_U']:.1f}")
    for c, p in joint["per_catalog"].items():
        print(f"  {c:5s}: a(z={zref}) = {p['a']:+.4f}+/-{p['a_err']:.4f}   drift = {p['c_dex_per_z']:+.3f}+/-{p['c_err']:.3f} dex per unit z")
    res["joint"] = joint
    L.save_json("t1_shape", res)



# ============================================================================
# helpers shared by t3/t5
# ============================================================================
def _mock_bias(z):
    """Pipeline model bias b_model(z) = <eta - eta_true> from t6_mock.json, interpolated
    linearly in z between bin centres (constant beyond the ends)."""
    r = L.load_json("t6_mock")
    if r is None:
        raise SystemExit("run t6_mock first (needs results/t6_mock.json)")
    zc = np.array([b["zmean"] for b in r["bias"]]); m = np.array([b["mean"] for b in r["bias"]])
    return np.interp(np.asarray(z, float), zc, m)


def _pvhub_models():
    sys.path.insert(0, os.path.join(L.ARIEL, "data", "external"))
    from pvhub import pvhub as P
    return {"2M++_SDSS (Said+20/Carr+21)": P.TwoMPP_SDSS, "2M++ (Carrick+15)": P.TwoMPP_redshift}


def _vpred(model_cls, d, cache_tag):
    """Predicted radial peculiar velocity [km/s] from a pvhub reconstruction, cached per
    (model, catalogue, subset) under data/external/."""
    path = os.path.join(L.ARIEL, "data", "external", f"vpred_{cache_tag}.npy")
    if os.path.exists(path):
        v = np.load(path)
        if len(v) == d["n"]:
            return v
    rec = model_cls()
    v = np.asarray(rec.calculate_pv(d["ra"], d["dec"], d["z"], extrapolation=True), float)
    np.save(path, v)
    return v


# ============================================================================
# t5_external
# ============================================================================
def cmd_t5_external(args):
    """External velocity field. For every galaxy, the radial peculiar velocity predicted by the
    2M++ density reconstruction (pvhub: Said+2020/Carr+2021 and Carrick+2015 variants) is
    turned into eta_pred = eta_model(z, v_pred). Means per z bin are compared (regression
    slopes are not used: ~150 km/s reconstruction noise would bias them low). The remainder
    eta - eta_pred - b_model(z) (b_model from t6_mock) is what neither a known flow nor the
    pipeline explains."""
    import matplotlib.pyplot as plt
    L.style()
    bins = _zbins(args); cats = _cats(args)
    if args.source == "cf4":
        return _cf4(args)
    models = _pvhub_models()
    res = {"bins": bins, "subset": args.subset, "models": list(models), "catalogs": {}}
    f, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for c in cats:
        d = L.load(c, args.subset)
        bm = _mock_bias(d["z"])
        res["catalogs"][c] = {}
        print(f"\n== {c} ({d['info'].framework}; n={d['n']})")
        for mi, (mname, mcls) in enumerate(models.items()):
            tag = f"{mname.split(' ')[0].replace('+','p')}_{c}_{args.subset}"
            v = _vpred(mcls, d, tag)
            ok = np.isfinite(v)
            eta_pred = np.where(ok, b0.eta_model(d["z"], np.nan_to_num(v)), np.nan)
            rows_obs = L.zbin_stats(d["z"][ok], d["eta"][ok], d["err"][ok], bins, K=d["K"][ok])
            rows_pred = L.zbin_stats(d["z"][ok], eta_pred[ok], None, bins, K=d["K"][ok], nboot=50)
            rows_res = L.zbin_stats(d["z"][ok], (d["eta"] - eta_pred)[ok], d["err"][ok], bins, K=d["K"][ok])
            rows_rem = L.zbin_stats(d["z"][ok], (d["eta"] - eta_pred - bm)[ok], d["err"][ok], bins, K=d["K"][ok])
            # fraction of the low-z offset (z<0.025) accounted for by the predicted flow
            lo = (d["z"] < 0.025) & ok
            frac = float(np.sum(eta_pred[lo]) / np.sum(d["eta"][lo])) if np.sum(d["eta"][lo]) != 0 else np.nan
            sh_rem = L.fit_shape(d["z"][ok & (d["z"] < args.zmax)], (d["eta"] - eta_pred - bm)[ok & (d["z"] < args.zmax)],
                                 d["err"][ok & (d["z"] < args.zmax)])
            print(f"  [{mname}]  finite v_pred for {ok.sum()}/{d['n']}; predicted flow explains "
                  f"{100*frac:.0f}% of sum(eta) at z<0.025")
            print(f"  {'z range':>13} {'n':>5} {'<v_pred>':>8} {'<eta>obs':>9} {'<eta>pred':>9} {'obs-pred':>9} {'-b_model':>9} {'rem km/s':>9} {'+/-':>5}")
            for ro, rp, rr, rm in zip(rows_obs, rows_pred, rows_res, rows_rem):
                print(f"  {ro['zlo']:6.3f}-{ro['zhi']:<6.3f} {ro['n']:5d} {rp['kms']:+8.0f} {ro['mean']:+9.4f} {rp['mean']:+9.4f} "
                      f"{rr['mean']:+9.4f} {rm['mean']:+9.4f} {rm['kms']:+9.0f} {rm['kms_se']:5.0f}")
            m1 = sh_rem["models"]["M1"]
            print(f"  remainder shape fit z<{args.zmax}: residual radial velocity b = {m1['coef']['b_kms']:+.0f} +/- {m1['err']['b_kms']:.0f} km/s"
                  f"  (dAIC vs const {m1['dAIC_vs_M0']:+.1f})")
            res["catalogs"][c][mname] = dict(n_finite=int(ok.sum()), frac_lowz=frac, obs=rows_obs, pred=rows_pred,
                                             obs_minus_pred=rows_res, remainder=rows_rem, remainder_shape=sh_rem)
            if mi == 0:
                col, ls = L.COLORS[c], L.LINESTYLE[d["info"].framework]
                _plot_bins(axs[0], rows_obs, col, f"{c}", ls=ls, key="kms", ekey="kms_se")
                _plot_bins(axs[1], rows_pred, col, f"{c}", ls=ls, key="kms", ekey="kms_se")
                _plot_bins(axs[2], rows_rem, col, f"{c}", ls=ls, key="kms", ekey="kms_se")
    axs[0].set_title("Observed ⟨η⟩/⟨K⟩"); axs[1].set_title("2M++ predicted ⟨v_r⟩ for the same galaxies")
    axs[2].set_title("Remainder: η − η_pred − b_model(z)")
    for ax in axs:
        L.zero_line(ax); ax.set_xlabel("z_CMB"); ax.set_ylabel("km/s"); ax.set_ylim(-450, 650)
    axs[0].legend(loc="upper left", ncol=2)
    f.suptitle("T5 — External velocity field (pvhub 2M++_SDSS) vs the low-z offset; solid = Hier, dashed = Hyper", x=0.01, ha="left")
    L.fig("t5_external")
    L.save_json("t5_external", res)


def _cf4(args):
    """Cosmicflows-4 cross-match by PGC. Requires a locally downloaded CF4 table
    (--cf4-file; EDD 'CF4 All Distances' export with columns PGC, DM or log-distance)."""
    if not args.cf4_file or not os.path.exists(args.cf4_file):
        print("CF4 table not available (pass --cf4-file). The EDD export is form-driven and was not "
              "automated; the pvhub test above is the primary external comparison.")
        return
    raise SystemExit("CF4 branch not implemented in this version")


# ============================================================================
# t3_lever
# ============================================================================
def _wls_props(d, sel, props, pivots):
    X = [np.ones(sel.sum())]
    for p in props:
        X.append(d[p][sel] - pivots[p])
    X = np.column_stack(X)
    ok = np.all(np.isfinite(X), axis=1)
    beta, cov, chi2, _ = L.wls(X[ok], d["eta"][sel][ok], 1.0 / d["err"][sel][ok] ** 2)
    return beta, np.sqrt(np.diag(cov)), int(ok.sum())


def cmd_t3_lever(args):
    """Galaxy-property tests. (a) within-z-bin regressions of eta on DISTANCE-INDEPENDENT
    observables (logV, colours, axis ratio, surface brightness, apparent magnitude), for the
    data and for the mock pipeline bias; (b) property-split means at z<0.025 (does the offset
    live in one class of galaxy?); (c) transfer of the z>0.03 relation to low z; (d) inverse-TFR
    residual (immune to magnitude selection). Distance-dependent quantities (M_r, D26_kpc)
    are excluded from the covariates because eta is built from them."""
    import matplotlib.pyplot as plt
    L.style()
    bins = _zbins(args); cats = _cats(args)
    zb = [0.002, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07]
    pivots = {"logV": 2.1, "gr": 0.5, "rz": 0.3, "BA": 0.45, "sb_r": 22.5, "m_r": 15.0}
    res = {"zbins_reg": zb, "pivots": pivots, "catalogs": {}, "mock": {}}
    m = L.load_mock()
    mb = dict(m); mb["eta"] = m["eta"] - m["eta_true"]
    data = {c: L.load(c, args.subset) for c in cats}

    # (a) within-bin slopes --------------------------------------------------------------
    print("(a) within-bin WLS slopes of eta on distance-independent properties [dex per unit]")
    def slopes_table(d, props, label):
        out = []
        for lo, hi in zip(zb[:-1], zb[1:]):
            s = (d["z"] >= lo) & (d["z"] < hi)
            if s.sum() < 40:
                continue
            beta, se, n = _wls_props(d, s, props, pivots)
            out.append(dict(zlo=lo, zhi=hi, n=n, intercept=float(beta[0]),
                            **{p: float(beta[i + 1]) for i, p in enumerate(props)},
                            **{p + "_err": float(se[i + 1]) for i, p in enumerate(props)}))
        print(f"  {label}: " + ", ".join(props))
        for r in out:
            print(f"    {r['zlo']:.3f}-{r['zhi']:.3f} n={r['n']:5d} " +
                  " ".join(f"{p}={r[p]:+.3f}±{r[p+'_err']:.3f}" for p in props))
        return out
    # NOTE: apparent magnitude, surface brightness and angular size are excluded on purpose:
    # at fixed z they are monotonic in M_obs, from which eta is built, so they correlate
    # with eta trivially (through the TF residual itself), not through distance.
    props_data = ["logV", "gr", "rz", "BA"]
    props_mock = ["logV", "gr", "rz", "BA"]
    for c, d in data.items():
        res["catalogs"][c] = {"slopes": slopes_table(d, props_data, c)}
    res["mock"]["slopes_bias"] = slopes_table(mb, props_mock, "MOCK eta-eta_true (pipeline bias)")
    res["mock"]["slopes_eta"] = slopes_table(m, props_mock, "MOCK eta (pipeline prediction)")

    # (b) property splits at low z -------------------------------------------------------
    print("\n(b) property-split means at z<0.025, in km/s (<eta>/<K>); data vs mock-pipeline expectation")
    zlo_max = 0.025
    split_props = ["logV", "gr", "rz", "BA"]
    def split_table(d, props, label, zmax=zlo_max):
        s0 = d["z"] < zmax
        out = {}
        for p in props:
            if d.get(p) is None:
                continue
            x = d[p]; ok = s0 & np.isfinite(x)
            med = np.median(x[ok])
            lo_ = ok & (x <= med); hi_ = ok & (x > med)
            def mv(sel):
                K = d["K"][sel].mean(); e = d["eta"][sel]
                return e.mean() / K, e.std(ddof=1) / np.sqrt(sel.sum()) / K
            a, ea = mv(lo_); b, eb = mv(hi_)
            out[p] = dict(median=float(med), n_lo=int(lo_.sum()), n_hi=int(hi_.sum()), lo_kms=a, lo_err=ea,
                          hi_kms=b, hi_err=eb, diff_kms=b - a, diff_err=float(np.hypot(ea, eb)))
        print(f"  {label} (n={int(s0.sum())}):")
        for p, r in out.items():
            print(f"    {p:5s} split at {r['median']:7.3f}: below {r['lo_kms']:+5.0f}±{r['lo_err']:3.0f}  above {r['hi_kms']:+5.0f}±{r['hi_err']:3.0f}"
                  f"   diff {r['diff_kms']:+5.0f}±{r['diff_err']:3.0f} km/s")
        return out
    for c, d in data.items():
        res["catalogs"][c]["splits"] = split_table(d, split_props, c)
        # morphology / population classes
        if d["population"] is not None:
            s0 = d["z"] < zlo_max
            cls = {}
            for name in np.unique(d["population"][s0]):
                sel = s0 & (d["population"] == name)
                if sel.sum() >= 15:
                    K = d["K"][sel].mean()
                    cls[name] = dict(n=int(sel.sum()), kms=float(d["eta"][sel].mean() / K),
                                     err=float(d["eta"][sel].std(ddof=1) / np.sqrt(sel.sum()) / K))
            res["catalogs"][c]["population"] = cls
            print("    population: " + "  ".join(f"{k}: {v['kms']:+.0f}±{v['err']:.0f} km/s (n={v['n']})" for k, v in cls.items()))
        # a "normal spiral" subset: spiral, logV>2.0, BA<0.7, 0.35<g-r<0.65
        s0 = d["z"] < zlo_max
        normal = s0 & (d["logV"] > 2.0) & (d["BA"] < 0.7) & (d["gr"] > 0.35) & (d["gr"] < 0.65)
        if d["population"] is not None:
            normal &= d["population"] == "spiral"
        rest = s0 & ~normal
        def mv2(sel):
            K = d["K"][sel].mean(); e = d["eta"][sel]
            return float(e.mean() / K), float(e.std(ddof=1) / np.sqrt(max(sel.sum(), 2)) / K)
        nk, ne = mv2(normal); rk, re_ = mv2(rest)
        res["catalogs"][c]["normal_vs_rest"] = dict(n_normal=int(normal.sum()), normal_kms=nk, normal_err=ne,
                                                    n_rest=int(rest.sum()), rest_kms=rk, rest_err=re_)
        print(f"    'normal' (logV>2, BA<0.7, 0.35<g-r<0.65, spiral): {nk:+.0f}±{ne:.0f} km/s (n={normal.sum()})   rest: {rk:+.0f}±{re_:.0f} km/s (n={rest.sum()})")
    res["mock"]["splits_eta"] = split_table(m, ["logV", "gr", "rz", "BA"], "MOCK eta (pipeline)")
    res["mock"]["splits_bias"] = split_table(mb, ["logV", "gr", "rz", "BA"], "MOCK eta-eta_true")

    # (c) transfer: fit at 0.03<z<zmax with linear z drift, predict low-z bins -------------
    print("\n(c) transfer test: eta ~ a + c(z-0.045) + properties, fit at 0.03<z<%.2f, extrapolated to low z" % args.zmax)
    def transfer(d, props, label):
        hi = (d["z"] >= 0.03) & (d["z"] < args.zmax)
        X = [np.ones(hi.sum()), d["z"][hi] - 0.045] + [d[p][hi] - pivots[p] for p in props]
        X = np.column_stack(X); ok = np.all(np.isfinite(X), axis=1)
        beta, cov, _, _ = L.wls(X[ok], d["eta"][hi][ok], 1 / d["err"][hi][ok] ** 2)
        rows = []
        for lo, hi_ in zip(bins[:-1], bins[1:]):
            s = (d["z"] >= lo) & (d["z"] < hi_)
            if s.sum() < 10:
                continue
            Xs = np.column_stack([np.ones(s.sum()), d["z"][s] - 0.045] + [d[p][s] - pivots[p] for p in props])
            oks = np.all(np.isfinite(Xs), axis=1)
            pred = Xs[oks] @ beta
            K = d["K"][s][oks].mean()
            obs = d["eta"][s][oks].mean(); se = d["eta"][s][oks].std(ddof=1) / np.sqrt(oks.sum())
            rows.append(dict(zlo=lo, zhi=hi_, n=int(oks.sum()), obs=float(obs), pred=float(pred.mean()),
                             resid=float(obs - pred.mean()), resid_err=float(se), resid_kms=float((obs - pred.mean()) / K),
                             resid_kms_err=float(se / K)))
        print(f"  {label}: coefficients " + ", ".join(f"{n}={b:+.3f}" for n, b in zip(["a", "c_z"] + props, beta)))
        for r in rows:
            if r["zhi"] <= 0.03:
                print(f"    {r['zlo']:.3f}-{r['zhi']:.3f} n={r['n']:5d} obs={r['obs']:+.4f} pred={r['pred']:+.4f} resid={r['resid']:+.4f}±{r['resid_err']:.4f}  ({r['resid_kms']:+.0f}±{r['resid_kms_err']:.0f} km/s)")
        return dict(coef=dict(zip(["a", "c_z"] + props, map(float, beta))), bins=rows)
    for c, d in data.items():
        res["catalogs"][c]["transfer"] = transfer(d, ["logV", "gr", "BA"], c)
    res["mock"]["transfer_eta"] = transfer(m, ["logV", "gr", "BA"], "MOCK eta (pipeline; residual should track <eta_true>)")
    rows_true = L.zbin_stats(m["z"], m["eta_true"], None, bins, K=m["K"], nboot=50)
    print("    MOCK <eta_true> for reference: " + " ".join(f"{r['zlo']:.3f}-{r['zhi']:.3f}:{r['mean']:+.4f}" for r in rows_true if r["zhi"] <= 0.03))

    # (d) inverse TFR ---------------------------------------------------------------------
    print("\n(d) inverse-TFR residual (independent of the pipeline): logV ~ a + b*M_obs + colour + BA, fit at 0.03<z<zmax,")
    print("    uniform weights; eta_inv = resid/(5b). Immune to selection on M; a trend with z means the (logV, M_obs)")
    print("    relation itself drifts with z. Also: linear z-trend over all z, N/S split, and (Hyper) split by Hier membership.")
    hier_ids = set(L.load("v5b", "MAIN")["sga_id"].tolist())

    def inverse(d, label, sel_extra=None):
        base = np.ones(d["n"], bool) if sel_extra is None else sel_extra
        hi = base & (d["z"] >= 0.03) & (d["z"] < args.zmax)
        Xf = lambda s: np.column_stack([np.ones(s.sum()), d["M_r"][s] + 20.0, d["gr"][s] - 0.5, d["BA"][s] - 0.45])
        ok = np.all(np.isfinite(Xf(hi)), axis=1) & np.isfinite(d["logV"][hi])
        beta, cov, _, r = L.wls(Xf(hi)[ok], d["logV"][hi][ok], np.ones(ok.sum()))
        b = beta[1]; scat = float(np.std(r))
        allok = base & np.all(np.isfinite(Xf(np.ones(d["n"], bool))), axis=1) & np.isfinite(d["logV"])
        eta_inv = np.full(d["n"], np.nan)
        eta_inv[allok] = (d["logV"][allok] - Xf(allok) @ beta) / (5.0 * b)
        rows = []
        for lo, hi_ in zip(bins[:-1], bins[1:]):
            s = allok & (d["z"] >= lo) & (d["z"] < hi_)
            if s.sum() < 10:
                continue
            K = d["K"][s].mean(); e = eta_inv[s]
            rows.append(dict(zlo=lo, zhi=hi_, n=int(s.sum()), eta_inv=float(e.mean()), eta_inv_err=float(e.std(ddof=1) / np.sqrt(s.sum())),
                             kms=float(e.mean() / K), kms_err=float(e.std(ddof=1) / np.sqrt(s.sum()) / K),
                             eta_pipeline=float(d["eta"][s].mean())))
        # linear trend of eta_inv with z over the whole range, and the low-z residual about it
        s = allok & (d["z"] < args.zmax)
        Xz = np.column_stack([np.ones(s.sum()), d["z"][s] - 0.03])
        bz, cz, _, rz_ = L.wls(Xz, eta_inv[s], np.ones(s.sum()))
        lowz = d["z"][s] < 0.02
        trend = dict(slope_dex_per_z=float(bz[1]), slope_err=float(np.sqrt(cz[1, 1])), value_at_0p03=float(bz[0]),
                     lowz_resid_about_trend=float(rz_[lowz].mean()), lowz_resid_err=float(rz_[lowz].std(ddof=1) / np.sqrt(lowz.sum())),
                     n_lowz=int(lowz.sum()))
        print(f"  {label}: b = {b:+.4f} dex/mag (inverse TF slope {1/b:+.2f} mag/dex), logV scatter {scat:.3f} dex;  "
              f"trend d<eta_inv>/dz = {trend['slope_dex_per_z']:+.2f}±{trend['slope_err']:.2f} dex/z;  "
              f"z<0.02 residual about the trend = {trend['lowz_resid_about_trend']:+.4f}±{trend['lowz_resid_err']:.4f}")
        for r_ in rows:
            print(f"    {r_['zlo']:.3f}-{r_['zhi']:.3f} n={r_['n']:5d} <eta_inv>={r_['eta_inv']:+.4f}±{r_['eta_inv_err']:.4f} ({r_['kms']:+.0f}±{r_['kms_err']:.0f} km/s)   pipeline <eta>={r_['eta_pipeline']:+.4f}")
        return dict(b=float(b), scatter=scat, bins=rows, trend=trend)

    for c, d in data.items():
        # "Unsplit" (v11a) is included: it uses the Hier selection, just fit
        # as one population, so the inverse-TF test applies to it exactly as it
        # does to Hier. Without this it would be silently skipped.
        if d["info"].framework in ("Hier", "Unsplit") or c == "v7a":
            res["catalogs"][c]["inverse"] = inverse(d, c)
            if d["photsys"] is not None:
                res["catalogs"][c]["inverse_N"] = inverse(d, c + " PHOTSYS=N", d["photsys"] == "N")
                res["catalogs"][c]["inverse_S"] = inverse(d, c + " PHOTSYS=S", d["photsys"] == "S")
            if c == "v7a":
                inh = np.isin(d["sga_id"], list(hier_ids))
                res["catalogs"][c]["inverse_inHier"] = inverse(d, c + " galaxies also in Hier MAIN", inh)
                res["catalogs"][c]["inverse_notHier"] = inverse(d, c + " galaxies NOT in Hier MAIN", ~inh)
    res["mock"]["inverse"] = inverse(m, "MOCK (truth <eta_true> ~ 0 at all z)")

    # (e) does the drift track isophotal observables? one extra covariate at a time. NOTE: adding BOTH
    # surface brightness and angular size together with M_obs lets the regression rebuild the distance
    # modulus (SB - 5 log D26 - 2.5 log BA = m_r, m_r - M_obs = mu), so that combination is degenerate
    # with z and is deliberately not used.
    print("\n(e) inverse-TFR drift with one extra distance-independent covariate (v5b sample)")
    d = data["v5b"] if "v5b" in data else data[cats[0]]
    extra = {}
    for covs, lab in ((["gr", "BA"], "baseline"), (["gr", "BA", "sb_r"], "+SB_r"), (["gr", "BA", "D26"], "+angular D26"),
                      (["gr", "BA", "logV_err"], "+logV_err"), (["gr", "rz", "BA"], "+r-z")):
        hi = (d["z"] >= 0.03) & (d["z"] < args.zmax)
        Xf = lambda s: np.column_stack([np.ones(s.sum()), d["M_r"][s] + 20.0] + [d[c][s] - np.nanmedian(d[c]) for c in covs])
        ok = np.all(np.isfinite(Xf(hi)), axis=1) & np.isfinite(d["logV"][hi])
        beta, _, _, r = L.wls(Xf(hi)[ok], d["logV"][hi][ok], np.ones(ok.sum())); b = beta[1]
        allok = np.all(np.isfinite(Xf(np.ones(d["n"], bool))), axis=1) & np.isfinite(d["logV"])
        ei = np.full(d["n"], np.nan); ei[allok] = (d["logV"][allok] - Xf(allok) @ beta) / (5 * b)
        s = allok & (d["z"] < args.zmax)
        bz, cz, _, _ = L.wls(np.column_stack([np.ones(s.sum()), d["z"][s] - 0.03]), ei[s], np.ones(s.sum()))
        lowz = allok & (d["z"] >= 0.01) & (d["z"] < 0.02)
        extra[lab] = dict(covs=covs, b=float(b), trend=float(bz[1]), trend_err=float(np.sqrt(cz[1, 1])),
                          mean_0p01_0p02=float(np.nanmean(ei[lowz])))
        print(f"  {lab:14s} trend = {bz[1]:+.2f} ± {np.sqrt(cz[1,1]):.2f} dex/z   <eta_inv>(0.01<z<0.02) = {np.nanmean(ei[lowz]):+.4f}")
    res["inverse_extra_covariates"] = extra

    # (f) who are the low-z galaxies that the Hyper catalogue keeps but the Hier phase-space cut removes?
    if "v7a" in data:
        h = data["v7a"]; inh = np.isin(h["sga_id"], list(hier_ids)); lo = h["z"] < 0.025
        out = {}
        for lab, s in (("in_Hier", lo & inh), ("not_in_Hier", lo & ~inh)):
            morph, cnt = np.unique(h["morph"][s], return_counts=True)
            out[lab] = dict(n=int(s.sum()), logV=float(np.median(h["logV"][s])), M_r=float(np.median(h["M_r"][s])),
                            m_r=float(np.median(h["m_r"][s])), gr=float(np.nanmedian(h["gr"][s])), BA=float(np.median(h["BA"][s])),
                            sb_r=float(np.nanmedian(h["sb_r"][s])), D26_arcmin=float(np.median(h["D26"][s])),
                            logV_err=float(np.median(h["logV_err"][s])), eta_mean=float(h["eta"][s].mean()),
                            frac_logV_lt_1p9=float(np.mean(h["logV"][s] < 1.9)), frac_Mr_fainter_m18p5=float(np.mean(h["M_r"][s] > -18.5)),
                            morph={str(k): int(v) for k, v in zip(morph, cnt)})
        res["hyper_lowz_split_by_hier_membership"] = out
        print("\n(f) Hyper v7a MAIN galaxies at z<0.025 split by Hier-MAIN membership (medians):")
        for lab, o in out.items():
            print(f"  {lab:12s} n={o['n']:5d} logV={o['logV']:.3f} M_r={o['M_r']:.2f} g-r={o['gr']:.3f} BA={o['BA']:.2f} SB_r={o['sb_r']:.2f} "
                  f"D26={o['D26_arcmin']:.2f}' logV_err={o['logV_err']:.3f} <eta>={o['eta_mean']:+.4f} frac(logV<1.9)={o['frac_logV_lt_1p9']:.2f} morph={o['morph']}")

    # figure: slope of eta on logV vs z (data vs mock bias) and the low-z property splits
    f, axs = plt.subplots(1, 2, figsize=(12, 4))
    for c in cats:
        d = data[c]; sl = res["catalogs"][c]["slopes"]
        axs[0].errorbar([0.5 * (r["zlo"] + r["zhi"]) for r in sl], [r["logV"] for r in sl], yerr=[r["logV_err"] for r in sl],
                        color=L.COLORS[c], ls=L.LINESTYLE[d["info"].framework], marker="o", ms=4, label=c)
    sl = res["mock"]["slopes_bias"]
    axs[0].errorbar([0.5 * (r["zlo"] + r["zhi"]) for r in sl], [r["logV"] for r in sl], yerr=[r["logV_err"] for r in sl],
                    color=L.COLORS["mock"], marker="s", ms=5, label="mock: η − η_true")
    axs[0].set_xlabel("z_CMB"); axs[0].set_ylabel("∂η/∂logV within z bin  [dex/dex]")
    axs[0].set_title("The η–logV slope is a pipeline feature (also in the mock)"); L.zero_line(axs[0]); axs[0].legend(ncol=2)
    c0 = "v10b" if "v10b" in cats else cats[0]
    sp = res["catalogs"][c0]["splits"]; names = list(sp)
    y = np.arange(len(names))
    axs[1].errorbar([sp[p]["lo_kms"] for p in names], y - 0.12, xerr=[sp[p]["lo_err"] for p in names], fmt="o", color=L.COLORS["v5b"], label="below median")
    axs[1].errorbar([sp[p]["hi_kms"] for p in names], y + 0.12, xerr=[sp[p]["hi_err"] for p in names], fmt="o", color=L.COLORS["v5_3"], label="above median")
    axs[1].set_yticks(y); axs[1].set_yticklabels(names); axs[1].axvline(0, color=L.AXIS, lw=0.8)
    axs[1].set_xlabel("⟨η⟩/⟨K⟩ at z<0.025  [km/s]"); axs[1].set_title(f"Property splits of the low-z offset ({c0})"); axs[1].legend()
    f.suptitle("T3 — Galaxy-property tests", x=0.01, ha="left")
    L.fig("t3_lever")
    L.save_json("t3_lever", res)


# ============================================================================
# t4_spatial
# ============================================================================
SHELLS = [0.0, 0.008, 0.014, 0.020, 0.026, 0.032, 0.040]
VIRGO = dict(ra=187.7, dec=12.4, z=0.0038)


def _flow_fit_diag(z, eta, err, nhat, photsys, shells=SHELLS, sigma_grid=None, alpha_grid=None,
                   monopole=True, dipole=True, ns=True):
    """Monopole + dipole + shell-offset GLS with DIAGONAL noise  alpha^2 err^2 + (K sigma_v)^2,
    (sigma_v, alpha) profiled by maximum likelihood. Same design as b1_fit.build_design."""
    K = b0.kernel(z)
    A, names = b1.build_design(z, nhat, photsys, shells=shells, monopole=monopole, dipole=dipole, ns=ns, nsz=False)
    sgrid = np.arange(0.0, 601.0, 25.0) if sigma_grid is None else sigma_grid
    agrid = np.arange(0.6, 1.31, 0.05) if alpha_grid is None else alpha_grid
    best, bl = None, -np.inf
    for a in agrid:
        for sv in sgrid:
            var = (a * err) ** 2 + (K * sv) ** 2
            beta, cov, chi2, r = L.wls(A, eta, 1.0 / var)
            ll = -0.5 * chi2 - 0.5 * np.sum(np.log(var))
            if ll > bl:
                bl, best = ll, (sv, a, beta, cov, chi2, var)
    sv, a, beta, cov, chi2, var = best
    # one exact-map iteration (move the nonlinearity of eta(v) to the data side), as b1.fit does
    iU = names.index("U") if "U" in names else None
    iV = [names.index(f"V{l}") for l in "xyz" if f"V{l}" in names]
    vr = np.zeros_like(z)
    if iU is not None:
        vr += beta[iU]
    if iV:
        vr += nhat @ beta[iV]
    eta_corr = eta - (b0.eta_model(z, vr) - K * vr)
    beta, cov, chi2, r = L.wls(A, eta_corr, 1.0 / var)
    se = np.sqrt(np.diag(cov))
    out = dict(names=names, theta=beta, err=se, cov=cov, chi2=chi2, dof=len(eta) - A.shape[1],
               sigma_v=float(sv), alpha=float(a), n=int(len(eta)), A=A, var=var)
    if iV:
        V = beta[iV]; PV = cov[np.ix_(iV, iV)]
        amp = float(np.linalg.norm(V))
        out["V_amp"] = amp
        out["V_amp_err"] = float(np.sqrt(V @ PV @ V) / amp) if amp > 0 else np.nan
        out["V_dchi2"] = float(V @ np.linalg.inv(PV) @ V)
        ra = np.degrees(np.arctan2(V[1], V[0])) % 360; dec = np.degrees(np.arcsin(V[2] / amp))
        g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
        out["V_dir"] = dict(ra=ra, dec=dec, l=float(g.l.deg), b=float(g.b.deg))
        if iU is not None:
            nbar = nhat.mean(0)
            # correlation between U and the footprint-projected dipole V.<n>
            gvec = np.zeros(len(beta)); gvec[iV] = nbar
            cUV = float(gvec @ cov[:, iU] / np.sqrt(cov[iU, iU] * (gvec @ cov @ gvec)))
            out["corr_U_Vproj"] = cUV
            out["nbar"] = nbar
    return out


def _shuffle_null(z, eta, err, nhat, photsys, var, nshuf, seed=7, shells=SHELLS):
    """Sky-shuffle null for the dipole: permute n_hat among galaxies at fixed noise."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nshuf):
        nh = nhat[rng.permutation(len(z))]
        A, names = b1.build_design(z, nh, photsys, shells=shells, nsz=False)
        beta, cov, chi2, _ = L.wls(A, eta, 1.0 / var)
        iV = [names.index(f"V{l}") for l in "xyz"]
        V = beta[iV]; out.append(float(V @ np.linalg.inv(cov[np.ix_(iV, iV)]) @ V))
    return np.array(out)


from astropy.coordinates import SkyCoord
import astropy.units as u


def _report_flow(tag, r):
    print(f"  {tag}: n={r['n']} chi2/dof={r['chi2']/r['dof']:.3f} sigma_v={r['sigma_v']:.0f} km/s alpha={r['alpha']:.2f}")
    for nm, t, s in zip(r["names"], r["theta"], r["err"]):
        unit_ = "km/s" if (nm == "U" or nm.startswith("V")) else "dex "
        print(f"    {nm:22s} {t:+10.4f} +/- {s:8.4f} {unit_} ({t/s:+5.1f} sigma)")
    if "V_amp" in r:
        d = r["V_dir"]
        print(f"    |V| = {r['V_amp']:.0f} +/- {r['V_amp_err']:.0f} km/s toward l={d['l']:.0f} b={d['b']:+.0f} (RA={d['ra']:.0f} Dec={d['dec']:+.0f}); "
              f"dchi2(V=0)={r['V_dchi2']:.1f}/3 dof;  corr(U, V.<n>) = {r.get('corr_U_Vproj', np.nan):+.2f}")


def cmd_t4_spatial(args):
    """Spatial coherence of the low-z offset. Per catalogue (group-averaged, z<zmax_gls): GLS fit
    of zero-point Z + radial monopole U + bulk flow V + shell offsets + N/S term, with
    sigma_v and the error scale profiled; sky-shuffle null for the dipole; the SAME design
    fitted to the 2M++ predicted velocities of the same galaxies (apples to apples); within-
    vs between-group dispersion; eta vs angular distance from Virgo. --grouped reproduces the
    v5b full-covariance result of output/DR2_TF_v5_2color_bulkflow/."""
    import matplotlib.pyplot as plt
    L.style()
    cats = _cats(args)
    res = {"zmax_gls": args.zmax_gls, "shells": SHELLS, "catalogs": {}}
    models = _pvhub_models()
    mname, mcls = next(iter(models.items()))
    if args.grouped:
        print(f"== v5b grouped catalogue with full covariance (b1_fit.fit), z<{args.zmax_gls}")
        d = b1.load_grouped(L.GROUPED, L.GROUPED_COV, hdu="GROUPS")
        m = d["z"] < args.zmax_gls
        ii = np.where(m)[0]
        sub = dict(z=d["z"][m], eta=d["eta"][m], err=d["err"][m], C=d["C"][np.ix_(ii, ii)],
                   nhat=d["nhat"][m], photsys=d["photsys"][m], n=int(m.sum()))
        r = b1.fit(sub, shells=SHELLS, ns=True)
        res["v5b_grouped_fullcov"] = dict(names=r["names"], theta=r["theta"], err=np.sqrt(np.diag(r["P"])),
                                          sigma_v=r["sigma_v"], alpha=r["alpha"], chi2=r["chi2"], dof=r["dof"], n=r["n"])
    f, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in cats:
        d = L.load(c, args.subset)
        g = L.group_average(d, zmax=args.zmax_gls)
        print(f"\n== {c} ({d['info'].framework}), one row per group, z<{args.zmax_gls}: {g['n']} groups from {int((d['z']<args.zmax_gls).sum())} galaxies")
        r = _flow_fit_diag(g["z"], g["eta"], g["err"], g["nhat"], g["photsys"])
        _report_flow("data", r)
        null = _shuffle_null(g["z"], g["eta"], g["err"], g["nhat"], g["photsys"], r["var"], args.nshuffle)
        pval = float(np.mean(null >= r["V_dchi2"]))
        print(f"    sky-shuffle null ({args.nshuffle}): P(dchi2 >= observed) = {pval:.3f}")
        # same design on the 2M++ prediction for the same galaxies (group-averaged the same way)
        v = _vpred(mcls, d, f"{mname.split(' ')[0].replace('+','p')}_{c}_{args.subset}")
        dd = dict(d); dd["eta"] = d["K"] * np.nan_to_num(v); dd["err"] = d["err"]
        gp = L.group_average(dd, zmax=args.zmax_gls)
        rp = _flow_fit_diag(gp["z"], gp["eta"], gp["err"], gp["nhat"], gp["photsys"], sigma_grid=np.array([0.0]),
                            alpha_grid=np.array([1.0]))
        _report_flow(f"2M++ prediction, same galaxies/design ({mname})", rp)
        # literature flows projected on this sample, in the same shells
        lit = {}
        for ln, lf in L.LIT_FLOWS.items():
            proj = L.project_bulk_flow(g["nhat"], lf["V"], lf["l"], lf["b"])
            lit[ln] = float(proj.mean())
        print("    footprint-projected literature bulk flows <n.V> over these groups: " +
              "; ".join(f"{k.split(' ')[0]} {v:+.0f} km/s" for k, v in lit.items()))

        # groups: within vs between dispersion for multiplets at low z
        grp = {}
        sel = d["z"] < 0.03
        gid = d["group_id"][sel]; e = d["eta"][sel]; er = d["err"][sel]; zz = d["z"][sel]; KK = d["K"][sel]
        uniq, inv, cnt = np.unique(gid, return_inverse=True, return_counts=True)
        multi = cnt[inv] >= 2
        if multi.sum() > 20:
            gm = np.bincount(inv, e) / np.bincount(inv)
            within = e[multi] - gm[inv][multi]
            nm = cnt[inv][multi]
            s_within = float(np.sqrt(np.sum(within ** 2) / np.sum(1 - 1.0 / nm)))  # unbiased for group means
            s_err = float(np.sqrt(np.mean(er[multi] ** 2)))
            gmeans = gm[cnt >= 2]; gz = (np.bincount(inv, zz) / np.bincount(inv))[cnt >= 2]
            s_between = float(np.std(gmeans, ddof=1))
            grp = dict(n_multiplet_galaxies=int(multi.sum()), n_groups=int((cnt >= 2).sum()),
                       sigma_within=s_within, err_rms=s_err, sigma_between_groupmeans=s_between,
                       mean_z=float(gz.mean()))
            print(f"    groups z<0.03: {grp['n_groups']} multiplets ({grp['n_multiplet_galaxies']} galaxies); within-group rms {s_within:.4f} dex "
                  f"vs reported err rms {s_err:.4f}; rms of group means {s_between:.4f}")
        # Virgo
        sc = SkyCoord(ra=d["ra"] * u.deg, dec=d["dec"] * u.deg)
        sep = sc.separation(SkyCoord(ra=VIRGO["ra"] * u.deg, dec=VIRGO["dec"] * u.deg)).deg
        virgo = []
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 70), (70, 180)]:
            s = (sep >= lo) & (sep < hi) & (d["z"] < 0.015)
            if s.sum() >= 8:
                K = d["K"][s].mean()
                virgo.append(dict(lo=lo, hi=hi, n=int(s.sum()), kms=float(d["eta"][s].mean() / K),
                                  err=float(d["eta"][s].std(ddof=1) / np.sqrt(s.sum()) / K)))
        res["catalogs"][c] = dict(n_groups=g["n"], flow=dict(names=r["names"], theta=r["theta"], err=r["err"], sigma_v=r["sigma_v"],
                                                            alpha=r["alpha"], chi2=r["chi2"], dof=r["dof"], V_amp=r.get("V_amp"),
                                                            V_amp_err=r.get("V_amp_err"), V_dir=r.get("V_dir"), V_dchi2=r.get("V_dchi2"),
                                                            corr_U_Vproj=r.get("corr_U_Vproj"), shuffle_p=pval),
                                  flow_2mpp=dict(names=rp["names"], theta=rp["theta"], err=rp["err"], V_amp=rp.get("V_amp"), V_dir=rp.get("V_dir")),
                                  lit_projection=lit, groups=grp, virgo=virgo)
        col, ls = L.COLORS[c], L.LINESTYLE[d["info"].framework]
        if virgo:
            axs[1].errorbar([0.5 * (v_["lo"] + v_["hi"]) for v_ in virgo], [v_["kms"] for v_ in virgo],
                            yerr=[v_["err"] for v_ in virgo], color=col, ls=ls, marker="o", ms=4, label=c)
        # shells panel: shell offsets (relative to outermost) data vs 2M++ prediction
        iS = [i for i, nm in enumerate(r["names"]) if nm.startswith("S")]
        zc = [0.5 * (SHELLS[i] + SHELLS[i + 1]) for i in range(len(iS))]
        axs[0].errorbar(zc, r["theta"][iS], yerr=r["err"][iS], color=col, ls=ls, marker="o", ms=4, label=c)
    axs[0].set_title("Shell offsets relative to 0.032<z<0.040 (after Z, U, V)"); axs[0].set_xlabel("z_CMB"); axs[0].set_ylabel("dex")
    L.zero_line(axs[0]); axs[0].legend(ncol=2)
    axs[1].set_title("η at z<0.015 vs angular distance from Virgo"); axs[1].set_xlabel("separation from Virgo [deg]")
    axs[1].set_ylabel("⟨η⟩/⟨K⟩  [km/s]"); L.zero_line(axs[1]); axs[1].legend(ncol=2)
    f.suptitle("T4 — Spatial structure", x=0.01, ha="left")
    L.fig("t4_spatial")

    # sky map of the low-z offset: v10b (or first catalogue) z<0.025, cells of 15 deg in RA x 10 deg in Dec
    c0 = "v10b" if "v10b" in cats else cats[0]
    d = L.load(c0, args.subset)
    v = _vpred(mcls, d, f"{mname.split(' ')[0].replace('+','p')}_{c0}_{args.subset}")
    s = d["z"] < 0.025
    f, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, val, title in ((axs[0], d["eta_kms"], f"data {c0}: η/K at z<0.025"), (axs[1], np.nan_to_num(v), "2M++ predicted v_r, same galaxies")):
        ra = d["ra"][s]; dec = d["dec"][s]; x = val[s]
        rb = np.arange(0, 361, 15); db = np.arange(-20, 81, 10)
        H = np.full((len(db) - 1, len(rb) - 1), np.nan)
        for i in range(len(db) - 1):
            for j in range(len(rb) - 1):
                m_ = (ra >= rb[j]) & (ra < rb[j + 1]) & (dec >= db[i]) & (dec < db[i + 1])
                if m_.sum() >= 5:
                    H[i, j] = np.mean(x[m_])
        im = ax.pcolormesh(rb, db, H, cmap="RdBu_r", vmin=-400, vmax=400, shading="flat")
        ax.set_title(title); ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]"); ax.grid(False)
        ax.scatter(VIRGO["ra"], VIRGO["dec"], marker="x", color=L.INK, s=40)
        ax.annotate("Virgo", (VIRGO["ra"], VIRGO["dec"]), textcoords="offset points", xytext=(5, 5), fontsize=8, color=L.INK2)
        ax.annotate("Perseus-Pisces", (40, 35), fontsize=8, color=L.INK2); ax.annotate("Coma", (195, 28), fontsize=8, color=L.INK2)
    cb = f.colorbar(im, ax=axs, shrink=0.9, pad=0.02); cb.set_label("mean radial velocity  [km/s]  (cells with ≥5 galaxies)")
    f.suptitle("T4 — Sky distribution of the low-z offset: receding (red) vs infalling (blue)", x=0.01, ha="left")
    L.fig("t4_skymap")
    prev = L.load_json("t4_spatial")
    if prev:   # merge, so a --grouped or single-catalogue rerun does not discard earlier results
        prev.setdefault("catalogs", {}).update(res["catalogs"])
        for k in ("v5b_grouped_fullcov",):
            if k in res:
                prev[k] = res[k]
        res = prev
    L.save_json("t4_spatial", res)


# ============================================================================
# t2_scatter
# ============================================================================
def cmd_t2_scatter(args):
    """Consistency check only: sigma_v from the growth of eta scatter toward low z, profiling
    Sigma = alpha^2 err^2 + (K sigma_v)^2 about a mean model a + c(z - zref). The flat scatter
    in the Hier catalogues cannot rule out sigma_v ~ 300 km/s (see T2 discussion in the report)."""
    cats = _cats(args)
    res = {"catalogs": {}}
    for c in cats:
        d = L.load(c, args.subset)
        s = d["z"] < args.zmax
        z, eta, err, K = d["z"][s], d["eta"][s], d["err"][s], d["K"][s]
        X = np.column_stack([np.ones(s.sum()), z - 0.03])
        best, bl = None, -np.inf
        prof = {}
        for sv in np.arange(0, 701, 25):
            lbest = -np.inf
            for a in np.arange(0.5, 1.31, 0.02):
                var = (a * err) ** 2 + (K * sv) ** 2
                _, _, chi2, _ = L.wls(X, eta, 1 / var)
                ll = -0.5 * chi2 - 0.5 * np.sum(np.log(var))
                lbest = max(lbest, ll)
                if ll > bl:
                    bl, best = ll, (sv, a)
            prof[int(sv)] = lbest
        svs = np.array(sorted(prof)); lls = np.array([prof[k] for k in svs])
        ci = svs[lls >= lls.max() - 0.5]
        res["catalogs"][c] = dict(sigma_v=best[0], alpha=best[1], ci68=[float(ci.min()), float(ci.max())], n=int(s.sum()),
                                  profile={int(k): float(v) for k, v in prof.items()})
        print(f"  {c:5s}: sigma_v = {best[0]:.0f} km/s (68% {ci.min():.0f}-{ci.max():.0f}), error scale alpha = {best[1]:.2f}, n={s.sum()}")
    L.save_json("t2_scatter", res)


# ============================================================================
# t8_size
# ============================================================================
def cmd_t8_size(args):
    """D26 size test. A distance error moves M_obs and log D26_kpc together with a fixed ratio
    (dlogD/dM = -1/5: an infalling galaxy looks fainter AND smaller), whereas an intrinsic TFR
    deviation moves them with the intrinsic size-luminosity covariance kappa_int. Measures
    kappa_int at 0.03<z<zmax (PV negligible) and the slope and mean ratio at z<0.015."""
    cats = [c for c in _cats(args)]
    res = {"catalogs": {}}
    for c in cats:
        d = L.load(c, args.subset)
        ok = np.isfinite(d["logD26kpc"]) & np.isfinite(d["gr"]) & np.isfinite(d["BA"])
        hi = ok & (d["z"] >= 0.03) & (d["z"] < args.zmax)
        lo = ok & (d["z"] < 0.015)
        # size relation from high z: logD26kpc ~ a + b logV + c (g-r) + e BA + f (z-0.045)
        X = lambda s: np.column_stack([np.ones(s.sum()), d["logV"][s] - 2.1, d["gr"][s] - 0.5, d["BA"][s] - 0.45, d["z"][s] - 0.045])
        beta, _, _, _ = L.wls(X(hi), d["logD26kpc"][hi], np.ones(hi.sum()))
        dlogD = d["logD26kpc"] - (X(ok) @ beta if False else np.nan)  # placeholder
        dlogD = np.full(d["n"], np.nan); dlogD[ok] = d["logD26kpc"][ok] - X(ok) @ beta
        dM = -5.0 * d["eta"]            # M_obs - M_TF
        def kappa(s):
            x = dM[s]; y = dlogD[s]
            k = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
            # error of a regression slope
            r = y - (y.mean() + k * (x - x.mean()))
            ke = float(np.sqrt(np.sum(r ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2)))
            return k, ke
        k_hi, ke_hi = kappa(hi); k_lo, ke_lo = kappa(lo)
        mean_dM = float(dM[lo].mean()); mean_dD = float(dlogD[lo].mean())
        se_dD = float(dlogD[lo].std(ddof=1) / np.sqrt(lo.sum()))
        ratio = mean_dD / mean_dM if mean_dM != 0 else np.nan
        # expectation under pure-PV for the low-z mean: dlogD = -0.2 * dM
        exp_pv = -0.2 * mean_dM; exp_int = k_hi * mean_dM
        res["catalogs"][c] = dict(n_hi=int(hi.sum()), n_lo=int(lo.sum()), size_relation=beta, kappa_int=k_hi, kappa_int_err=ke_hi,
                                  kappa_lowz=k_lo, kappa_lowz_err=ke_lo, mean_dM_lowz=mean_dM, mean_dlogD_lowz=mean_dD,
                                  mean_dlogD_err=se_dD, ratio_lowz=ratio, expected_dlogD_if_PV=exp_pv, expected_dlogD_if_intrinsic=exp_int,
                                  scatter_dlogD=float(dlogD[hi].std()))
        print(f"  {c:5s}: kappa_int (0.03<z<{args.zmax}, n={hi.sum()}) = {k_hi:+.4f}±{ke_hi:.4f} dex/mag;  "
              f"slope at z<0.015 (n={lo.sum()}) = {k_lo:+.4f}±{ke_lo:.4f};  PV expectation -0.2000")
        print(f"         low-z means: <dM> = {mean_dM:+.4f} mag, <dlogD> = {mean_dD:+.4f}±{se_dD:.4f}  "
              f"(if PV: {exp_pv:+.4f}; if intrinsic with kappa_int: {exp_int:+.4f})   scatter(dlogD)={dlogD[hi].std():.3f}")
    L.save_json("t8_size", res)


# ============================================================================
# t7_pairs
# ============================================================================
def cmd_t7_pairs(args):
    """Catalogue-pair decomposition: delta eta between catalogues vs z, and its regression on
    (logV - 2.1, g-r - 0.5, z - 0.03). Within a framework the galaxies are identical so the
    regression is (nearly) exact; across frameworks it is noisy."""
    import matplotlib.pyplot as plt
    L.style()
    bins = _zbins(args)
    # v11a vs v11a_fixedslope is the controlled pair: identical sample and
    # cuts, differing only in slope_std_1, so their delta eta isolates the
    # redshift-dependent slope. v11a vs v5_3 is the split/unsplit comparison at
    # the same z cut; v11a vs v10b is unsplit/split at the same slope freedom.
    pairs = [("v10b", "v5b"), ("v5_3", "v5b"), ("v7a", "v5a"), ("v9", "v7a"),
             ("v5b", "v5a"), ("v10b", "v9"), ("v5_3", "v7a"),
             ("v11a", "v11a_fixedslope"), ("v11a", "v5_3"), ("v11a", "v10b")]
    data = {c: L.load(c, args.subset) for c in L.available()}
    # Drop pairs whose catalogues are not on disk yet (v11a until Phase 1 runs).
    skipped = [(a, b) for a, b in pairs if a not in data or b not in data]
    if skipped:
        print("  skipping pairs with no catalogue on disk: "
              + ", ".join(f"{a}-{b}" for a, b in skipped))
    pairs = [(a, b) for a, b in pairs if a in data and b in data]
    res = {"bins": bins, "pairs": {}}
    f, ax = plt.subplots(figsize=(8, 4.2))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
            "#7a3fbf", "#b07fd8", "#e34948"]
    for (a, b), col in zip(pairs, cols):
        da, db = data[a], data[b]
        ia, ib = L.match(da, db)
        z = da["z"][ia]; de = da["eta"][ia] - db["eta"][ib]
        rows = L.zbin_stats(z, de, None, bins, K=da["K"][ia], nboot=100)
        X = np.column_stack([np.ones(len(z)), da["logV"][ia] - 2.1, da["gr"][ia] - 0.5, z - 0.03])
        ok = np.all(np.isfinite(X), axis=1)
        beta, cov, chi2, r = L.wls(X[ok], de[ok], np.ones(ok.sum()))
        r2 = 1 - np.var(r) / np.var(de[ok])
        label = f"{a} − {b}"
        res["pairs"][label] = dict(n=int(len(ia)), bins=rows, coef=dict(zip(["a", "logV", "gr", "z"], map(float, beta))),
                                   coef_err=dict(zip(["a", "logV", "gr", "z"], map(float, np.sqrt(np.diag(cov))))), r2=float(r2),
                                   rms_resid=float(np.std(r)))
        print(f"  {label:12s} n={len(ia):5d}  dEta = {beta[0]:+.4f} {beta[1]:+.3f}(logV-2.1) {beta[2]:+.3f}(g-r-0.5) {beta[3]:+.3f}(z-0.03)   R2={r2:.2f} rms={np.std(r):.4f}")
        print("               by z: " + " ".join(f"{rr['zlo']:.3f}-{rr['zhi']:.3f}:{rr['mean']:+.4f}" for rr in rows))
        ax.errorbar([rr["zmean"] for rr in rows], [rr["mean"] for rr in rows], yerr=[rr["se"] for rr in rows], color=col, marker="o", ms=4, label=label)
    ax.set_xlabel("z_CMB"); ax.set_ylabel("⟨Δη⟩  [dex]"); L.zero_line(ax); ax.legend(ncol=2, fontsize=8)
    ax.set_title("T7 — Catalogue-pair differences vs redshift (matched galaxies)")
    L.fig("t7_pairs")
    L.save_json("t7_pairs", res)


# ============================================================================
# report
# ============================================================================
def _tbl(rows, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(f.format(r) for f in fmt) + " |\n" for r in rows)
    return head + body


def cmd_report(args):
    """Assemble output/lowz_diagnosis/tables.md (numbers appendix) from results/*.json. The
    narrative lives in LOWZ_DIAGNOSIS.md."""
    out = ["# Low-z diagnosis — numbers appendix\n", "Generated by `diagnose_lowz.py report` from `output/lowz_diagnosis/results/*.json`.\n"]
    t1 = L.load_json("t1_shape"); t5 = L.load_json("t5_external"); t6 = L.load_json("t6_mock"); t3 = L.load_json("t3_lever")
    t4 = L.load_json("t4_spatial"); t7 = L.load_json("t7_pairs"); t8 = L.load_json("t8_size"); t2 = L.load_json("t2_scatter")
    if t1:
        out.append("\n## T1 — mean η by redshift bin (MAIN rows)\n")
        for c, v in t1["catalogs"].items():
            out.append(f"\n**{c}** ({v['framework']}, n={v['n']})\n\n")
            out.append(_tbl(v["bins"], ["z range", "n", "⟨η⟩ [dex]", "± ", "⟨η⟩/⟨K⟩ [km/s]", "±"],
                            ["{0[zlo]:.3f}–{0[zhi]:.3f}", "{0[n]}", "{0[mean]:+.4f}", "{0[se]:.4f}", "{0[kms]:+.0f}", "{0[kms_se]:.0f}"]))
            m = v["shape"]["models"]
            out.append(f"\nShape fit (group-averaged, z<{t1['zmax_shape']}): 1/z-only b = {m['M1']['coef']['b_kms']:+.0f}±{m['M1']['err']['b_kms']:.0f} km/s "
                       f"(ΔAIC {m['M1']['dAIC_vs_M0']:+.1f}); linear-only c = {m['M2']['coef']['c_dex_per_z']:+.3f}±{m['M2']['err']['c_dex_per_z']:.3f} dex/z "
                       f"(ΔAIC {m['M2']['dAIC_vs_M0']:+.1f}); both: b = {m['M3']['coef']['b_kms']:+.0f}±{m['M3']['err']['b_kms']:.0f}, "
                       f"c = {m['M3']['coef']['c_dex_per_z']:+.3f}±{m['M3']['err']['c_dex_per_z']:.3f} (ΔAIC {m['M3']['dAIC_vs_M0']:+.1f}).\n")
        j = t1["joint"]
        out.append(f"\nJoint fit, shared radial velocity U = {j['U_kms']:+.0f} ± {j['U_err']:.0f} km/s (formal; {j['caveat']}).\n")
    if t6:
        out.append("\n## T6 — mock control (Hier pipeline on AbacusSummit, MAIN rows)\n\n")
        rows = [dict(zlo=a["zlo"], zhi=a["zhi"], n=a["n"], eta=a["mean"], true=b["mean"], bias=c["mean"], bias_se=c["se"])
                for a, b, c in zip(t6["eta"], t6["eta_true"], t6["bias"])]
        out.append(_tbl(rows, ["z range", "n", "⟨η⟩ pipeline", "⟨η_true⟩ (PV)", "⟨η−η_true⟩ bias", "±"],
                        ["{0[zlo]:.3f}–{0[zhi]:.3f}", "{0[n]}", "{0[eta]:+.4f}", "{0[true]:+.4f}", "{0[bias]:+.4f}", "{0[bias_se]:.4f}"]))
    if t5:
        out.append("\n## T5 — 2M++ predicted flow vs observed offset (first model)\n")
        for c, v in t5["catalogs"].items():
            mname = list(v)[0]; w = v[mname]
            out.append(f"\n**{c}** — {mname}; predicted flow accounts for {100*w['frac_lowz']:.0f}% of Ση at z<0.025\n\n")
            rows = [dict(zlo=a["zlo"], zhi=a["zhi"], n=a["n"], vpred=b["kms"], obs=a["mean"], pred=b["mean"], rem=d_["mean"], remk=d_["kms"], remse=d_["kms_se"])
                    for a, b, d_ in zip(w["obs"], w["pred"], w["remainder"])]
            out.append(_tbl(rows, ["z range", "n", "⟨v_pred⟩ km/s", "⟨η⟩ obs", "⟨η⟩ pred", "remainder [dex]", "remainder [km/s]", "±"],
                            ["{0[zlo]:.3f}–{0[zhi]:.3f}", "{0[n]}", "{0[vpred]:+.0f}", "{0[obs]:+.4f}", "{0[pred]:+.4f}", "{0[rem]:+.4f}", "{0[remk]:+.0f}", "{0[remse]:.0f}"]))
    if t3:
        out.append("\n## T3 — inverse Tully-Fisher residual η_inv (pipeline-independent)\n")
        for c, v in t3["catalogs"].items():
            if "inverse" in v:
                iv = v["inverse"]
                out.append(f"\n**{c}**: inverse slope b = {iv['b']:+.4f} dex/mag; trend d⟨η_inv⟩/dz = {iv['trend']['slope_dex_per_z']:+.2f}±{iv['trend']['slope_err']:.2f} dex/z\n\n")
                out.append(_tbl(iv["bins"], ["z range", "n", "⟨η_inv⟩", "±", "km/s", "pipeline ⟨η⟩"],
                                ["{0[zlo]:.3f}–{0[zhi]:.3f}", "{0[n]}", "{0[eta_inv]:+.4f}", "{0[eta_inv_err]:.4f}", "{0[kms]:+.0f}", "{0[eta_pipeline]:+.4f}"]))
        iv = t3["mock"]["inverse"]
        out.append(f"\n**mock**: trend {iv['trend']['slope_dex_per_z']:+.2f}±{iv['trend']['slope_err']:.2f} dex/z (truth: 0)\n\n")
        out.append(_tbl(iv["bins"], ["z range", "n", "⟨η_inv⟩", "±"], ["{0[zlo]:.3f}–{0[zhi]:.3f}", "{0[n]}", "{0[eta_inv]:+.4f}", "{0[eta_inv_err]:.4f}"]))
        if "inverse_extra_covariates" in t3:
            out.append("\nDrift with one extra covariate (v5b): " + "; ".join(f"{k}: {v['trend']:+.2f}±{v['trend_err']:.2f} dex/z" for k, v in t3["inverse_extra_covariates"].items()) + "\n")
        out.append("\n### Property splits at z<0.025 (⟨η⟩/⟨K⟩ in km/s, below / above the median)\n\n")
        for c, v in t3["catalogs"].items():
            sp = v["splits"]
            out.append(f"- **{c}**: " + "; ".join(f"{p}: {r['lo_kms']:+.0f}±{r['lo_err']:.0f} / {r['hi_kms']:+.0f}±{r['hi_err']:.0f}" for p, r in sp.items()))
            nr = v["normal_vs_rest"]
            out.append(f"; normal spirals {nr['normal_kms']:+.0f}±{nr['normal_err']:.0f} (n={nr['n_normal']}) vs rest {nr['rest_kms']:+.0f}±{nr['rest_err']:.0f} (n={nr['n_rest']})")
            if "population" in v:
                out.append("; " + ", ".join(f"{k} {r['kms']:+.0f}±{r['err']:.0f}" for k, r in v["population"].items()))
            out.append("\n")
        sp = t3["mock"]["splits_bias"]
        out.append("- **mock pipeline bias**: " + "; ".join(f"{p}: {r['lo_kms']:+.0f}±{r['lo_err']:.0f} / {r['hi_kms']:+.0f}±{r['hi_err']:.0f}" for p, r in sp.items()) + "\n")
        if "hyper_lowz_split_by_hier_membership" in t3:
            out.append("\n### Hyper v7a MAIN galaxies at z<0.025 by Hier-MAIN membership (medians)\n\n")
            rows = [dict(lab=k, **v) for k, v in t3["hyper_lowz_split_by_hier_membership"].items()]
            out.append(_tbl(rows, ["subset", "n", "logV", "M_r", "g−r", "BA", "SB_r", "D26 ['']", "σ_logV", "⟨η⟩", "frac logV<1.9"],
                            ["{0[lab]}", "{0[n]}", "{0[logV]:.3f}", "{0[M_r]:.2f}", "{0[gr]:.3f}", "{0[BA]:.2f}", "{0[sb_r]:.2f}", "{0[D26_arcmin]:.2f}",
                             "{0[logV_err]:.3f}", "{0[eta_mean]:+.4f}", "{0[frac_logV_lt_1p9]:.2f}"]))
    if t4:
        out.append(f"\n## T4 — monopole / dipole / shells (group-averaged, z<{t4['zmax_gls']}, diagonal errors)\n\n")
        rows = []
        for c, v in t4["catalogs"].items():
            f = v["flow"]; nm = f["names"]
            def g(n):
                i = nm.index(n); return f["theta"][i], f["err"][i]
            Z, Ze = g("Z"); U, Ue = g("U")
            fp = v["flow_2mpp"]
            rows.append(dict(c=c, Z=Z, Ze=Ze, U=U, Ue=Ue, V=f["V_amp"], Ve=f["V_amp_err"], l=f["V_dir"]["l"], b=f["V_dir"]["b"], p=f["shuffle_p"],
                             sv=f["sigma_v"], corr=f["corr_U_Vproj"], V2=fp["V_amp"], l2=fp["V_dir"]["l"], b2=fp["V_dir"]["b"]))
        out.append(_tbl(rows, ["cat", "Z [dex]", "U [km/s]", "|V| [km/s]", "V dir (l,b)", "shuffle p", "σ_v", "corr(U,V·n̄)", "2M++ |V| same design", "2M++ dir (l,b)"],
                        ["{0[c]}", "{0[Z]:+.4f}±{0[Ze]:.4f}", "{0[U]:+.0f}±{0[Ue]:.0f}", "{0[V]:.0f}±{0[Ve]:.0f}", "({0[l]:.0f}, {0[b]:+.0f})", "{0[p]:.3f}",
                         "{0[sv]:.0f}", "{0[corr]:+.2f}", "{0[V2]:.0f}", "({0[l2]:.0f}, {0[b2]:+.0f})"]))
        if "v5b_grouped_fullcov" in t4:
            r = t4["v5b_grouped_fullcov"]
            out.append("\nFull-covariance reproduction (v5b grouped, `b1_fit.fit`): " + "; ".join(f"{n} = {t:+.4f}±{e:.4f}" for n, t, e in zip(r["names"], r["theta"], r["err"]) if n in ("Z", "U", "NS")) +
                       f"; σ_v = {r['sigma_v']:.0f}, α = {r['alpha']:.2f}\n")
    if t2:
        out.append("\n## T2 — σ_v from scatter growth (consistency only)\n\n" + "; ".join(f"{c}: {v['sigma_v']:.0f} km/s (68% {v['ci68'][0]:.0f}–{v['ci68'][1]:.0f}), α={v['alpha']:.2f}" for c, v in t2["catalogs"].items()) + "\n")
    if t8:
        out.append("\n## T8 — D26 size test\n\n")
        rows = [dict(c=c, **v) for c, v in t8["catalogs"].items()]
        out.append(_tbl(rows, ["cat", "κ_int (z>0.03)", "slope z<0.015", "⟨ΔM⟩ z<0.015 [mag]", "⟨ΔlogD⟩", "±", "if PV", "if intrinsic"],
                        ["{0[c]}", "{0[kappa_int]:+.4f}±{0[kappa_int_err]:.4f}", "{0[kappa_lowz]:+.4f}±{0[kappa_lowz_err]:.4f}", "{0[mean_dM_lowz]:+.3f}",
                         "{0[mean_dlogD_lowz]:+.4f}", "{0[mean_dlogD_err]:.4f}", "{0[expected_dlogD_if_PV]:+.4f}", "{0[expected_dlogD_if_intrinsic]:+.4f}"]))
    if t7:
        out.append("\n## T7 — catalogue pairs (matched galaxies)\n\n")
        for lab, v in t7["pairs"].items():
            co = v["coef"]
            out.append(f"- **{lab}** (n={v['n']}): Δη = {co['a']:+.4f} {co['logV']:+.3f}(logV−2.1) {co['gr']:+.3f}(g−r−0.5) {co['z']:+.3f}(z−0.03), R²={v['r2']:.2f}; by z: "
                       + ", ".join(f"{r['zlo']:.3f}–{r['zhi']:.3f}: {r['mean']:+.4f}" for r in v["bins"]) + "\n")
    L.ensure_dirs()
    path = os.path.join(L.OUT_DIR, "tables.md")
    with open(path, "w") as fh:
        fh.write("".join(out))
    print(f"  -> {os.path.relpath(path, L.ARIEL)}")

# ============================================================================
# main
# ============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # L.available(), not L.ALL: the registry carries v11a before its FITS
    # exists, and the default must not make every command fail on a missing
    # file. Naming a catalogue explicitly still fails loudly, as it should.
    p.add_argument("--catalogs", default=",".join(L.available()))
    p.add_argument("--subset", default="MAIN", choices=["MAIN", "ANALYSIS", "ALL"])
    p.add_argument("--zmax", type=float, default=0.07, help="upper z for shape / regression fits")
    p.add_argument("--zbins", default=None, help="comma-separated bin edges")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in COMMANDS.items():
        sp = sub.add_parser(name, help=(fn.__doc__ or "").strip().split("\n")[0])
        if name == "t4_spatial":
            sp.add_argument("--grouped", action="store_true", help="use the v5b grouped catalogue + full covariance")
            sp.add_argument("--zmax_gls", type=float, default=0.04)
            sp.add_argument("--nshuffle", type=int, default=300)
        if name == "t5_external":
            sp.add_argument("--source", default="pvhub", choices=["pvhub", "cf4"])
            sp.add_argument("--cf4-file", default=None)
        sp.set_defaults(func=fn)
    args = p.parse_args(argv)
    L.ensure_dirs()
    args.func(args)


COMMANDS = {"t6_mock": cmd_t6_mock, "t1_shape": cmd_t1_shape, "t3_lever": cmd_t3_lever,
            "t5_external": cmd_t5_external, "t4_spatial": cmd_t4_spatial,
            "t2_scatter": cmd_t2_scatter, "t8_size": cmd_t8_size, "t7_pairs": cmd_t7_pairs, "report": cmd_report}

if __name__ == "__main__":
    main()
