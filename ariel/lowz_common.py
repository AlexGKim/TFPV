#!/usr/bin/env python
"""Shared loaders, statistics and figure helpers for the low-redshift LOGDIST diagnosis.

Conventions used throughout (see LOWZ_DIAGNOSIS.md):

    eta  = LOGDIST = 0.2*(MU_ZCMB - MU_TF) = log10(d_z / d_TF)
    eta  ~ +K(z) * v_r   with  K = d eta / d v_r  (exact kernel from b0_transform.py)

so a galaxy receding faster than Hubble flow (v_r > 0) has eta > 0, and the negative
low-z means in the catalogues correspond to v_r < 0, i.e. net infall toward us.

The exact eta <-> v_r map and the GLS machinery live in ariel/{b0_transform,
b1_fit}.py and are imported normally. They used to sit in
output/DR2_TF_v5_2color_bulkflow/ and be reached by inserting that directory
into sys.path -- but output/ is gitignored, so on a fresh clone the import
failed and nothing in this module could be used at all. They were moved up to
ariel/ and tracked; b2_shells.py and b4_nulls.py, which stayed behind in the
bulkflow directory, add ariel/ to sys.path themselves.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

ARIEL = os.path.abspath(os.path.dirname(__file__))
if ARIEL not in sys.path:
    sys.path.insert(0, ARIEL)
import b0_transform as b0  # noqa: E402
import b1_fit as b1        # noqa: E402

C_KMS = b0.C_KMS
OUT_DIR = os.path.join(ARIEL, "output", "lowz_diagnosis")
FIG_DIR = os.path.join(OUT_DIR, "figs")
RES_DIR = os.path.join(OUT_DIR, "results")
CAT_DIR = os.path.join(ARIEL, "catalogs")
MOCK_PRED = os.path.join(ARIEL, "output", "abacus_spec_c000_ph000_r001_2color",
                         "color_xonly_catalog.fits")
GROUPED = os.path.join(ARIEL, "output", "DR2_TF_v5_2color_grouped",
                       "DESI-DR2_TF_pv_cat_v5b_grouped.fits")
GROUPED_COV = GROUPED.replace(".fits", "_cov.h5")

# Sign check: eta_model(0.01, -150) must be about -0.022 (infall => eta < 0).
_chk = float(b0.eta_model(0.01, -150.0))
assert -0.025 < _chk < -0.019, f"eta sign convention changed: eta_model(0.01,-150)={_chk}"

DEFAULT_ZBINS = [0.002, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.070]


# ----------------------------------------------------------------------------
# Catalogue registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CatInfo:
    name: str
    file: str
    framework: str          # "Hier" | "Hyper" | "Unsplit"
    z_train_min: float | None
    zslope: bool            # redshift-dependent TF slope
    note: str


CATALOGS: dict[str, CatInfo] = {
    "v5b":  CatInfo("v5b",  "DESI-DR2_TF_pv_cat_v5b.fits",     "Hier",  0.01, False, "z>0.01 training"),
    # v5_3 is a FIXED-slope rerun. Its configs differ from v5's in "run" and
    # "z_obs_min" only and both set "model": "2color"; run_v5_3.sh's header says
    # the same. zslope was True here (and catalogs/README.txt listed v5_3 under
    # the z-dependent-slope analyses) -- both wrong, both corrected.
    "v5_3": CatInfo("v5_3", "DESI-DR2_TF_pv_cat_v5_3.fits",    "Hier",  0.03, False, "z>0.03 training, fixed slope"),
    "v10b": CatInfo("v10b", "DESI-DR2_TF_pv_cat_v10b.fits",    "Hier",  0.01, True,  "z-dependent slope"),
    "v5a":  CatInfo("v5a",  "DESI-DR2_TF_pv_cat_v5a.fits",     "Hyper", 0.01, False, "z>0.01"),
    "v7a":  CatInfo("v7a",  "DESI-DR2_TF_pv_cat_v7a.fits",     "Hyper", 0.03, False, "z>0.03"),
    "v9":   CatInfo("v9",   "DESI-DR2_TF_pv_test_cat_v9.fits", "Hyper", None, True,  "z-dependent slope (test)"),
    "v5b_2m": CatInfo("v5b_2m", "DESI-DR2_TF_pv_cat_v5b_2m.fits", "Hier", 0.01, False, "v5b with 2M++ PV-corrected training magnitudes"),
    # v11a: the combined (unsplit) disk sample, z>0.03 training, redshift-
    # dependent slope, with v11a_fixedslope as its control twin on the same
    # sample and cuts. See the framework note below for why these are neither
    # Hier nor Hyper.
    "v11a": CatInfo("v11a", "DESI-DR2_TF_pv_cat_v11a.fits", "Unsplit", 0.03, True,
                    "combined spirals+irregulars, z>0.03 training, z-dep slope"),
    "v11a_fixedslope": CatInfo("v11a_fixedslope", "DESI-DR2_TF_pv_cat_v11a_fixedslope.fits",
                               "Unsplit", 0.03, False,
                               "v11a control: same sample and cuts, fixed slope"),
}
HIER = ["v5b", "v5_3", "v10b"]
HYPER = ["v5a", "v7a", "v9"]
# A THIRD framework value, rather than filing v11a under Hyper. In this registry
# "Hyper" denotes a looser-selection pipeline: 28,856-29,152 MAIN rows and NO
# ANALYSIS hold-out. v11a will have ~13,100 MAIN and DOES have a hold-out (the
# existing DR2_TF_disks_v5_all_2color gives MAIN 16,918 / ANALYSIS 11,918 --
# essentially v5b's counts, not v5a's). Per LOWZ_DIAGNOSIS.md finding #3 the
# repo's Hyper catalogues keep 1,188 low-mass galaxies at z<0.025 that fail the
# Hier phase-space cut, and those triple the low-z offset. Registering v11a as
# Hyper would silently change the meaning of every HYPER-indexed comparison in
# diagnose_lowz.py and LOWZ_DIAGNOSIS.md. "Unsplit" says what it actually is:
# the Hier selection, fit as one population instead of two.
UNSPLIT = ["v11a", "v11a_fixedslope"]
ALL = HIER + HYPER + UNSPLIT

# Fixed categorical colour per catalogue (dataviz reference palette, fixed order).
COLORS = {"v5b": "#2a78d6", "v5_3": "#eb6834", "v10b": "#1baf7a",
          "v5a": "#eda100", "v7a": "#e87ba4", "v9": "#008300", "v5b_2m": "#e34948",
          "v11a": "#7a3fbf", "v11a_fixedslope": "#b07fd8",
          "mock": "#4a3aa7", "other": "#e34948"}
# ":" was the only free style; "-" and "--" are taken by Hier and Hyper.
LINESTYLE = {"Hier": "-", "Hyper": "--", "Unsplit": ":"}
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def available(names: list[str] | None = None) -> list[str]:
    """Those of *names* (default ALL) whose FITS file is actually in catalogs/.

    v11a and v11a_fixedslope are registered before they have been produced, so
    anything that defaults to "all catalogues" must go through here or every
    command fails with FileNotFoundError on a file that does not exist yet.
    """
    return [n for n in (ALL if names is None else names)
            if os.path.exists(os.path.join(CAT_DIR, CATALOGS[n].file))]


def _str(col):
    return np.char.strip(np.asarray(col).astype(str))


def load(name: str, subset: str = "MAIN", eta_abs_max: float = 1.0) -> dict:
    """Load one catalogue into a dict of aligned numpy arrays with derived columns.

    subset: "MAIN"     -> MAIN rows, minus visual-inspection rejects / OUTLIER (Hyper)
            "ANALYSIS" -> Hier hold-out rows only (not used in training); for Hyper
                          this is the same as MAIN (no flag exists)
            "ALL"      -> every row with finite eta
    """
    info = CATALOGS[name]
    with fits.open(os.path.join(CAT_DIR, info.file)) as hl:
        d = hl[1].data
        names = d.columns.names
        eta = np.asarray(d["LOGDIST"], float)
        err = np.asarray(d["LOGDIST_ERR"], float)
        z = np.asarray(d["Z_DESI_CMB"], float)
        keep = np.isfinite(eta) & np.isfinite(err) & (err > 0) & np.isfinite(z) & (z > 1e-3)
        keep &= np.abs(eta) < eta_abs_max
        if subset in ("MAIN", "ANALYSIS"):
            if "MAIN" in names:
                keep &= np.asarray(d["MAIN"]).astype(bool)
            if "OUTLIER" in names:
                keep &= ~np.asarray(d["OUTLIER"]).astype(bool)
            if "JOHN_VI" in names:
                keep &= _str(d["JOHN_VI"]) != "reject"
        if subset == "ANALYSIS" and "ANALYSIS" in names:
            keep &= np.asarray(d["ANALYSIS"]).astype(bool)

        def col(c, dtype=float):
            return np.asarray(d[c], dtype)[keep] if c in names else None

        out = dict(
            name=name, info=info, subset=subset, n=int(keep.sum()),
            eta=eta[keep], err=err[keep], z=z[keep],
            z_hel=col("Z_DESI"), ra=col("RA"), dec=col("DEC"),
            logV=col("LOGVROT"), logV_err=col("LOGVROT_ERR"),
            M_r=col("R_ABSMAG_SB26"), m_r=col("R_MAG_SB26_CORR"),
            mu_tf=col("MU_TF"), mu_z=col("MU_ZCMB"),
            BA=col("BA"), D26=col("D26"), D26_kpc=col("D26_kpc"),
            sga_id=col("SGA_ID", np.int64), pgc=col("PGC", np.int64),
            group_id=col("GROUP_ID", np.int64), group_mult=col("GROUP_MULT", int),
            photsys=_str(d["PHOTSYS"])[keep] if "PHOTSYS" in names else None,
            morph=_str(d["MORPHTYPE"])[keep] if "MORPHTYPE" in names else None,
            population=_str(d["POPULATION"])[keep] if "POPULATION" in names else None,
            analysis=col("ANALYSIS", bool) if "ANALYSIS" in names else None,
        )
        g = col("G_MAG_SB26_CORR"); r = col("R_MAG_SB26_CORR"); zz = col("Z_MAG_SB26_CORR")
        out["gr"] = g - r if g is not None else None
        out["rz"] = r - zz if zz is not None else None
        # r-band mean surface brightness inside the SB26 isophote (mag/arcsec^2):
        # D26 is in arcmin; area = pi (D26*60/2)^2 * BA
        if out["D26"] is not None and out["BA"] is not None:
            area = np.pi * (out["D26"] * 60.0 / 2.0) ** 2 * out["BA"]
            out["sb_r"] = r + 2.5 * np.log10(area)
        else:
            out["sb_r"] = None
        out["logD26kpc"] = np.log10(out["D26_kpc"]) if out["D26_kpc"] is not None else None
    out["K"] = b0.kernel(out["z"])
    out["eta_kms"] = out["eta"] / out["K"]
    out["nhat"] = b1.unit(out["ra"], out["dec"])
    gal = SkyCoord(ra=out["ra"] * u.deg, dec=out["dec"] * u.deg, frame="icrs").galactic
    out["l"] = gal.l.deg; out["b"] = gal.b.deg
    return out


def load_mock(path: str = MOCK_PRED, main_only: bool = True) -> dict:
    """Load the Hier prediction on the AbacusSummit mock (has LOGDIST_TRUE, vx,vy,vz)."""
    with fits.open(path) as hl:
        d = hl[1].data
        names = d.columns.names
        z = np.asarray(d["ZOBS"], float)
        eta = np.asarray(d["LOGDIST"], float)
        keep = np.isfinite(z) & (z > 1e-3) & np.isfinite(eta)
        if main_only and "MAIN" in names:
            keep &= np.asarray(d["MAIN"]).astype(bool)

        def col(c, dtype=float):
            return np.asarray(d[c], dtype)[keep] if c in names else None

        out = dict(name="mock", n=int(keep.sum()), z=z[keep], eta=eta[keep],
                   err=col("LOGDIST_ERR"), eta_true=col("LOGDIST_TRUE"),
                   zcos=col("ZCOS"), ra=col("RA"), dec=col("DEC"),
                   logV=col("LOGVROT"), M_r=col("R_ABSMAG_SB26"), M_r_true=col("R_ABSMAG_SB26_TRUE"),
                   m_r=col("R_MAG_SB26"), BA=col("BA_RATIO"),
                   mu_tf=col("MU_TF"), analysis=col("ANALYSIS", bool))
        v = np.column_stack([col("vx"), col("vy"), col("vz")])
        g = col("G_MAG_SB26"); r = col("R_MAG_SB26"); zz = col("Z_MAG_SB26")
        out["gr"] = g - r if g is not None else None
        out["rz"] = r - zz if zz is not None else None
    out["nhat"] = b1.unit(out["ra"], out["dec"])
    out["v_r"] = np.einsum("ij,ij->i", out["nhat"], v)
    out["v_vec"] = v
    out["K"] = b0.kernel(out["z"])
    out["eta_kms"] = out["eta"] / out["K"]
    return out


def match(a: dict, b: dict, key: str = "sga_id"):
    """Indices (ia, ib) of rows common to two loaded catalogues, by SGA_ID."""
    ia = {s: i for i, s in enumerate(a[key])}
    pairs = [(ia[s], j) for j, s in enumerate(b[key]) if s in ia]
    if not pairs:
        return np.array([], int), np.array([], int)
    ja, jb = map(np.array, zip(*pairs))
    return ja, jb


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def zbin_stats(z, eta, err=None, bins=DEFAULT_ZBINS, K=None, min_n=10, nboot=400, seed=1):
    """Per-bin mean eta with bootstrap error, plus the velocity-unit equivalent.

    The km/s column is <eta>/<K> (a mean radial velocity for the bin under eta = K v).
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float); eta = np.asarray(eta, float)
    if K is None:
        K = b0.kernel(z)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (z >= lo) & (z < hi) & np.isfinite(eta)
        n = int(s.sum())
        if n < min_n:
            continue
        e = eta[s]; k = K[s]
        boot = np.array([e[rng.integers(0, n, n)].mean() for _ in range(nboot)])
        m = float(e.mean()); se = float(boot.std())
        rows.append(dict(zlo=lo, zhi=hi, zmean=float(z[s].mean()), n=n, mean=m, se=se,
                         median=float(np.median(e)), std=float(e.std(ddof=1)),
                         err_rms=float(np.sqrt(np.mean(err[s] ** 2))) if err is not None else None,
                         kms=m / float(k.mean()), kms_se=se / float(k.mean())))
    return rows


def wls(X, y, w):
    """Weighted least squares. Returns (beta, cov, chi2, resid)."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float)
    Xw = X * w[:, None]
    F = X.T @ Xw
    cov = np.linalg.inv(F)
    beta = cov @ (Xw.T @ y)
    r = y - X @ beta
    chi2 = float(np.sum(w * r ** 2))
    return beta, cov, chi2, r


def profile_alpha(X, y, err):
    """Error scale alpha such that chi2/dof = 1 for the given design (errors are known
    to be ~20% conservative in these catalogues)."""
    beta, cov, chi2, r = wls(X, y, 1.0 / err ** 2)
    dof = len(y) - X.shape[1]
    return float(np.sqrt(chi2 / dof))


def fit_shape(z, eta, err, K=None, alpha=None, zref=None):
    """Nested shape models for the mean eta(z):

        M0: a                     constant dex offset (zero-point)
        M1: a + b*K(z)            radial velocity b [km/s]           (peculiar-velocity shape)
        M2: a + c*(z - zref)      linear drift in dex                (population / selection shape)
        M3: a + b*K(z) + c*(z - zref)

    Weights are 1/(alpha*err)^2 with alpha profiled on M3 unless given.
    Returns a dict of per-model coefficients, errors, chi2, AIC and deltas vs M0.
    """
    z = np.asarray(z, float); eta = np.asarray(eta, float); err = np.asarray(err, float)
    if K is None:
        K = b0.kernel(z)
    if zref is None:
        zref = float(np.mean(z))
    one = np.ones_like(z)
    designs = {"M0": (one[:, None], ["a"]),
               "M1": (np.column_stack([one, K]), ["a", "b_kms"]),
               "M2": (np.column_stack([one, z - zref]), ["a", "c_dex_per_z"]),
               "M3": (np.column_stack([one, K, z - zref]), ["a", "b_kms", "c_dex_per_z"])}
    if alpha is None:
        alpha = profile_alpha(designs["M3"][0], eta, err)
    w = 1.0 / (alpha * err) ** 2
    out = {"alpha": alpha, "zref": zref, "n": int(len(z)), "models": {}}
    for m, (X, nm) in designs.items():
        beta, cov, chi2, _ = wls(X, eta, w)
        k = X.shape[1]
        out["models"][m] = dict(coef={n: float(b) for n, b in zip(nm, beta)},
                                err={n: float(np.sqrt(cov[i, i])) for i, n in enumerate(nm)},
                                chi2=chi2, k=k, aic=chi2 + 2 * k)
    a0 = out["models"]["M0"]["aic"]
    for m in out["models"]:
        out["models"][m]["dAIC_vs_M0"] = out["models"][m]["aic"] - a0
    return out


def group_average(d: dict, zmax=None):
    """Collapse a loaded catalogue to one row per GROUP_ID (inverse-variance mean of eta,
    mean z and position). Shared PV within a group is preserved; virial motions average down."""
    gid = d["group_id"]
    sel = np.ones(d["n"], bool) if zmax is None else d["z"] < zmax
    uniq, inv = np.unique(gid[sel], return_inverse=True)
    w = 1.0 / d["err"][sel] ** 2
    sw = np.bincount(inv, w)
    eta = np.bincount(inv, w * d["eta"][sel]) / sw
    z = np.bincount(inv, w * d["z"][sel]) / sw
    ra = np.bincount(inv, w * d["ra"][sel]) / sw      # fine for group scales (<1 deg)
    dec = np.bincount(inv, w * d["dec"][sel]) / sw
    mult = np.bincount(inv)
    out = dict(name=d["name"] + "_grp", n=len(uniq), eta=eta, err=1.0 / np.sqrt(sw), z=z,
               ra=ra, dec=dec, mult=mult, group_id=uniq)
    out["K"] = b0.kernel(z); out["nhat"] = b1.unit(ra, dec)
    ps = d["photsys"][sel] if d["photsys"] is not None else np.full(sel.sum(), "S")
    first = np.zeros(len(uniq), int); first[inv[::-1]] = np.arange(sel.sum())[::-1]
    out["photsys"] = ps[first]
    return out


# ----------------------------------------------------------------------------
# Bulk flows
# ----------------------------------------------------------------------------
# Literature bulk flows in the CMB frame: amplitude [km/s], galactic l, b [deg], scale.
LIT_FLOWS = {
    "Watkins+2023 CF4 (R=150 Mpc/h)": dict(V=230.0, l=297.0, b=-6.0),
    "Carrick+2015 V_ext (2M++)":      dict(V=159.0, l=304.0, b=6.0),
    "Hoffman+2024 CF4 (R~100 Mpc/h)": dict(V=250.0, l=298.0, b=-8.0),
}
SUN_CMB = dict(V=369.82, l=264.021, b=48.253)   # Planck 2018 solar motion wrt CMB


def gal_unit(l_deg, b_deg):
    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    return b1.unit(c.ra.deg, c.dec.deg).ravel()


def project_bulk_flow(nhat, V, l, b):
    """Radial component n_hat . V for a bulk flow of amplitude V toward galactic (l, b)."""
    return V * (np.asarray(nhat) @ gal_unit(l, b))


def recompute_zcmb(z_hel, ra, dec):
    """z_cmb from heliocentric z with the Planck solar apex, exact (1+z) composition."""
    cos = b1.unit(ra, dec) @ gal_unit(SUN_CMB["l"], SUN_CMB["b"])
    return (1.0 + np.asarray(z_hel)) * (1.0 + SUN_CMB["V"] * cos / C_KMS) - 1.0


# ----------------------------------------------------------------------------
# I/O and figures
# ----------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(test: str, obj: dict):
    ensure_dirs()
    path = os.path.join(RES_DIR, f"{test}.json")
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=1)
    print(f"  -> {os.path.relpath(path, ARIEL)}")
    return path


def load_json(test: str):
    path = os.path.join(RES_DIR, f"{test}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "medium",
        "font.family": "sans-serif", "savefig.dpi": 160, "figure.dpi": 110,
    })


def fig(name: str, f=None):
    """Save the current (or given) matplotlib figure under output/lowz_diagnosis/figs/."""
    import matplotlib.pyplot as plt
    ensure_dirs()
    f = f or plt.gcf()
    path = os.path.join(FIG_DIR, f"{name}.png")
    f.savefig(path, bbox_inches="tight")
    plt.close(f)
    print(f"  -> {os.path.relpath(path, ARIEL)}")
    return path


def zero_line(ax):
    ax.axhline(0.0, color=AXIS, lw=0.8, zorder=0)
