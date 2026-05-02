"""
explore_residuals.py

Explore correlations between magnitude residuals (mean_pred - yhat) and
additional galaxy properties from the SGA FITS catalogue.

Usage
-----
python explore_residuals.py --run-dir output/DR1_zmax008 --kind tophat
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.stats import pearsonr

# Import prediction utilities from predict.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from predict import (
    _apply_main_cuts,
    read_cmdstan_posterior,
    ystar_pp_mean_sd_normal_vectorized,
    ystar_pp_mean_sd_tophat_vectorized,
)

# ── constants matching DESI() defaults ────────────────────────────────────────
V0 = 100.0
N_BINS = 15  # equal-count bins for binned plots


# ── helpers ───────────────────────────────────────────────────────────────────


def _resolve_run_dir() -> str:
    r"""Fall back to \ResultDir in paper/main.tex if --run-dir not given."""
    tex_path = Path(__file__).parent / "paper" / "main.tex"
    if tex_path.exists():
        tex = tex_path.read_text()
        m = re.search(r"\\newcommand\{\\ResultDir\}\{([^}]+)\}", tex)
        if m:
            return m.group(1).lstrip("../").rstrip("/")
    return "output/DR1_zmax008"


def _binned_mean(x, y, n_bins):
    """Equal-count bins of x; return bin centres, means, stds."""
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    edges = np.array_split(np.arange(len(x_s)), n_bins)
    centres, means, stds = [], [], []
    for idx in edges:
        if len(idx) == 0:
            continue
        centres.append(x_s[idx].mean())
        means.append(y_s[idx].mean())
        stds.append(y_s[idx].std())
    return np.array(centres), np.array(means), np.array(stds)


def _scatter_binned(ax, x, y, label, color, n_bins=N_BINS):
    ax.scatter(x, y, alpha=0.08, s=3, color=color)
    cx, cm, cs = _binned_mean(x, y, n_bins)
    ax.errorbar(
        cx,
        cm,
        yerr=cs,
        fmt="o-",
        color=color,
        lw=1.5,
        markersize=4,
        label=label,
        capsize=3,
    )


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default=None, help="Run name; sets run_dir to output/<run>/"
    )
    parser.add_argument(
        "--run-dir", default=None, help="Path to run directory (e.g. output/DR1_corr)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file (e.g., configs/dr1_v3.json)",
    )
    parser.add_argument("--kind", default="tophat", choices=["tophat", "normal"])
    args = parser.parse_args()

    from config_utils import apply_config

    cfg = apply_config(args)
    if args.run and not args.run_dir:
        args.run_dir = os.path.join("output", args.run)
    run_dir = args.run_dir or _resolve_run_dir()

    kind = args.kind
    out_dir = os.path.join(run_dir, "explore_residuals")
    os.makedirs(out_dir, exist_ok=True)

    fits_path = cfg.get("fits_file")
    print(f"run_dir : {run_dir}")
    print(f"kind    : {kind}")
    print(f"FITS    : {fits_path}")

    # ── load FITS — single pass, apply validity mask ───────────────────────────
    with fits.open(fits_path) as hdul:
        data = hdul[1].data  # type: ignore[union-attr]
        names = set(data.dtype.names or ())

        def _get_col(col_name, dtype: type = float):
            if col_name in names:
                return np.asarray(data[col_name], dtype=dtype)
            else:
                print(f"Warning: column '{col_name}' missing, filling with NaNs")
                if dtype is str:
                    return np.full(len(data), "", dtype=object)
                return np.full(len(data), np.nan, dtype=dtype)

        from mag_utils import get_mag_cols

        col_abs, col_abs_err, col_app = get_mag_cols(names)

        V = _get_col("V_0p4R26")
        V_err = _get_col("V_0p4R26_ERR")
        yhat_raw = _get_col(col_abs)
        sigma_y_raw = _get_col(col_abs_err)
        zobs_raw = _get_col("Z_DESI")

        # extra columns
        ba = _get_col("BA")
        morphtype = _get_col("MORPHTYPE", dtype=str)
        d26_kpc = _get_col("D26_kpc")
        g_mag = _get_col("G_MAG_SB26_CORR")
        r_mag = _get_col(col_app)
        z_mag = _get_col("Z_MAG_SB26_CORR")
        g_mag_obs = _get_col("G_MAG_SB26")
        r_mag_obs = _get_col("R_MAG_SB26")
        z_mag_obs = _get_col("Z_MAG_SB26")
        sma_sb26 = _get_col("SMA_SB26")
        sma_sb22 = _get_col("SMA_SB22")
        g_sma50 = _get_col("G_SMA50")
        r_sma50 = _get_col("R_SMA50")
        z_sma50 = _get_col("Z_SMA50")
        group_mult = _get_col("GROUP_MULT")
        photsys = _get_col("PHOTSYS", dtype=str)

    # validity mask (mirrors load_xy_and_uncertainties_from_desi)
    mask = (
        np.isfinite(V)
        & np.isfinite(V_err)
        & np.isfinite(yhat_raw)
        & np.isfinite(sigma_y_raw)
        & np.isfinite(zobs_raw)
        & (V > 0)
        & (V_err > 0)
        & (sigma_y_raw >= 0)
    )

    def _m(arr):
        return arr[mask]

    xhat = np.log10(_m(V) / V0)
    sigma_x = _m(V_err) / (_m(V) * np.log(10))
    yhat = _m(yhat_raw)
    sigma_y = _m(sigma_y_raw)
    zobs = _m(zobs_raw)

    ba = _m(ba)
    morphtype = _m(morphtype)
    d26_kpc = _m(d26_kpc)
    g_mag = _m(g_mag)
    r_mag = _m(r_mag)
    z_mag = _m(z_mag)
    g_mag_obs = _m(g_mag_obs)
    r_mag_obs = _m(r_mag_obs)
    z_mag_obs = _m(z_mag_obs)
    sma_sb26 = _m(sma_sb26)
    sma_sb22 = _m(sma_sb22)
    g_sma50 = _m(g_sma50)
    r_sma50 = _m(r_sma50)
    z_sma50 = _m(z_sma50)
    group_mult = _m(group_mult)
    photsys = _m(photsys)

    print(f"N galaxies (valid mask): {mask.sum()}")

    # derived quantities
    g_r = g_mag - r_mag
    r_z = r_mag - z_mag
    g_z = g_mag - z_mag
    g_r_obs = g_mag_obs - r_mag_obs
    r_z_obs = r_mag_obs - z_mag_obs
    g_z_obs = g_mag_obs - z_mag_obs
    with np.errstate(invalid="ignore", divide="ignore"):
        sma_ratio = np.where(sma_sb22 > 0, sma_sb26 / sma_sb22, np.nan)

    # load bounds from input.json
    with open(os.path.join(run_dir, "input.json"), "r") as f:
        input_data = json.load(f)
    y_min = input_data.get("y_min", -22.6)
    y_max = input_data.get("y_max", -18.4)

    # ── posterior draws → residuals ────────────────────────────────────────────
    if kind == "tophat":
        draws = read_cmdstan_posterior(
            os.path.join(run_dir, "tophat_?.csv"),
            keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y"],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
            draws,
            xhat,
            sigma_x,
            y_min=y_min,
            y_max=y_max,
            on_bad_Z="floor",
            Z_floor=1e-300,
        )
    else:
        draws = read_cmdstan_posterior(
            os.path.join(run_dir, "normal_?.csv"),
            keep=[
                "slope",
                "intercept.1",
                "sigma_int_x",
                "sigma_int_y",
                "mu_y_TF",
                "tau",
            ],
            drop_diagnostics=True,
        )
        mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat, sigma_x)

    mean_y = mean_pred - yhat
    main_mask = _apply_main_cuts(cfg, xhat, yhat, zobs=zobs)
    print(f"N main-sample: {main_mask.sum()}")

    # ── continuous parameter plots ─────────────────────────────────────────────
    cont_params = [
        ("ba", ba, r"$b/a$ (axis ratio)"),
        ("d26_kpc", d26_kpc, r"$D_{26}$ (kpc)"),
        ("g_r", g_r, r"$(g - r)_{\rm corr}$"),
        ("r_z", r_z, r"$(r - z)_{\rm corr}$"),
        ("g_z", g_z, r"$(g - z)_{\rm corr}$"),
        ("g_r_obs", g_r_obs, r"$(g - r)_{\rm obs}$"),
        ("r_z_obs", r_z_obs, r"$(r - z)_{\rm obs}$"),
        ("g_z_obs", g_z_obs, r"$(g - z)_{\rm obs}$"),
        ("sma_sb26", sma_sb26, r"SMA$_{\rm SB26}$ (arcsec)"),
        ("sma_ratio", sma_ratio, r"SMA$_{\rm SB26}$/SMA$_{\rm SB22}$"),
        ("g_sma50", g_sma50, r"$g$-band $r_{50}$ (arcsec)"),
        ("r_sma50", r_sma50, r"$r$-band $r_{50}$ (arcsec)"),
        ("z_sma50", z_sma50, r"$z$-band $r_{50}$ (arcsec)"),
    ]

    for name, param, xlabel in cont_params:
        # require finite param values in both samples
        ok_full = np.isfinite(param) & np.isfinite(mean_y)
        ok_main = ok_full & main_mask

        if not ok_full.any():
            print(f"Skipping {name} plot due to lack of valid data.")
            continue

        x_lo, x_hi = np.nanpercentile(param[ok_full], [1, 99])

        fig, ax = plt.subplots(figsize=(7, 4))
        _scatter_binned(
            ax, param[ok_full], mean_y[ok_full], label="Full sample", color="steelblue"
        )
        _scatter_binned(
            ax, param[ok_main], mean_y[ok_main], label="Main sample", color="tomato"
        )
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\langle\hat{y}_* - \hat{y}\rangle$ (mag)")
        ax.legend(fontsize=8)
        _save(fig, out_dir, f"resid_vs_{name}.png")

    # ── categorical: MORPHTYPE ─────────────────────────────────────────────────
    ok = np.isfinite(mean_y)
    types, counts = np.unique(morphtype[ok], return_counts=True)
    # keep types with ≥ 20 galaxies
    keep_types = types[counts >= 20]
    means_full, means_main = [], []
    stds_full, stds_main = [], []
    labels = []
    for t in keep_types:
        sel_full = ok & (morphtype == t)
        sel_main = sel_full & main_mask
        means_full.append(mean_y[sel_full].mean())
        stds_full.append(mean_y[sel_full].std())
        means_main.append(mean_y[sel_main].mean() if sel_main.sum() > 0 else np.nan)
        stds_main.append(mean_y[sel_main].std() if sel_main.sum() > 0 else np.nan)
        labels.append(t.strip())

    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5 + 1), 4))
    ax.bar(
        x_pos - 0.2,
        means_full,
        0.4,
        yerr=stds_full,
        label="Full",
        color="steelblue",
        alpha=0.8,
        capsize=3,
    )
    ax.bar(
        x_pos + 0.2,
        means_main,
        0.4,
        yerr=stds_main,
        label="Main",
        color="tomato",
        alpha=0.8,
        capsize=3,
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"Mean residual (mag)")
    ax.set_title("Residual by morphological type")
    ax.legend()
    _save(fig, out_dir, "resid_by_morphtype.png")

    # ── categorical: PHOTSYS (N vs S) ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, label, color in zip(axes, ["Full", "Main"], ["steelblue", "tomato"]):
        msk_label = ok if label == "Full" else ok & main_mask
        for sys_val, ls in [("N", "solid"), ("S", "dashed")]:
            sel = msk_label & (np.char.strip(photsys) == sys_val)
            if sel.sum() == 0:
                continue
            ax.hist(
                mean_y[sel],
                bins=40,
                histtype="step",
                linestyle=ls,
                label=f"PHOTSYS={sys_val} (N={sel.sum()})",
                density=True,
            )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(r"Residual (mag)")
        ax.set_title(label)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Density")
    fig.suptitle("Residual distribution: PHOTSYS")
    _save(fig, out_dir, "resid_by_photsys.png")

    # ── categorical: isolated (GROUP_MULT=1) vs group (GROUP_MULT>1) ──────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, sel_extra, color, ls in [
        ("Isolated (N=1)", group_mult == 1, "steelblue", "solid"),
        ("Group (N>1)", group_mult > 1, "tomato", "dashed"),
    ]:
        sel = ok & sel_extra
        if sel.sum() == 0:
            continue
        ax.hist(
            mean_y[sel],
            bins=40,
            histtype="step",
            linestyle=ls,
            color=color,
            density=True,
            label=f"{label} ({sel.sum()})",
        )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"Residual (mag)")
    ax.set_ylabel("Density")
    ax.set_title("Residual: isolated vs group galaxies (full sample)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "resid_by_group.png")

    # ── correlation summary ────────────────────────────────────────────────────
    corr_names, corr_full, corr_main = [], [], []
    for name, param, xlabel in cont_params:
        ok_full = np.isfinite(param) & np.isfinite(mean_y)
        ok_main = ok_full & main_mask
        if ok_full.sum() > 10:
            r_full, _ = pearsonr(param[ok_full], mean_y[ok_full])
        else:
            r_full = np.nan
        if ok_main.sum() > 10:
            r_main, _ = pearsonr(param[ok_main], mean_y[ok_main])
        else:
            r_main = np.nan
        corr_names.append(xlabel)
        corr_full.append(r_full)
        corr_main.append(r_main)

    y_pos = np.arange(len(corr_names))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(corr_names) + 1.5))
    ax.barh(
        y_pos - 0.18, corr_full, 0.36, label="Full sample", color="steelblue", alpha=0.8
    )
    ax.barh(
        y_pos + 0.18, corr_main, 0.36, label="Main sample", color="tomato", alpha=0.8
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_names, fontsize=9)
    ax.set_xlabel("Pearson r with residual")
    ax.set_title("Correlation summary")
    ax.legend()
    _save(fig, out_dir, "correlation_summary.png")

    print("Done.")


if __name__ == "__main__":
    main()
