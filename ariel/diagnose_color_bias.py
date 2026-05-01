"""
diagnose_color_bias.py

Systematic diagnostic for color-residual bias in TFR residuals.
Runs Phases 1-3 from the diagnostic plan.

Usage
-----
python diagnose_color_bias.py --run-dir output/DR1_v3 [--kind tophat]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from predict import (
    _apply_main_cuts,
    read_cmdstan_posterior,
    ystar_pp_mean_sd_normal_vectorized,
    ystar_pp_mean_sd_tophat_vectorized,
)
from mag_utils import get_mag_cols
from config_utils import apply_config

V0 = 100.0

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--run", default=None)
parser.add_argument("--run-dir", default=None)
parser.add_argument("--config", default=None)
parser.add_argument("--kind", default="tophat", choices=["tophat", "normal"])
args = parser.parse_args()

cfg = apply_config(args)
if args.run and not args.run_dir:
    args.run_dir = os.path.join("output", args.run)
run_dir = args.run_dir or "output/DR1_v3"
kind = args.kind
out_dir = os.path.join(run_dir, "diagnose_color_bias")
os.makedirs(out_dir, exist_ok=True)

fits_path = cfg.get("fits_file")
print(f"run_dir : {run_dir}")
print(f"kind    : {kind}")
print(f"FITS    : {fits_path}")
print(f"out_dir : {out_dir}")
print()

# ── load FITS ─────────────────────────────────────────────────────────────────

with fits.open(fits_path) as hdul:
    data = hdul[1].data
    names = set(data.dtype.names or ())

    def _get_col(col_name, dtype=float):
        if col_name in names:
            return np.asarray(data[col_name], dtype=dtype)
        else:
            print(f"  WARNING: column '{col_name}' missing, filling with NaNs")
            if dtype is str:
                return np.full(len(data), "", dtype=object)
            return np.full(len(data), np.nan, dtype=float)

    col_abs, col_abs_err, col_app = get_mag_cols(names)

    V_raw = _get_col("V_0p4R26")
    V_err_raw = _get_col("V_0p4R26_ERR")
    yhat_raw = _get_col(col_abs)
    sigma_y_raw = _get_col(col_abs_err)
    zobs_raw = _get_col("Z_DESI")

    g_mag_corr_raw = _get_col("G_MAG_SB26_CORR")
    r_mag_corr_raw = _get_col(col_app)  # R_MAG_SB26_CORR
    z_mag_corr_raw = _get_col("Z_MAG_SB26_CORR")
    g_mag_raw_raw = _get_col("G_MAG_SB26")
    r_mag_raw_raw = _get_col("R_MAG_SB26")
    z_mag_raw_raw = _get_col("Z_MAG_SB26")  # may be missing
    yhat_uncorr_raw = _get_col("R_ABSMAG_SB26")  # uncorrected abs mag (may be missing)

    morphtype_raw = _get_col("MORPHTYPE", dtype=str)
    photsys_raw = _get_col("PHOTSYS", dtype=str)
    group_mult_raw = _get_col("GROUP_MULT")

# ── PHASE 1a: column statistics ───────────────────────────────────────────────

print("=" * 60)
print("PHASE 1a: Column statistics")
print("=" * 60)

check_cols = {
    "G_MAG_SB26_CORR": g_mag_corr_raw,
    "R_MAG_SB26_CORR": r_mag_corr_raw,
    "Z_MAG_SB26_CORR": z_mag_corr_raw,
    "G_MAG_SB26": g_mag_raw_raw,
    "R_MAG_SB26": r_mag_raw_raw,
    "Z_MAG_SB26": z_mag_raw_raw,
    "R_ABSMAG_SB26_CORR": yhat_raw,
    "R_ABSMAG_SB26": yhat_uncorr_raw,
}

for cname, vals in check_cols.items():
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        print(f"  {cname:30s}: ALL NaN / missing")
    else:
        print(
            f"  {cname:30s}: n={len(finite):6d}, mean={finite.mean():8.3f}, "
            f"std={finite.std():6.3f}, min={finite.min():8.3f}, max={finite.max():8.3f}"
        )

# ── validity mask ─────────────────────────────────────────────────────────────

mask = (
    np.isfinite(V_raw)
    & np.isfinite(V_err_raw)
    & np.isfinite(yhat_raw)
    & np.isfinite(sigma_y_raw)
    & np.isfinite(zobs_raw)
    & (V_raw > 0)
    & (V_err_raw > 0)
    & (sigma_y_raw >= 0)
)


def _m(arr):
    return arr[mask]


xhat = np.log10(_m(V_raw) / V0)
sigma_x = _m(V_err_raw) / (_m(V_raw) * np.log(10))
yhat = _m(yhat_raw)
sigma_y = _m(sigma_y_raw)
zobs = _m(zobs_raw)

g_corr = _m(g_mag_corr_raw)
r_corr = _m(r_mag_corr_raw)
z_corr = _m(z_mag_corr_raw)
g_raw = _m(g_mag_raw_raw)
r_raw = _m(r_mag_raw_raw)
z_raw = _m(z_mag_raw_raw)
yhat_unc = _m(yhat_uncorr_raw)
morphtype = _m(morphtype_raw)
photsys = _m(photsys_raw)
group_mult = _m(group_mult_raw)

print(f"\n  N galaxies (valid mask): {mask.sum()}")

# ── PHASE 1b: dust correction delta ──────────────────────────────────────────

print()
print("=" * 60)
print("PHASE 1b: Dust correction delta")
print("=" * 60)

delta_g = g_corr - g_raw
delta_r = r_corr - r_raw
delta_z = z_corr - z_raw
delta_color_gr = (g_corr - r_corr) - (g_raw - r_raw)  # effect on g-r
color_gr_corr = g_corr - r_corr
color_gr_raw = g_raw - r_raw

ok_dust = np.isfinite(delta_g) & np.isfinite(delta_r) & np.isfinite(color_gr_corr)

print(
    f"\n  delta_g (dust corr in g): mean={delta_g[ok_dust].mean():.4f}, std={delta_g[ok_dust].std():.4f}"
)
print(
    f"  delta_r (dust corr in r): mean={delta_r[ok_dust].mean():.4f}, std={delta_r[ok_dust].std():.4f}"
)
print(
    f"  delta_color (effect on g-r): mean={delta_color_gr[ok_dust].mean():.4f}, std={delta_color_gr[ok_dust].std():.4f}"
)

r_dust_gr, p_dust_gr = pearsonr(color_gr_corr[ok_dust], delta_color_gr[ok_dust])
print(f"\n  Pearson r(g-r_corr, delta_color_gr) = {r_dust_gr:.4f}  (p={p_dust_gr:.2e})")
print(
    f"  => {'SIGNIFICANT: dust correction is color-dependent' if abs(r_dust_gr) > 0.2 else 'Low correlation — dust correction not strongly color-dependent'}"
)

# Also check if raw vs corrected g-r differ significantly
ok_both = np.isfinite(color_gr_corr) & np.isfinite(color_gr_raw)
diff_gr = color_gr_corr[ok_both] - color_gr_raw[ok_both]
print(
    f"\n  Corrected g-r vs raw g-r shift: mean={diff_gr.mean():.4f}, std={diff_gr.std():.4f}"
)
print(f"  (positive = dust correction makes galaxies appear bluer in g-r)")

# scatter plot: delta_color vs g-r_corr
fig, ax = plt.subplots(figsize=(6, 4))
ok_plot = ok_dust & (np.abs(delta_color_gr) < 2)
ax.scatter(
    color_gr_corr[ok_plot], delta_color_gr[ok_plot], alpha=0.05, s=2, color="steelblue"
)
# binned
order = np.argsort(color_gr_corr[ok_plot])
xs = color_gr_corr[ok_plot][order]
ys = delta_color_gr[ok_plot][order]
bins = np.array_split(np.arange(len(xs)), 20)
bx = [xs[b].mean() for b in bins if len(b) > 0]
by = [ys[b].mean() for b in bins if len(b) > 0]
ax.plot(bx, by, "r-o", markersize=4, lw=1.5, label=f"binned mean (r={r_dust_gr:.3f})")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("g-r (corrected)")
ax.set_ylabel("delta_color = (g-r)_corr - (g-r)_raw")
ax.set_title("Dust correction effect on g-r color")
ax.legend(fontsize=8)
fig.savefig(
    os.path.join(out_dir, "phase1b_dust_delta_color.png"), dpi=150, bbox_inches="tight"
)
plt.close(fig)
print(f"\n  Saved: phase1b_dust_delta_color.png")

# ── load posterior → residuals ────────────────────────────────────────────────

with open(os.path.join(run_dir, "input.json"), "r") as f:
    input_data = json.load(f)
y_min = input_data.get("y_min", -22.6)
y_max = input_data.get("y_max", -18.4)

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
        keep=["slope", "intercept.1", "sigma_int_x", "sigma_int_y", "mu_y_TF", "tau"],
        drop_diagnostics=True,
    )
    mean_pred, sd_pred = ystar_pp_mean_sd_normal_vectorized(draws, xhat, sigma_x)

mean_y = mean_pred - yhat
main_mask = _apply_main_cuts(cfg, xhat, yhat, zobs=zobs)
print(f"\n  N main-sample: {main_mask.sum()}")

g_r = g_corr - r_corr
r_z = r_corr - z_corr

# ── PHASE 1c: index alignment check ──────────────────────────────────────────

print()
print("=" * 60)
print("PHASE 1c: Index alignment check")
print("=" * 60)
print("\n  First 5 main-sample galaxies (xhat, yhat, g-r, mean_y, z_desi, V):")
idx = np.where(main_mask)[0][:5]
for i in idx:
    print(
        f"    [{i:5d}] xhat={xhat[i]:.4f} yhat={yhat[i]:.3f} g-r={g_r[i]:.3f} "
        f"mean_y={mean_y[i]:.3f} z={zobs[i]:.4f} V={10 ** (xhat[i]) * V0:.1f}"
    )

# Check: are there duplicate (xhat, yhat) pairs that could indicate misalignment?
n_unique = len(set(zip(np.round(xhat, 5), np.round(yhat, 5))))
print(f"\n  N unique (xhat, yhat) pairs: {n_unique} / {len(xhat)} total")
print(
    f"  => {'OK: all unique' if n_unique == len(xhat) else 'WARNING: duplicates detected'}"
)

# Define color iterations for Phase 2 and 3a
color_defs = [
    ("corr", g_corr - r_corr, r"$(g-r)_{\rm corr}$"),
    ("obs", g_raw - r_raw, r"$(g-r)_{\rm obs}$"),
]

for color_kind, g_r_current, color_label in color_defs:
    # Check g-r range is sensible
    ok_color = np.isfinite(g_r_current)
    print(
        f"\n  {color_label} range (valid): [{g_r_current[ok_color].min():.3f}, {g_r_current[ok_color].max():.3f}]"
    )
    print(f"  {color_label} median: {np.median(g_r_current[ok_color]):.3f}")

    # ── PHASE 2a: split by photometric system ─────────────────────────────────────

    print()
    print("=" * 60)
    print(f"PHASE 2a: Color-residual correlation split by photometric system [{color_kind}]")
    print("=" * 60)

    ok_full = np.isfinite(g_r_current) & np.isfinite(mean_y)
    ok_main = ok_full & main_mask

    for sys_val in ['N', 'S']:
        m = ok_full & (np.char.strip(photsys) == sys_val)
        r_f, p_f = (pearsonr(g_r_current[m], mean_y[m]) if m.sum() > 20 else (np.nan, np.nan))
        
        ok_m = ok_main & (np.char.strip(photsys) == sys_val)
        r_m, p_m = (pearsonr(g_r_current[ok_m], mean_y[ok_m]) if ok_m.sum() > 20 else (np.nan, np.nan))
        print(f"\n  PHOTSYS={sys_val}:")
        print(f"    Full: r={r_f:.4f}  p={p_f:.2e}  n={m.sum()}")
        print(f"    Main: r={r_m:.4f}  p={p_m:.2e}  n={ok_m.sum()}")

    # Plot: color-residual by photsys
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, label, msk_base in zip(axes, ['Full', 'Main'], [ok_full, ok_main]):
        for sys_val, color, ls in [('N', 'steelblue', '-'), ('S', 'tomato', '--')]:
            sys_mask = (np.char.strip(photsys) == sys_val)
            sel = msk_base & sys_mask
            if sel.sum() < 10:
                continue
            ok_s = np.isfinite(g_r_current[sel]) & np.isfinite(mean_y[sel])
            if ok_s.sum() < 10:
                continue
            xs, ys = g_r_current[sel][ok_s], mean_y[sel][ok_s]
            
            # binned
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            bins = np.array_split(np.arange(len(xs)), 15)
            bx = [xs[b].mean() for b in bins if len(b) > 0]
            by = [ys[b].mean() for b in bins if len(b) > 0]
            r_s, _ = pearsonr(xs, ys)
            ax.scatter(xs, ys, alpha=0.05, s=2, color=color)
            ax.plot(bx, by, color=color, ls=ls, lw=1.5, marker='o', markersize=3,
                    label=f'PHOTSYS={sys_val} (r={r_s:.3f}, n={sel.sum()})')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(color_label)
        ax.set_ylabel('Residual (mean_pred - yhat)')
        ax.set_title(f'{label} sample')
        ax.legend(fontsize=7)
    fig.suptitle(f'Phase 2a: Color-residual by photometric system [{color_kind}]')
    fig.savefig(os.path.join(out_dir, f'phase2a_photsys_{color_kind}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: phase2a_photsys_{color_kind}.png")

    # ── PHASE 2b: redshift slices ─────────────────────────────────────────────────

    print()
    print("=" * 60)
    print(f"PHASE 2b: Color-residual correlation in narrow redshift slices [{color_kind}]")
    print("=" * 60)

    z_bins = [(0.0, 0.04), (0.04, 0.08), (0.08, 1.0)]

    for zlo, zhi in z_bins:
        for msk, label in [(ok_full, 'Full'), (ok_main, 'Main')]:
            m = msk & (zobs > zlo) & (zobs <= zhi) & np.isfinite(g_r_current)
            if m.sum() < 10:
                continue
            r_v, p_v = pearsonr(g_r_current[m], mean_y[m])
            print(f"  z=[{zlo:.3f},{zhi:.3f}]  {label}: r={r_v:.4f}  p={p_v:.2e}  n={m.sum()}")

    # Plot
    fig, axes = plt.subplots(1, len(z_bins), figsize=(5*len(z_bins), 4), sharey=True)
    colors = ['steelblue', 'forestgreen', 'tomato']
    for i, (zlo, zhi) in enumerate(z_bins):
        ax = axes[i]
        m = ok_main & (zobs > zlo) & (zobs <= zhi)
        if m.sum() < 10:
            ax.set_title(f'z=[{zlo:.3f},{zhi:.3f}] n<10')
            continue
        
        xs, ys = g_r_current[m], mean_y[m]
        order = np.argsort(xs)
        xs2, ys2 = xs[order], ys[order]
        bins = np.array_split(np.arange(len(xs2)), 12)
        bx = [xs2[b].mean() for b in bins if len(b) > 0]
        by = [ys2[b].mean() for b in bins if len(b) > 0]
        r_v, _ = pearsonr(xs, ys)
        ax.scatter(xs, ys, alpha=0.1, s=3, color=colors[i])
        ax.plot(bx, by, color=colors[i], lw=1.5, marker='o', markersize=3)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(color_label)
        ax.set_title(f'z=[{zlo:.3f},{zhi:.3f}]\nn={m.sum()}  r={r_v:.3f}')
    axes[0].set_ylabel('Residual (mag)')
    fig.suptitle(f'Phase 2b: Color-residual slope vs redshift (main sample) [{color_kind}]')
    fig.savefig(os.path.join(out_dir, f'phase2b_zslices_{color_kind}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: phase2b_zslices_{color_kind}.png")

    # ── PHASE 2c: red vs blue TFR slope comparison ────────────────────────────────

    print()
    print("=" * 60)
    print(f"PHASE 2c: TFR slope — red vs blue color quartiles (main sample) [{color_kind}]")
    print("=" * 60)

    ok_m = ok_main & np.isfinite(g_r_current)
    q25, q75 = np.nanpercentile(g_r_current[ok_m], [25, 75])
    blue_mask = ok_m & (g_r_current < q25)
    red_mask  = ok_m & (g_r_current > q75)

    print(f"\n  Blue quartile threshold ({color_label} < {q25:.3f}): n={blue_mask.sum()}")
    print(f"  Red quartile threshold  ({color_label} > {q75:.3f}): n={red_mask.sum()}")

    def _get_slope(x, y):
        if len(x) < 5: return np.nan
        return np.polyfit(x, y, 1)[0]

    s_blue = _get_slope(xhat[blue_mask], mean_y[blue_mask])
    s_red  = _get_slope(xhat[red_mask],  mean_y[red_mask])
    print(f"\n  Residual slope vs xhat (blue): {s_blue:+.4f}")
    print(f"  Residual slope vs xhat (red):  {s_red:+.4f}")
    print(f"  => Difference: {s_red - s_blue:+.4f}")

    # Fit simple slope for each
    from numpy.polynomial import polynomial as P

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, m, color in [(f'Blue ({color_label}<Q25)', blue_mask, 'steelblue'), (f'Red ({color_label}>Q75)', red_mask, 'tomato')]:
        ax.scatter(xhat[m], mean_y[m], alpha=0.1, s=3, color=color)
        # binned
        order = np.argsort(xhat[m])
        xs = xhat[m][order]
        ys = mean_y[m][order]
        bins = np.array_split(np.arange(len(xs)), 12)
        bx = [xs[b].mean() for b in bins if len(b) > 0]
        by = [ys[b].mean() for b in bins if len(b) > 0]
        ax.plot(bx, by, color=color, lw=1.5, marker='o', markersize=3, label=label)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$\hat{x} = \log_{10}(V/100)$')
    ax.set_ylabel('Residual (mag)')
    ax.set_title(f'Phase 2c: Residual vs velocity — red vs blue (main sample) [{color_kind}]')
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, f'phase2c_red_blue_xhat_{color_kind}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: phase2c_red_blue_xhat_{color_kind}.png")

    # ── PHASE 2d: Residual vs observed absolute magnitude ────────────────────────────────────

    print()
    print("=" * 60)
    print(f"PHASE 2d: Residual vs observed absolute magnitude [{color_kind}]")
    print("=" * 60)

    ok_m = ok_main & np.isfinite(g_r_current)
    r_yhat, p_yhat = pearsonr(yhat[ok_m], mean_y[ok_m])
    print(f"\n  r(yhat, residual) = {r_yhat:.4f}  p={p_yhat:.2e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(yhat[ok_m], mean_y[ok_m], c=g_r_current[ok_m], cmap='coolwarm', alpha=0.5, s=6)
    # binned mean
    order = np.argsort(yhat[ok_m])
    xs = yhat[ok_m][order]
    ys = mean_y[ok_m][order]
    bins = np.array_split(np.arange(len(xs)), 15)
    bx = [xs[b].mean() for b in bins if len(b) > 0]
    by = [ys[b].mean() for b in bins if len(b) > 0]
    ax.plot(bx, by, 'k-o', lw=1.5, markersize=3, label=f'binned mean (r={r_yhat:.3f})')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    plt.colorbar(sc, ax=ax, label=color_label)
    ax.set_xlabel(r'$\hat{y}$ = observed abs magnitude')
    ax.set_ylabel('Residual (mag)')
    ax.set_title(f'Phase 2d: Residual vs absolute magnitude, colored by color (main sample) [{color_kind}]')
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, f'phase2d_resid_vs_yhat_{color_kind}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: phase2d_resid_vs_yhat_{color_kind}.png")

    # ── PHASE 3a: morphological type ──────────────────────────────────────────────

    print()
    print("=" * 60)
    print(f"PHASE 3a: Color and residual by morphological type (main sample) [{color_kind}]")
    print("=" * 60)

    ok_m = ok_main & np.isfinite(mean_y)
    morph_types, morph_counts = np.unique(morphtype[ok_m], return_counts=True)

    keep = [t for t, c in zip(morph_types, morph_counts) if c >= 5 and str(t).strip()]
    print(f"\n  Morphological types with N>=5:")
    morph_data = []
    for t in keep:
        sel = ok_m & (morphtype == t)
        if not np.isfinite(g_r_current[sel]).any():
            continue
        mean_r = mean_y[sel].mean()
        mean_color = np.nanmean(g_r_current[sel])
        n = sel.sum()
        morph_data.append((t.strip(), mean_r, mean_color, n))
        print(f"  {t.strip():12s}  n={n:4d}  mean_resid={mean_r:+.4f}  mean_color={mean_color:.3f}")

    # Scatter: mean_resid vs mean_color per morphtype
    if morph_data:
        labels_m, resids_m, colors_m, ns_m = zip(*morph_data)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(colors_m, resids_m, c=range(len(labels_m)), cmap='tab20', s=80, zorder=3)
        for l, x, y in zip(labels_m, colors_m, resids_m):
            ax.annotate(l, (x, y), fontsize=7, ha='left', va='bottom')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel(f'Mean {color_label} per morphtype')
        ax.set_ylabel('Mean residual per morphtype')
        ax.set_title(f'Phase 3a: Mean residual vs mean color by morphtype (main sample) [{color_kind}]')
        fig.savefig(os.path.join(out_dir, f'phase3a_morphtype_{color_kind}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved: phase3a_morphtype_{color_kind}.png")

# Notice: Loop ends here. Phase 3b and onwards evaluate against `g_r = g_corr - r_corr`.



# ── PHASE 3b: corrected vs uncorrected abs magnitude ─────────────────────────

print()
print("=" * 60)
print("PHASE 3b: Residual bias with uncorrected absolute magnitude")
print("=" * 60)

has_uncorr = np.isfinite(yhat_unc).any()
if not has_uncorr:
    print("  R_ABSMAG_SB26 column is all NaN — skipping")
else:
    # Recompute residual using uncorrected yhat
    mean_y_unc = mean_pred - yhat_unc
    ok_unc = (
        main_mask & np.isfinite(yhat_unc) & np.isfinite(mean_y_unc) & np.isfinite(g_r)
    )
    if ok_unc.sum() > 20:
        r_unc, p_unc = pearsonr(g_r[ok_unc], mean_y_unc[ok_unc])
        r_cor, p_cor = pearsonr(g_r[ok_unc], mean_y[ok_unc])
        print(
            f"\n  Using CORRECTED yhat:   r(g-r, resid) = {r_cor:.4f}  p={p_cor:.2e}  n={ok_unc.sum()}"
        )
        print(f"  Using UNCORRECTED yhat: r(g-r, resid) = {r_unc:.4f}  p={p_unc:.2e}")
        print(
            f"\n  => {'Correction REDUCES bias (working correctly)' if abs(r_unc) > abs(r_cor) else 'Correction INTRODUCES or WORSENS bias'}"
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, ys, label, r_v in [
            (axes[0], mean_y[ok_unc], f"Corrected yhat (r={r_cor:.3f})", r_cor),
            (axes[1], mean_y_unc[ok_unc], f"Uncorrected yhat (r={r_unc:.3f})", r_unc),
        ]:
            xs = g_r[ok_unc]
            ax.scatter(xs, ys, alpha=0.1, s=3, color="steelblue")
            order = np.argsort(xs)
            xs2, ys2 = xs[order], ys[order]
            bins = np.array_split(np.arange(len(xs2)), 15)
            bx = [xs2[b].mean() for b in bins if len(b) > 0]
            by = [ys2[b].mean() for b in bins if len(b) > 0]
            ax.plot(bx, by, "r-o", lw=1.5, markersize=3)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("g - r")
            ax.set_ylabel("Residual (mag)")
            ax.set_title(label)
        fig.suptitle("Phase 3b: Corrected vs uncorrected yhat (main sample)")
        fig.savefig(
            os.path.join(out_dir, "phase3b_corr_vs_uncorr.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"\n  Saved: phase3b_corr_vs_uncorr.png")
    else:
        print(
            f"  Too few galaxies with both corrected and uncorrected magnitudes ({ok_unc.sum()})"
        )

# ── PHASE 3c: extreme outliers ────────────────────────────────────────────────

print()
print("=" * 60)
print("PHASE 3c: Extreme residual outliers (main sample)")
print("=" * 60)

ok_m_idx = np.where(ok_main & np.isfinite(mean_y))[0]
sorted_by_resid = ok_m_idx[np.argsort(mean_y[ok_m_idx])]
top_neg = sorted_by_resid[:10]
top_pos = sorted_by_resid[-10:]

print("\n  10 most NEGATIVE residuals (model predicts much fainter than observed):")
print(
    f"  {'idx':>6}  {'yhat':>7}  {'mean_pred':>9}  {'resid':>7}  {'g-r':>6}  {'z':>7}  {'V(km/s)':>8}  {'morph':>12}  {'photsys':>7}  {'group':>5}"
)
for i in top_neg:
    V_val = 10 ** (xhat[i]) * V0
    print(
        f"  {i:6d}  {yhat[i]:7.3f}  {mean_pred[i]:9.3f}  {mean_y[i]:7.3f}  "
        f"{g_r[i]:6.3f}  {zobs[i]:7.4f}  {V_val:8.1f}  "
        f"{morphtype[i].strip():>12}  {photsys[i].strip():>7}  {int(group_mult[i]):>5}"
    )

print("\n  10 most POSITIVE residuals (model predicts much brighter than observed):")
print(
    f"  {'idx':>6}  {'yhat':>7}  {'mean_pred':>9}  {'resid':>7}  {'g-r':>6}  {'z':>7}  {'V(km/s)':>8}  {'morph':>12}  {'photsys':>7}  {'group':>5}"
)
for i in top_pos:
    V_val = 10 ** (xhat[i]) * V0
    print(
        f"  {i:6d}  {yhat[i]:7.3f}  {mean_pred[i]:9.3f}  {mean_y[i]:7.3f}  "
        f"{g_r[i]:6.3f}  {zobs[i]:7.4f}  {V_val:8.1f}  "
        f"{morphtype[i].strip():>12}  {photsys[i].strip():>7}  {int(group_mult[i]):>5}"
    )

# Fraction of outliers in groups
neg_group = np.mean(group_mult[top_neg] > 1)
pos_group = np.mean(group_mult[top_pos] > 1)
print(
    f"\n  Fraction in groups: negative outliers={neg_group:.0%}, positive outliers={pos_group:.0%}"
)
print(f"  Overall main-sample group fraction: {(group_mult[ok_main] > 1).mean():.0%}")

# ── PHASE 3 bonus: partial correlations ──────────────────────────────────────

print()
print("=" * 60)
print("BONUS: Partial correlation — does color predict residual beyond yhat?")
print("=" * 60)

ok_m = ok_main & np.isfinite(g_r) & np.isfinite(mean_y)

# Regress out yhat from residual
from numpy.linalg import lstsq

X_yhat = np.column_stack([np.ones(ok_m.sum()), yhat[ok_m]])
resid_resid_yhat = mean_y[ok_m] - X_yhat @ lstsq(X_yhat, mean_y[ok_m], rcond=None)[0]

r_partial, p_partial = pearsonr(g_r[ok_m], resid_resid_yhat)
print(f"\n  r(g-r, resid | yhat) = {r_partial:.4f}  p={p_partial:.2e}")
print(
    f"  => Color {'STILL predicts residual after removing yhat trend' if abs(r_partial) > 0.05 else 'NOT significant after removing yhat trend'}"
)

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
  Phase 1a: Column stats — check output above
  Phase 1b: Dust correction Pearson r(g-r, delta_color) = {r_dust_gr:.4f}
  Phase 1c: Index alignment check — see sample rows above
  Phase 2a: N/S photsys split — see phase2a_photsys.png
  Phase 2b: Redshift slices — see phase2b_zslices.png
  Phase 2c: Red vs blue TFR — see phase2c_red_blue_xhat.png
  Phase 2d: Residual vs yhat r = {r_yhat:.4f} — see phase2d_resid_vs_yhat.png
  Phase 3a: Morphology-color-residual — see phase3a_morphtype.png
  Phase 3b: Corrected vs uncorrected — see phase3b_corr_vs_uncorr.png (if available)
  Phase 3c: Outlier inspection — see above
  Bonus:    Partial r(g-r | yhat) = {r_partial:.4f}

  All plots saved to: {out_dir}
""")
