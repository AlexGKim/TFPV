#!/usr/bin/env python3
"""
ellipse_sweep.py — Selection-cut optimisation and parametric sweep for TFR fitting.

Three operating modes, selected by CLI flags:

─── Default mode (no flag) ───────────────────────────────────────────────────
Two-phase 2D fiducial search over (n_σ_perp, n_σ_mag), followed by a 1D
parametric sweep of each cut parameter.

  Phase 1 (coarse): evaluates MLE slope on an n_σ_perp × n_σ_mag grid.
  Phase 2 (fine):   zooms into the transition band identified in phase 1.

  The fiducial point is the highest-N grid point with |MLE slope − GMM slope|
  ≤ slope_tol, contracted slightly inward.

  Outputs:
    output/<run>/fiducial_search.png   — 2×2 coarse/fine heatmaps
    output/<run>/ellipse_sweep.json    — 1D sweep slopes and ∂s/∂(n_σ)
    output/<run>/ellipse_sweep.png     — 1D slope profiles and derivatives

─── --coarse_only ────────────────────────────────────────────────────────────
Runs only Phase 1 (coarse grid); skips Phase 2, the 1D sweep, and the
summary table.  Produces a 1×2 heatmap instead of 2×2.

  Output:
    output/<run>/fiducial_search.png   — 1×2 coarse heatmap

─── --mag_split ──────────────────────────────────────────────────────────────
2D grid over (n_σ_ŷmin, n_σ_ŷmax) with haty_min and haty_max scaled
independently, repeated for each n_σ_perp in --n_sigma_perp_min/max/n.
Results are saved to JSON for fast replotting.

  Outputs:
    output/<run>/mag_split_grid.json   — grid results
    output/<run>/mag_split_grid.png    — (n_perp × 2) heatmap

─── --mag_split_plot ─────────────────────────────────────────────────────────
Replot mag_split_grid.png from a previously saved mag_split_grid.json
without rerunning any Stan calls.

Requires output/<run>/selection_ellipse.json (produced by selection_ellipse.py).

Usage:
  # Default: fiducial search + 1D sweep
  python ellipse_sweep.py --source DESI --fits_file data/... --run DESI

  # Coarse-only fiducial search
  python ellipse_sweep.py --source DESI --fits_file data/... --run DESI \\
      --coarse_only --n_sigma_perp_min 3 --n_sigma_perp_max 9 --n_sigma_perp_n 4 \\
      --n_sigma_mag_min 2 --n_sigma_mag_max 6 --n_sigma_mag_n 3

  # Mag-split grid
  python ellipse_sweep.py --source DESI --fits_file data/... --run DESI \\
      --mag_split \\
      --n_sigma_perp_min 2 --n_sigma_perp_max 10 --n_sigma_perp_n 5 \\
      --n_sigma_mag_lo_min 1 --n_sigma_mag_lo_max 5 --n_sigma_mag_lo_n 3 \\
      --n_sigma_mag_hi_min 1 --n_sigma_mag_hi_max 5 --n_sigma_mag_hi_n 3

  # Replot mag-split from saved JSON
  python ellipse_sweep.py --source DESI --fits_file data/... --run DESI \\
      --mag_split_plot
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_fullmocks(fits_file):
    """Load raw (uncut, MAIN=True) galaxy data from an AbacusSummit FITS file.

    Returns a dict with keys: x, sigma_x, y, sigma_y, z_obs  (1-D float arrays).
    x = log10(V_rot / 100 km/s) = LOGVROT − 2.
    """
    print(f"Loading fullmocks FITS: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        main_mask = np.asarray(data["MAIN"], dtype=bool)
        d = data[main_mask]

    logvrot     = np.asarray(d["LOGVROT"],           dtype=float)
    logvrot_err = np.asarray(d["LOGVROT_ERR"],       dtype=float)
    absmag      = np.asarray(d["R_ABSMAG_SB26"],     dtype=float)
    absmag_err  = np.asarray(d["R_ABSMAG_SB26_ERR"], dtype=float)
    zobs        = np.asarray(d["ZOBS"],              dtype=float)

    x_raw = logvrot - 2.0
    valid = (
        np.isfinite(x_raw) & np.isfinite(logvrot_err)
        & np.isfinite(absmag) & np.isfinite(absmag_err)
        & np.isfinite(zobs)
        & (logvrot > 0) & (logvrot_err > 0) & (absmag_err >= 0)
    )
    print(f"  MAIN=True: {int(main_mask.sum())}  |  valid: {int(valid.sum())}")
    return dict(
        x       = x_raw[valid],
        sigma_x = logvrot_err[valid],
        y       = absmag[valid],
        sigma_y = absmag_err[valid],
        z_obs   = zobs[valid],
    )


def load_desi(fits_file="data/DESI-DR1_TF_pv_cat_v15.fits"):
    """Load raw DESI TF galaxy data from FITS file.

    Columns: V_0p4R26, V_0p4R26_ERR, R_ABSMAG_SB26, R_ABSMAG_SB26_ERR, ZOBS.
    x = log10(V_0p4R26 / 100 km/s).
    """
    V0 = 100.0
    print(f"Loading DESI FITS: {fits_file}")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data

    V     = np.asarray(data["V_0p4R26"],         dtype=float)
    V_err = np.asarray(data["V_0p4R26_ERR"],     dtype=float)
    mag   = np.asarray(data["R_ABSMAG_SB26"],    dtype=float)
    mag_e = None
    for col in ("R_ABSMAG_SB26_ERR", "R_MAG_SB26_ERR_CORR", "R_MAG_SB26_ERR"):
        if col in data.names:
            mag_e = np.asarray(data[col], dtype=float)
            break
    if mag_e is None:
        raise KeyError("No magnitude error column found; tried "
                       "R_ABSMAG_SB26_ERR, R_MAG_SB26_ERR_CORR, R_MAG_SB26_ERR")

    # Redshift — try several candidate column names
    zobs = None
    for col in ("Z_DESI", "Z_DESI_CMB", "ZOBS", "Z_OBS", "zobs", "Z", "ZHELIO"):
        if col in data.names:
            zobs = np.asarray(data[col], dtype=float)
            break
    if zobs is None:
        zobs = np.zeros(len(V))

    valid = (
        np.isfinite(V) & np.isfinite(V_err)
        & np.isfinite(mag) & np.isfinite(mag_e)
        & (V > 0) & (V_err > 0) & (mag_e >= 0)
    )
    x     = np.log10(V[valid] / V0)
    sx    = V_err[valid] / (V[valid] * np.log(10.0))
    print(f"  Valid rows: {int(valid.sum())}")
    return dict(
        x       = x,
        sigma_x = sx,
        y       = mag[valid],
        sigma_y = mag_e[valid],
        z_obs   = zobs[valid],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SELECTION CUTS
# ─────────────────────────────────────────────────────────────────────────────

def apply_cuts(raw_data, cuts):
    """Apply selection cuts and return filtered (x, sigma_x, y, sigma_y) arrays.

    cuts is a dict with keys:
      haty_max, haty_min           — magnitude window
      slope_plane, intercept_plane — lower oblique boundary (required)
      intercept_plane2             — upper oblique boundary (optional)
      z_obs_min, z_obs_max         — redshift window (optional)
    """
    x   = raw_data["x"]
    sx  = raw_data["sigma_x"]
    y   = raw_data["y"]
    sy  = raw_data["sigma_y"]
    z   = raw_data.get("z_obs", np.zeros(len(x)))

    mask = (y > cuts["haty_min"]) & (y < cuts["haty_max"])

    z_obs_min = cuts.get("z_obs_min")
    z_obs_max = cuts.get("z_obs_max")
    if z_obs_min is not None:
        mask &= z > z_obs_min
    if z_obs_max is not None:
        mask &= z <= z_obs_max

    sp  = cuts.get("slope_plane")
    ip  = cuts.get("intercept_plane")
    ip2 = cuts.get("intercept_plane2")
    if sp is not None and ip is not None:
        lb = np.maximum(cuts["haty_min"], sp * x + ip)
        mask &= y >= lb
        if ip2 is not None:
            ub = np.minimum(cuts["haty_max"], sp * x + ip2)
            mask &= y <= ub

    return x[mask], sx[mask], y[mask], sy[mask]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ELLIPSE GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _cuts_at_nsigma(mu, sigma, n_sigma):
    """Compute selection cut parameters for a GMM ellipse scaled by n_sigma.

    Returns a dict with keys: haty_min, haty_max, slope_plane,
    intercept_plane, intercept_plane2.  slope_plane is fixed (the ellipse
    orientation does not change with n_sigma).
    """
    vals, vecs = np.linalg.eigh(sigma)
    sigma_minor = np.sqrt(vals[0])
    sigma_major = np.sqrt(vals[1])
    angle_rad   = np.arctan2(vecs[1, -1], vecs[0, -1])

    y_extent = np.sqrt(sigma_major**2 * np.sin(angle_rad)**2
                       + sigma_minor**2 * np.cos(angle_rad)**2)
    haty_min = float(mu[1] - n_sigma * y_extent)
    haty_max = float(mu[1] + n_sigma * y_extent)
    slope    = float(np.tan(angle_rad))

    minor_vec = vecs[:, 0]
    p1 = mu + n_sigma * sigma_minor * minor_vec
    p2 = mu - n_sigma * sigma_minor * minor_vec
    i1 = float(p1[1] - slope * p1[0])
    i2 = float(p2[1] - slope * p2[0])

    return dict(
        haty_min=haty_min,
        haty_max=haty_max,
        slope_plane=slope,
        intercept_plane=min(i1, i2),
        intercept_plane2=max(i1, i2),
    )


def _cuts_mag_asymmetric(mu, sigma, n_sigma_perp, n_sigma_lo, n_sigma_hi):
    """Cuts with independent n_σ scaling for haty_min and haty_max.

    haty_min = mu[1] - n_sigma_lo * y_extent
    haty_max = mu[1] + n_sigma_hi * y_extent
    intercept_plane, intercept_plane2, slope_plane → from n_sigma_perp
    """
    perp     = _cuts_at_nsigma(mu, sigma, n_sigma_perp)
    unit     = _cuts_at_nsigma(mu, sigma, 1.0)
    y_extent = unit["haty_max"] - float(mu[1])
    return dict(
        haty_min        = float(mu[1]) - n_sigma_lo * y_extent,
        haty_max        = float(mu[1]) + n_sigma_hi * y_extent,
        slope_plane     = perp["slope_plane"],
        intercept_plane = perp["intercept_plane"],
        intercept_plane2= perp["intercept_plane2"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: STAN INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

_N_MIN = 30   # minimum galaxies to attempt a fit


def _build_stan_dicts(raw_data, cuts):
    """Apply cuts and build Stan data/init dicts in memory.

    Returns (data_dict, init_dict), or (None, None) if the sample is too
    small or the cuts are geometrically invalid.
    """
    haty_min = cuts["haty_min"]
    haty_max = cuts["haty_max"]
    ip       = cuts.get("intercept_plane")
    ip2      = cuts.get("intercept_plane2")

    if haty_min >= haty_max:
        return None, None
    if ip is not None and ip2 is not None and ip >= ip2:
        return None, None

    x, sx, y, sy = apply_cuts(raw_data, cuts)
    if len(x) < _N_MIN:
        return None, None

    mean_x = float(np.mean(x))
    sd_x   = float(np.std(x, ddof=1))
    if sd_x < 1e-6:
        return None, None

    x_std = (x - mean_x) / sd_x
    slope_std, intercept_std = np.polyfit(x_std, y, 1)
    # Clamp to Stan model's parameter bounds: slope_std ∈ (−9·sd_x, −4·sd_x)
    slope_std = float(np.clip(slope_std, -9.0 * sd_x + 1e-4, -4.0 * sd_x - 1e-4))

    data_dict = {
        "N_bins":           1,
        "N_total":          len(x),
        "x":                x.tolist(),
        "sigma_x":          sx.tolist(),
        "y":                y.tolist(),
        "sigma_y":          sy.tolist(),
        "haty_min":         float(haty_min),
        "haty_max":         float(haty_max),
        "y_min":            float(haty_min) - 0.5,
        "y_max":            float(haty_max) + 1.0,
        "slope_plane":      float(cuts["slope_plane"]),
        "intercept_plane":  float(ip),
        "intercept_plane2": float(ip2),
    }
    init_dict = {
        "slope_std":     float(slope_std),
        "intercept_std": [float(intercept_std)],
        "sigma_int_x":   0.1,
        "sigma_int_y":   0.1,
    }
    return data_dict, init_dict


def _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir):
    """Run Stan optimize and return the MLE slope, or None on failure.

    Writes JSON files to tmp_dir, invokes exe_file directly via subprocess,
    and parses the output CSV for the 'slope' generated quantity.
    """
    input_path  = os.path.join(tmp_dir, "input.json")
    init_path   = os.path.join(tmp_dir, "init.json")
    output_path = os.path.join(tmp_dir, "optimize.csv")

    with open(input_path, "w") as f:
        json.dump(data_dict, f)
    with open(init_path, "w") as f:
        json.dump(init_dict, f)

    result = subprocess.run(
        [exe_file, "optimize",
         "data",   f"file={input_path}",
         f"init={init_path}",
         "output", f"file={output_path}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None

    try:
        with open(output_path) as f:
            lines = [l.strip() for l in f
                     if not l.startswith("#") and l.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        values = lines[1].split(",")
        row    = dict(zip(header, values))
        return float(row["slope"])
    except (KeyError, ValueError, FileNotFoundError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: REFERENCE SLOPE (1σ ELLIPSE)
# ─────────────────────────────────────────────────────────────────────────────

def compute_1sigma_slope(raw_data, mu, sigma, extra_cuts, exe_file, run_dir):
    """Compute MLE slope for the 1σ GMM ellipse and save to reference_slope.json.

    Returns (slope, N) or (None, 0) on failure.
    """
    cuts = _cuts_at_nsigma(mu, sigma, 1.0)
    cuts.update(extra_cuts)
    data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
    if data_dict is None:
        print("Warning: 1σ ellipse sample too small to fit")
        return None, 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        slope = _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir)
    N = data_dict["N_total"]
    result = {"slope": slope, "N": N,
              "haty_min": cuts["haty_min"], "haty_max": cuts["haty_max"]}
    path = os.path.join(run_dir, "reference_slope.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  1σ ellipse MLE slope: {slope:.4f}  N={N}")
    print(f"  Saved reference slope → {path}")
    return slope, N


def load_1sigma_slope(run_dir):
    """Load previously computed 1σ MLE slope, or return None."""
    path = os.path.join(run_dir, "reference_slope.json")
    if not os.path.exists(path):
        return None, 0
    with open(path) as f:
        d = json.load(f)
    return d.get("slope"), d.get("N", 0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MAG-SPLIT GRID
# ─────────────────────────────────────────────────────────────────────────────

def run_mag_split_grid(raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope,
                       perp_vals, lo_vals, hi_vals):
    """2D grid over (n_sigma_lo, n_sigma_hi) for each n_sigma_perp value.

    n_sigma_lo scales haty_min; n_sigma_hi scales haty_max independently.
    Returns dict {n_sigma_perp: [(nlo, nhi, slope, N), ...]}.
    """
    results_by_perp = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n_sigma_perp in perp_vals:
            results = []
            print(f"\n  [mag-split]  n_σ_perp={n_sigma_perp}")
            print(f"  {'n_σ_lo':>7}  {'n_σ_hi':>7}  {'MLE slope':>10}  {'diff':>8}  {'N':>6}")
            print(f"  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*6}")
            for nlo in lo_vals:
                for nhi in hi_vals:
                    cuts = _cuts_mag_asymmetric(mu, sigma, float(n_sigma_perp),
                                                float(nlo), float(nhi))
                    cuts.update(extra_cuts)
                    data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
                    if data_dict is None:
                        print(f"  {nlo:7.2f}  {nhi:7.2f}  {'—':>10}  {'—':>8}  {'<30':>6}")
                        results.append((float(nlo), float(nhi), float("nan"), 0))
                        continue
                    slope = _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir)
                    N = data_dict["N_total"]
                    if slope is None:
                        print(f"  {nlo:7.2f}  {nhi:7.2f}  {'failed':>10}  {'—':>8}  {N:6d}")
                        results.append((float(nlo), float(nhi), float("nan"), N))
                        continue
                    diff = slope - gmm_slope
                    print(f"  {nlo:7.2f}  {nhi:7.2f}  {slope:10.4f}  {diff:+8.4f}  {N:6d}")
                    results.append((float(nlo), float(nhi), slope, N))
            results_by_perp[float(n_sigma_perp)] = results
    return results_by_perp


def plot_mag_split_grid(results_by_perp, perp_vals, lo_vals, hi_vals,
                        gmm_slope, slope_tol, run_dir, fiducial=None,
                        mu=None, sigma=None):
    """Save (n_perp × 2) heatmap of mag-split grid to mag_split_grid.png.

    Each row is a fixed n_σ_perp value.
    x-axis: n_σ_hi (haty_max), y-axis: n_σ_lo (haty_min).
    Left column: MLE slope with tolerance contour. Right column: N.
    """
    import matplotlib.colors as mcolors

    lo_arr   = np.asarray(lo_vals)
    hi_arr   = np.asarray(hi_vals)
    n_perp   = len(perp_vals)

    # Shared colour scales across all perp rows
    all_slopes = [s for rs in results_by_perp.values() for _, _, s, _ in rs if np.isfinite(s)]
    all_N      = [float(N) for rs in results_by_perp.values() for _, _, _, N in rs if N > 0]
    N_vmin     = min(all_N) if all_N else 0
    N_vmax     = max(all_N) if all_N else 1

    # Centre the diverging colormap on the mean of cells within slope_tol of gmm_slope
    good_slopes = [s for s in all_slopes if abs(s - gmm_slope) <= slope_tol]
    center      = float(np.mean(good_slopes)) if good_slopes else gmm_slope

    all_diffs = [s - center for s in all_slopes]
    abs_max   = min(max(abs(d) for d in all_diffs) if all_diffs else 1.0,
                    2.0 * slope_tol)

    # Diverging symlog norm: linear within ±linthresh, log outside
    linthresh = slope_tol / 4.0
    diff_norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-abs_max, vmax=abs_max,
                                   base=10)

    # Compute actual cut values for secondary axes if GMM params provided
    if mu is not None and sigma is not None:
        unit     = _cuts_at_nsigma(mu, sigma, 1.0)
        y_extent = unit["haty_max"] - float(mu[1])
        haty_max_vals = float(mu[1]) + hi_arr * y_extent  # for top x-axis
        haty_min_vals = float(mu[1]) - lo_arr * y_extent  # for right y-axis
    else:
        haty_max_vals = None
        haty_min_vals = None

    fig, axes = plt.subplots(n_perp, 2, figsize=(11, 4.5 * n_perp),
                             squeeze=False)

    for row, n_sigma_perp in enumerate(perp_vals):
        results = results_by_perp[float(n_sigma_perp)]
        slope_grid = np.full((len(lo_arr), len(hi_arr)), np.nan)
        N_grid     = np.full((len(lo_arr), len(hi_arr)), np.nan)
        for nlo, nhi, slope, N in results:
            il = int(np.argmin(np.abs(lo_arr - nlo)))
            ih = int(np.argmin(np.abs(hi_arr - nhi)))
            slope_grid[il, ih] = slope
            N_grid[il, ih]     = float(N) if N > 0 else np.nan
        diff_grid = slope_grid - center

        for col, (data, cmap_name, norm, cbar_label) in enumerate([
                (diff_grid, "RdBu",    diff_norm,
                 "MLE slope"),
                (N_grid,    "viridis", mcolors.Normalize(vmin=N_vmin, vmax=N_vmax),
                 "N")]):
            ax = axes[row, col]
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("lightgray")
            pcm = ax.pcolormesh(hi_arr, lo_arr, data, cmap=cmap, norm=norm,
                                shading="nearest")
            cb = fig.colorbar(pcm, ax=ax, label=cbar_label)
            if col == 0:
                ticks = cb.get_ticks()
                cb.set_ticks(ticks)
                cb.set_ticklabels([f"{center + t:.2f}" for t in ticks])
            if col == 0:
                try:
                    ax.contour(hi_arr, lo_arr, np.abs(diff_grid),
                               levels=[slope_tol], colors="white", linewidths=1.2)
                except Exception:
                    pass
            ax.set_xlabel(r"$n_{\sigma,\hat{y}_\text{max}}$")
            ax.set_ylabel(r"$n_{\sigma,\hat{y}_\text{min}}$")
            col_title = "MLE slope" if col == 0 else "N"
            ax.set_title(f"$n_{{\\sigma,\\perp}}={n_sigma_perp:.1f}$  —  {col_title}")
            if haty_max_vals is not None and row == n_perp - 1:
                ax_top = ax.twiny()
                ax_top.set_xlim(ax.get_xlim())
                ax_top.set_xticks(hi_arr)
                ax_top.set_xticklabels([f"{v:.1f}" for v in haty_max_vals],
                                       fontsize=7, rotation=45, ha="left")
                ax_top.set_xlabel(r"$\hat{y}_\text{max}$", fontsize=8)
            if haty_min_vals is not None and row == n_perp - 1:
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())
                ax_right.set_yticks(lo_arr)
                ax_right.set_yticklabels([f"{v:.1f}" for v in haty_min_vals],
                                         fontsize=7)
                ax_right.set_ylabel(r"$\hat{y}_\text{min}$", fontsize=8)

    if fiducial is not None:
        fid_row = int(np.argmin(np.abs(np.array(list(perp_vals)) - fiducial["n_sigma_perp"])))
        for col in (0, 1):
            axes[fid_row, col].plot(fiducial["n_sigma_hi"], fiducial["n_sigma_lo"],
                                    "*", color="black", markersize=12,
                                    zorder=5)

    fig.suptitle(
        f"Mag-split grid  (GMM slope={gmm_slope:.3f}, color center={center:.3f}, tol={slope_tol})",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(run_dir, "mag_split_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved mag-split heatmap → {out_path}")


def plot_fiducial_slope_hist(results_by_perp, perp_vals, gmm_slope,
                             slope_tol, run_dir, ref_slope=None):
    """Histogram of all MLE slopes, with good-cell and 16–84 percentile subsets overlaid."""
    all_slopes, good_slopes = [], []
    for n_sigma_perp in perp_vals:
        for _, _, slope, _ in results_by_perp[float(n_sigma_perp)]:
            if not np.isfinite(slope):
                continue
            all_slopes.append(slope)
            if abs(slope - gmm_slope) < slope_tol:
                good_slopes.append(slope)
    if not all_slopes:
        return
    all_slopes  = np.array(all_slopes)
    good_slopes = np.array(good_slopes)
    lo_pct = float(np.percentile(good_slopes, 16)) if len(good_slopes) else gmm_slope
    hi_pct = float(np.percentile(good_slopes, 84)) if len(good_slopes) else gmm_slope
    mid_slopes = good_slopes[(good_slopes >= lo_pct) & (good_slopes <= hi_pct)]

    bins = np.linspace(good_slopes.min() - 0.05, good_slopes.max() + 0.05, 30)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(good_slopes, bins=bins, color="steelblue", alpha=0.7,
            label=f"|diff| < tol (N={len(good_slopes)})")
    ax.hist(mid_slopes,  bins=bins, color="orange",    alpha=0.9,
            label=f"16–84 pct band (N={len(mid_slopes)})")
    ax.axvline(lo_pct,    color="orange", linestyle="--", linewidth=1.2,
               label=f"16nd pct = {lo_pct:.3f}")
    ax.axvline(hi_pct,    color="orange", linestyle=":",  linewidth=1.2,
               label=f"84th pct = {hi_pct:.3f}")

    ax.set_xlabel("MLE slope")
    ax.set_ylabel("Count")
    ax.set_title("MLE slope distribution — fiducial selection")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "fiducial_slope_hist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved slope histogram → {out_path}")


def find_fiducial_from_mag_split(results_by_perp, perp_vals, gmm_slope,
                                 slope_tol, contraction=1.0):
    """Select fiducial (n_σ_perp, n_σ_lo, n_σ_hi) from mag-split grid.

    1. Filter to cells with |MLE slope − GMM slope| < slope_tol.
    2. Among those, keep only cells whose slope falls within the 16nd–84th
       percentile of good-cell slopes (the "typical" slope band).
    3. Among that subset, pick the cell with the maximum N.

    Returns dict with keys: n_sigma_perp, n_sigma_lo, n_sigma_hi,
    mle_slope, N.  Returns None if no good cell is found.
    """
    good = []
    for n_sigma_perp in perp_vals:
        for nlo, nhi, slope, N in results_by_perp[float(n_sigma_perp)]:
            if not np.isfinite(slope):
                continue
            if abs(slope - gmm_slope) < slope_tol:
                good.append((float(n_sigma_perp), float(nlo), float(nhi), slope, N))
    if not good:
        return None
    slopes = np.array([s for _, _, _, s, _ in good])
    lo_pct = float(np.percentile(slopes, 16))
    hi_pct = float(np.percentile(slopes, 84))
    mid = [(perp, nlo, nhi, slope, N) for perp, nlo, nhi, slope, N in good
           if lo_pct <= slope <= hi_pct]
    if not mid:
        mid = good
    best = max(mid, key=lambda r: r[4])
    perp, lo, hi, slope, N = best
    return dict(n_sigma_perp=contraction * perp,
                n_sigma_lo=contraction * lo,
                n_sigma_hi=contraction * hi,
                mle_slope=slope, N=N)


def save_fiducial(fiducial, mu, sigma, run_dir):
    """Compute actual cut values at the fiducial n_σ point and save to JSON.

    Writes output/<run>/mag_split_fiducial.json with keys:
      n_sigma_perp, n_sigma_lo, n_sigma_hi, mle_slope, N,
      haty_min, haty_max, slope_plane, intercept_plane, intercept_plane2
    Also prints a human-readable summary.
    """
    cuts = _cuts_mag_asymmetric(mu, sigma,
                                fiducial["n_sigma_perp"],
                                fiducial["n_sigma_lo"],
                                fiducial["n_sigma_hi"])
    out = {**fiducial, **cuts}
    path = os.path.join(run_dir, "mag_split_fiducial.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n→ Fiducial: n_σ_perp={fiducial['n_sigma_perp']:.3f}"
          f"  n_σ_lo={fiducial['n_sigma_lo']:.3f}"
          f"  n_σ_hi={fiducial['n_sigma_hi']:.3f}")
    print(f"           MLE slope={fiducial['mle_slope']:.4f}  N={fiducial['N']}")
    print(f"           haty_min={cuts['haty_min']:.3f}"
          f"  haty_max={cuts['haty_max']:.3f}")
    print(f"           slope_plane={cuts['slope_plane']:.4f}"
          f"  intercept_plane={cuts['intercept_plane']:.3f}"
          f"  intercept_plane2={cuts['intercept_plane2']:.3f}")
    print(f"  Saved fiducial → {path}")
    return out


def save_mag_split_results(results_by_perp, perp_vals, lo_vals, hi_vals, run_dir):
    """Save mag-split grid results to mag_split_grid.json."""
    out = {
        "perp_vals": list(perp_vals),
        "lo_vals":   list(lo_vals),
        "hi_vals":   list(hi_vals),
        "results_by_perp": {
            str(k): v for k, v in results_by_perp.items()
        },
    }
    out_path = os.path.join(run_dir, "mag_split_grid.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved mag-split results → {out_path}")


def load_mag_split_results(run_dir):
    """Load mag-split grid results from mag_split_grid.json."""
    in_path = os.path.join(run_dir, "mag_split_grid.json")
    with open(in_path) as f:
        data = json.load(f)
    perp_vals = np.array(data["perp_vals"])
    lo_vals   = np.array(data["lo_vals"])
    hi_vals   = np.array(data["hi_vals"])
    results_by_perp = {
        float(k): [tuple(row) for row in v]
        for k, v in data["results_by_perp"].items()
    }
    return results_by_perp, perp_vals, lo_vals, hi_vals


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep ellipse scale n_σ for each cut parameter using Stan optimize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["fullmocks", "DESI"], default="fullmocks",
                        help="Data source")
    parser.add_argument("--fits_file", default=None,
                        help="Path to a single FITS file; auto-detected from --dir if omitted")
    parser.add_argument("--dir", default="data/",
                        help="Directory searched for FITS files when --fits_file is omitted")
    parser.add_argument("--run", required=True,
                        help="Run name; reads output/<run>/selection_ellipse.json")
    parser.add_argument("--exe", default="tophat",
                        help="Path to compiled Stan tophat executable")
    parser.add_argument("--z_obs_min", type=float, default=0.03,
                        help="Minimum redshift cut")
    parser.add_argument("--z_obs_max", type=float, default=0.10,
                        help="Maximum redshift cut")
    parser.add_argument("--n_sweep_objects", type=int, default=10000,
                        help="Subsample raw data to this many objects (0 = use all)")
    parser.add_argument("--slope_tol", type=float, default=0.5,
                        help="Tolerance |MLE slope - GMM slope| for tolerance contour")
    parser.add_argument("--fiducial_contraction", type=float, default=1.0,
                        help="Factor applied to best grid point to get fiducial (< 1 pulls inward)")
    # Mag-split grid
    parser.add_argument("--mag_split_plot", action="store_true",
                        help="Replot mag-split grid from saved mag_split_grid.json (no Stan calls)")
    parser.add_argument("--n_sigma_perp_min", type=float, default=5.0,
                        help="Lower end of n_σ_perp grid")
    parser.add_argument("--n_sigma_perp_max", type=float, default=5.0,
                        help="Upper end of n_σ_perp grid")
    parser.add_argument("--n_sigma_perp_n",   type=int,   default=1,
                        help="Number of n_σ_perp grid points")
    parser.add_argument("--n_sigma_mag_lo_min", type=float, default=0.2,
                        help="Lower end of n_σ_lo grid (haty_min)")
    parser.add_argument("--n_sigma_mag_lo_max", type=float, default=1.4,
                        help="Upper end of n_σ_lo grid (haty_min)")
    parser.add_argument("--n_sigma_mag_lo_n",   type=int,   default=8,
                        help="Number of n_σ_lo grid points")
    parser.add_argument("--n_sigma_mag_hi_min", type=float, default=1,
                        help="Lower end of n_σ_hi grid (haty_max)")
    parser.add_argument("--n_sigma_mag_hi_max", type=float, default=3,
                        help="Upper end of n_σ_hi grid (haty_max)")
    parser.add_argument("--n_sigma_mag_hi_n",   type=int,   default=9,
                        help="Number of n_σ_hi grid points")

    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # Load ellipse JSON
    ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
    if not os.path.exists(ellipse_path):
        raise FileNotFoundError(
            f"{ellipse_path} not found — run selection_ellipse.py first.")
    with open(ellipse_path) as f:
        ell = json.load(f)
    mu    = np.array(ell["mean"])
    sigma = np.array(ell["covariance"])

    # Load raw data
    if args.source == "fullmocks":
        fits_file = args.fits_file
        if fits_file is None:
            pattern = os.path.join(args.dir, "TF_extended_AbacusSummit_*.fits")
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No FITS files found: {pattern}")
            fits_file = matches[0]
            print(f"Auto-selected: {fits_file}")
        raw_data = load_fullmocks(fits_file)
    elif args.source == "DESI":
        fits_file = args.fits_file or "data/DESI-DR1_TF_pv_cat_v15.fits"
        raw_data = load_desi(fits_file)
    else:
        raise NotImplementedError(f"Unsupported --source: {args.source}")

    n_cap = args.n_sweep_objects
    if n_cap and len(raw_data["x"]) > n_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(raw_data["x"]), size=n_cap, replace=False)
        raw_data = {k: (v[idx] if isinstance(v, np.ndarray) else v)
                    for k, v in raw_data.items()}
        print(f"Subsampled raw data to {n_cap} objects")

    extra_cuts = {}
    if args.z_obs_min is not None:
        extra_cuts["z_obs_min"] = args.z_obs_min
    if args.z_obs_max is not None:
        extra_cuts["z_obs_max"] = args.z_obs_max

    exe_file = args.exe
    if not os.path.isabs(exe_file):
        if os.path.exists(exe_file):
            exe_file = os.path.abspath(exe_file)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate  = os.path.join(script_dir, exe_file)
            if os.path.exists(candidate):
                exe_file = candidate

    gmm_slope = float(ell["slope_plane"])

    # ── Plot-only mode ───────────────────────────────────────────────────────
    if args.mag_split_plot:
        results_by_perp, perp_vals, lo_vals, hi_vals = load_mag_split_results(run_dir)
        fiducial = find_fiducial_from_mag_split(
            results_by_perp, perp_vals, gmm_slope,
            args.slope_tol, contraction=args.fiducial_contraction)
        plot_fiducial_slope_hist(results_by_perp, perp_vals, gmm_slope,
                                 args.slope_tol, run_dir)
        plot_mag_split_grid(results_by_perp, perp_vals, lo_vals, hi_vals,
                            gmm_slope, args.slope_tol, run_dir, fiducial=fiducial,
                            mu=mu, sigma=sigma)
        if fiducial is not None:
            save_fiducial(fiducial, mu, sigma, run_dir)
        import sys; sys.exit(0)

    # ── Mag-split grid ───────────────────────────────────────────────────────
    perp_vals = np.linspace(args.n_sigma_perp_min, args.n_sigma_perp_max,
                            args.n_sigma_perp_n)
    lo_vals   = np.linspace(args.n_sigma_mag_lo_min, args.n_sigma_mag_lo_max,
                            args.n_sigma_mag_lo_n)
    hi_vals   = np.linspace(args.n_sigma_mag_hi_min, args.n_sigma_mag_hi_max,
                            args.n_sigma_mag_hi_n)
    print(f"Running mag-split grid: perp={perp_vals}  "
          f"lo={lo_vals}  hi={hi_vals}  exe={exe_file}")
    mag_split_results = run_mag_split_grid(
        raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope,
        perp_vals, lo_vals, hi_vals)
    save_mag_split_results(mag_split_results, perp_vals, lo_vals, hi_vals, run_dir)
    fiducial = find_fiducial_from_mag_split(
        mag_split_results, perp_vals, gmm_slope,
        args.slope_tol, contraction=args.fiducial_contraction)
    if fiducial is None:
        print("Warning: no good cell found within slope_tol — no fiducial saved")
    else:
        save_fiducial(fiducial, mu, sigma, run_dir)
    plot_fiducial_slope_hist(mag_split_results, perp_vals, gmm_slope,
                             args.slope_tol, run_dir)
    plot_mag_split_grid(
        mag_split_results, perp_vals, lo_vals, hi_vals,
        gmm_slope, args.slope_tol, run_dir, fiducial=fiducial,
        mu=mu, sigma=sigma)
