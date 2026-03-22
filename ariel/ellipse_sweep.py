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
    mag_e = np.asarray(data["R_ABSMAG_SB26_ERR"], dtype=float)

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


def _cuts_mixed(mu, sigma, n_sigma_perp, n_sigma_mag):
    """Cuts with independent scaling for strip width and magnitude window.

    intercept_plane, intercept_plane2, slope_plane → from n_sigma_perp
    haty_min, haty_max                             → from n_sigma_mag
    """
    perp = _cuts_at_nsigma(mu, sigma, n_sigma_perp)
    mag  = _cuts_at_nsigma(mu, sigma, n_sigma_mag)
    return dict(
        haty_min        = mag["haty_min"],
        haty_max        = mag["haty_max"],
        slope_plane     = perp["slope_plane"],
        intercept_plane = perp["intercept_plane"],
        intercept_plane2= perp["intercept_plane2"],
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
# SECTION 5: SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def _run_grid(raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope, slope_tol,
              perp_vals, mag_vals, label, tmp_dir):
    """Run MLE slope at every (perp, mag) grid point; return list of (nsp, nsm, slope, N)."""
    results = []
    print(f"\n  [{label}]")
    print(f"  {'n_σ_perp':>9}  {'n_σ_mag':>7}  {'MLE slope':>10}  {'diff':>8}  {'N':>6}")
    print(f"  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*6}")
    for n_sigma_perp in perp_vals:
        for n_sigma_mag in mag_vals:
            cuts = _cuts_mixed(mu, sigma, float(n_sigma_perp), float(n_sigma_mag))
            cuts.update(extra_cuts)
            data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
            if data_dict is None:
                print(f"  {n_sigma_perp:9.2f}  {n_sigma_mag:7.2f}  {'—':>10}  {'—':>8}  {'<30':>6}")
                results.append((float(n_sigma_perp), float(n_sigma_mag), float("nan"), 0))
                continue
            slope = _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir)
            N = data_dict["N_total"]
            if slope is None:
                print(f"  {n_sigma_perp:9.2f}  {n_sigma_mag:7.2f}  {'failed':>10}  {'—':>8}  {N:6d}")
                results.append((float(n_sigma_perp), float(n_sigma_mag), float("nan"), N))
                continue
            diff = slope - gmm_slope
            print(f"  {n_sigma_perp:9.2f}  {n_sigma_mag:7.2f}  {slope:10.4f}  {diff:+8.4f}  {N:6d}")
            results.append((float(n_sigma_perp), float(n_sigma_mag), slope, N))
    return results


def find_fiducial_cuts(raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope,
                       n_sigma_perp_vals, n_sigma_mag_vals, slope_tol=0.5,
                       contraction=0.9, coarse_only=False, n_fine_perp=10, n_fine_mag=8):
    """Two-phase 2D grid search over (n_sigma_perp, n_sigma_mag) for the fiducial cuts.

    Phase 1 (coarse): runs on the supplied n_sigma_perp_vals × n_sigma_mag_vals grid.
    Phase 2 (fine): zooms into the perp/mag transition band found in phase 1.

    Best-point selection (max N within slope_tol) runs on fine-grid results only.

    Returns (fiducial_n_sigma_perp, fiducial_n_sigma_mag, ref_cuts, mle_slope,
             grid_info, raw_best_perp, raw_best_mag).
    """
    print(f"\nSearching for fiducial cuts (GMM slope={gmm_slope:.4f}, tol={slope_tol}) …")
    print(f"  Coarse n_σ_perp grid: {n_sigma_perp_vals}")
    print(f"  Coarse n_σ_mag  grid: {n_sigma_mag_vals}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # ── Phase 1: coarse ──────────────────────────────────────────────────
        coarse_results = _run_grid(
            raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope, slope_tol,
            n_sigma_perp_vals, n_sigma_mag_vals, "coarse", tmp_dir)

        # Detect transition band in n_σ_perp
        perp_good = {}
        for nsp, nsm, slope, N in coarse_results:
            if not np.isnan(slope) and abs(slope - gmm_slope) <= slope_tol:
                perp_good[nsp] = True
            elif nsp not in perp_good:
                perp_good[nsp] = False

        sorted_perps = np.sort(list(perp_good.keys()))
        no_good_perps  = [p for p in sorted_perps if not perp_good[p]]
        has_good_perps = [p for p in sorted_perps if perp_good[p]]

        if not no_good_perps:
            print("  Warning: coarse grid has no biased-slope cells — widen perp range")
        if not has_good_perps:
            print("  Warning: coarse grid has no cells within slope_tol — widen perp range")

        if no_good_perps and has_good_perps:
            coarse_perp_step = float(sorted_perps[1] - sorted_perps[0]) if len(sorted_perps) > 1 else 0
            fine_perp_lo = max(float(sorted_perps[0]),  no_good_perps[-1]  - coarse_perp_step)
            fine_perp_hi = min(float(sorted_perps[-1]), has_good_perps[-1] + coarse_perp_step)
        else:
            fine_perp_lo, fine_perp_hi = float(sorted_perps[0]), float(sorted_perps[-1])

        # Mag range: span of "good" cells ± 1 coarse step
        good_mags   = sorted({nsm for _, nsm, slope, _ in coarse_results
                               if not np.isnan(slope) and abs(slope - gmm_slope) <= slope_tol})
        sorted_mags = np.sort(list({nsm for _, nsm, _, _ in coarse_results}))
        if good_mags and len(sorted_mags) > 1:
            coarse_mag_step = float(sorted_mags[1] - sorted_mags[0])
            fine_mag_lo = max(float(sorted_mags[0]),  good_mags[0]  - coarse_mag_step)
            fine_mag_hi = min(float(sorted_mags[-1]), good_mags[-1] + coarse_mag_step)
        else:
            fine_mag_lo, fine_mag_hi = float(sorted_mags[0]), float(sorted_mags[-1])

        fine_perp_vals = np.linspace(fine_perp_lo, fine_perp_hi, n_fine_perp)
        fine_mag_vals  = np.linspace(fine_mag_lo,  fine_mag_hi,  n_fine_mag)

        if coarse_only:
            fine_results = []
            fine_perp_vals = n_sigma_perp_vals
            fine_mag_vals  = n_sigma_mag_vals
        else:
            print(f"\n  Fine n_σ_perp: [{fine_perp_lo:.3f}, {fine_perp_hi:.3f}]  ({n_fine_perp} pts)")
            print(f"  Fine n_σ_mag:  [{fine_mag_lo:.3f},  {fine_mag_hi:.3f}]  ({n_fine_mag} pts)")

            # ── Phase 2: fine ────────────────────────────────────────────────────
            fine_results = _run_grid(
                raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope, slope_tol,
                fine_perp_vals, fine_mag_vals, "fine", tmp_dir)

    # Best-point selection: fine results when available, coarse when coarse_only
    selection_results = coarse_results if coarse_only else fine_results
    best_perp  = None
    best_mag   = None
    best_cuts  = None
    best_slope = None
    best_N     = -1
    fb_perp  = None
    fb_mag   = None
    fb_cuts  = None
    fb_slope = None
    fb_diff  = float("inf")

    for nsp, nsm, slope, N in selection_results:
        if np.isnan(slope):
            continue
        diff = slope - gmm_slope
        if abs(diff) <= slope_tol and N > best_N:
            best_N     = N
            best_perp  = nsp
            best_mag   = nsm
            best_cuts  = _cuts_mixed(mu, sigma, nsp, nsm)
            best_cuts.update(extra_cuts)
            best_slope = slope
        if abs(diff) < fb_diff:
            fb_diff  = abs(diff)
            fb_perp  = nsp
            fb_mag   = nsm
            fb_cuts  = _cuts_mixed(mu, sigma, nsp, nsm)
            fb_cuts.update(extra_cuts)
            fb_slope = slope

    phase_label = "coarse" if coarse_only else "fine"
    if best_perp is None and fb_perp is not None:
        best_perp  = fb_perp
        best_mag   = fb_mag
        best_cuts  = fb_cuts
        best_slope = fb_slope
        print(f"  Warning: no {phase_label}-grid point within tol={slope_tol}; using min |diff| fallback.")

    if best_perp is None:
        best_perp  = float(fine_perp_vals[-1])
        best_mag   = float(fine_mag_vals[-1])
        best_cuts  = _cuts_mixed(mu, sigma, best_perp, best_mag)
        best_cuts.update(extra_cuts)
        best_slope = None
        print("  Warning: no valid fine-grid point found; using last grid point as fallback.")

    raw_best_perp = best_perp
    raw_best_mag  = best_mag

    # Apply slight contraction and re-run MLE at the contracted point
    if contraction != 1.0:
        c_perp = best_perp * contraction
        c_mag  = best_mag  * contraction
        print(f"\n  Contracting fiducial by {contraction}: "
              f"({best_perp:.2f}, {best_mag:.2f}) → ({c_perp:.2f}, {c_mag:.2f})")
        c_cuts = _cuts_mixed(mu, sigma, c_perp, c_mag)
        c_cuts.update(extra_cuts)
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dict, init_dict = _build_stan_dicts(raw_data, c_cuts)
            if data_dict is not None:
                c_slope = _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir)
                best_perp  = c_perp
                best_mag   = c_mag
                best_cuts  = c_cuts
                best_slope = c_slope
            else:
                print("  Warning: contracted point has N < _N_MIN; keeping uncontracted fiducial.")

    print(f"\n→ fiducial: n_sigma_perp={best_perp:.2f}  n_sigma_mag={best_mag:.2f}"
          f"  MLE slope={best_slope if best_slope is not None else 'n/a'}")

    grid_info = dict(
        coarse_results=coarse_results,
        coarse_perp_vals=n_sigma_perp_vals,
        coarse_mag_vals=n_sigma_mag_vals,
        fine_results=fine_results,
        fine_perp_vals=fine_perp_vals,
        fine_mag_vals=fine_mag_vals,
        coarse_only=coarse_only,
    )
    return best_perp, best_mag, best_cuts, best_slope, grid_info, raw_best_perp, raw_best_mag


def _make_grids(results, perp_vals, mag_vals):
    """Build (slope_grid, N_grid) from a list of (nsp, nsm, slope, N) results."""
    perp_arr = np.asarray(perp_vals)
    mag_arr  = np.asarray(mag_vals)
    slope_grid = np.full((len(mag_arr), len(perp_arr)), np.nan)
    N_grid     = np.full((len(mag_arr), len(perp_arr)), np.nan)
    for nsp, nsm, slope, N in results:
        ip = int(np.argmin(np.abs(perp_arr - nsp)))
        im = int(np.argmin(np.abs(mag_arr  - nsm)))
        slope_grid[im, ip] = slope
        N_grid[im, ip]     = float(N) if N > 0 else np.nan
    return slope_grid, N_grid


def plot_fiducial_search(grid_info, gmm_slope, slope_tol,
                         raw_best_perp, raw_best_mag,
                         fiducial_perp, fiducial_mag,
                         run_dir):
    """Save a heatmap of the fiducial grid search to fiducial_search.png.

    coarse_only=False (default): 2×2 layout.
      Row 0: coarse grid (slope | N) with dashed rectangle marking fine zoom region.
      Row 1: fine grid   (slope | N) with open-circle raw-best and yellow-star fiducial.

    coarse_only=True: 1×2 layout.
      Single row: coarse grid (slope | N) with markers on both panels.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    is_coarse_only = grid_info.get("coarse_only", False)

    coarse_perp = np.asarray(grid_info["coarse_perp_vals"])
    coarse_mag  = np.asarray(grid_info["coarse_mag_vals"])
    fine_perp   = np.asarray(grid_info["fine_perp_vals"])
    fine_mag    = np.asarray(grid_info["fine_mag_vals"])

    c_slope_grid, c_N_grid = _make_grids(
        grid_info["coarse_results"], coarse_perp, coarse_mag)
    c_diff_grid = c_slope_grid - gmm_slope

    if is_coarse_only:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        rows = [
            ([axes[0], axes[1]], coarse_perp, coarse_mag,
             c_slope_grid, c_N_grid, c_diff_grid, "Coarse"),
        ]
    else:
        f_slope_grid, f_N_grid = _make_grids(
            grid_info["fine_results"], fine_perp, fine_mag)
        f_diff_grid = f_slope_grid - gmm_slope
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        rows = [
            (axes[0], coarse_perp, coarse_mag, c_slope_grid, c_N_grid, c_diff_grid, "Coarse"),
            (axes[1], fine_perp,   fine_mag,   f_slope_grid, f_N_grid, f_diff_grid, "Fine"),
        ]

    for row_axes, perp_arr, mag_arr, slope_grid, N_grid, diff_grid, row_label in rows:
        for ax, data, cmap_name, cbar_label, title in zip(
                row_axes,
                [slope_grid, N_grid],
                ["RdYlGn", "viridis"],
                ["MLE slope", "N"],
                ["MLE slope" if is_coarse_only else f"{row_label}: MLE slope",
                 "N"         if is_coarse_only else f"{row_label}: N"]):

            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("lightgray")
            vmin = np.nanmin(data) if np.any(np.isfinite(data)) else 0
            vmax = np.nanmax(data) if np.any(np.isfinite(data)) else 1
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            pcm = ax.pcolormesh(perp_arr, mag_arr, data, cmap=cmap, norm=norm,
                                shading="nearest")
            fig.colorbar(pcm, ax=ax, label=cbar_label)

            # tolerance contour |diff| = slope_tol
            try:
                ax.contour(perp_arr, mag_arr, np.abs(diff_grid), levels=[slope_tol],
                           colors="white", linewidths=1.2)
            except Exception:
                pass

            ax.set_xlabel(r"$n_{\sigma,\perp}$")
            ax.set_ylabel(r"$n_{\sigma,\mathrm{mag}}$")
            ax.set_title(title)

        # Coarse row in two-phase mode: dashed rectangle showing fine zoom region
        if row_label == "Coarse" and not is_coarse_only:
            fp_lo, fp_hi = float(fine_perp[0]),  float(fine_perp[-1])
            fm_lo, fm_hi = float(fine_mag[0]),   float(fine_mag[-1])
            for ax in row_axes:
                rect = mpatches.Rectangle(
                    (fp_lo, fm_lo), fp_hi - fp_lo, fm_hi - fm_lo,
                    linewidth=1.5, edgecolor="white", facecolor="none",
                    linestyle="--", zorder=5)
                ax.add_patch(rect)

        # Markers: on fine row in two-phase mode; on coarse row in coarse_only mode
        if (row_label == "Fine" and not is_coarse_only) or is_coarse_only:
            for ax in row_axes:
                if raw_best_perp is not None and raw_best_mag is not None:
                    ax.plot(raw_best_perp, raw_best_mag, "o", color="white",
                            mfc="none", mew=1.5, ms=10, label="best grid pt", zorder=5)
                if fiducial_perp is not None and fiducial_mag is not None:
                    ax.plot(fiducial_perp, fiducial_mag, "*", color="yellow",
                            ms=14, label="fiducial", zorder=6)
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Fiducial search  (GMM slope={gmm_slope:.3f}, tol={slope_tol})")
    fig.tight_layout()
    out_path = os.path.join(run_dir, "fiducial_search.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved fiducial search heatmap → {out_path}")


def run_ellipse_sweep(raw_data, mu, sigma, n_sigma_vals, extra_cuts, exe_file,
                      ref_cuts=None):
    """Sweep each ellipse cut parameter over n_sigma_vals, others fixed at ref_cuts.

    For each of the four cut parameters (haty_min, haty_max, intercept_plane,
    intercept_plane2), varies the ellipse scale n_σ while keeping the others
    fixed at their ref_cuts values.  slope_plane is always fixed at the
    ref_cuts value.

    Returns dict mapping parameter name → list of (n_sigma, cut_value, slope, N).
    """
    _SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]
    if ref_cuts is None:
        ref_cuts = _cuts_at_nsigma(mu, sigma, 1.0)
    results = {p: [] for p in _SWEEP_PARAMS}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for p in _SWEEP_PARAMS:
            print(f"  Sweeping {p} …")
            for n_sigma in n_sigma_vals:
                cuts_nsigma = _cuts_at_nsigma(mu, sigma, n_sigma)
                cuts = dict(ref_cuts)
                cuts[p] = cuts_nsigma[p]
                cuts["slope_plane"] = ref_cuts["slope_plane"]
                cuts.update(extra_cuts)

                data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
                if data_dict is None:
                    results[p].append((float(n_sigma), cuts_nsigma[p], None, 0))
                    continue

                slope = _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir)
                results[p].append((float(n_sigma), cuts_nsigma[p], slope,
                                   data_dict["N_total"]))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_LABELS = {
    "haty_max":         r"$\hat{y}_{\max}$",
    "haty_min":         r"$\hat{y}_{\min}$",
    "slope_plane":      r"$\bar{s}$ (plane slope)",
    "intercept_plane":  r"$c_1$ (lower intercept)",
    "intercept_plane2": r"$c_2$ (upper intercept)",
}


def _label(p):
    return _PARAM_LABELS.get(p, p)


def save_sweep_results(sweep_results, run_dir):
    """Serialize sweep slopes and ∂s/∂(n_σ) to output/<run>/ellipse_sweep.json.

    JSON structure:
      {
        "<param>": {
          "n_sigma":           [float, ...],
          "cut_values":        [float, ...],
          "slopes":            [float or null, ...],
          "d_slope_d_nsigma":  [float or null, ...]
        },
        ...
      }
    Derivatives are computed on the valid (non-null) slope subset and mapped
    back to the full grid; positions with no slope are stored as null.
    """
    _SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]
    out = {}
    for p in _SWEEP_PARAMS:
        records  = sweep_results[p]
        ns_arr   = np.array([r[0] for r in records])
        sl_arr   = np.array([r[2] if r[2] is not None else np.nan for r in records])
        cut_vals = np.array([r[1] for r in records])

        valid       = np.isfinite(sl_arr)
        derivs_full = np.full(len(ns_arr), np.nan)
        if valid.sum() >= 2:
            derivs_full[valid] = np.gradient(sl_arr[valid], ns_arr[valid])

        def _to_list(arr):
            return [None if np.isnan(v) else float(v) for v in arr]

        out[p] = {
            "n_sigma":          ns_arr.tolist(),
            "cut_values":       cut_vals.tolist(),
            "slopes":           _to_list(sl_arr),
            "d_slope_d_nsigma": _to_list(derivs_full),
        }

    out_file = os.path.join(run_dir, "ellipse_sweep.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_file}")


def plot_ellipse_partial_deriv(sweep_results, run_dir, ref_n_sigma_by_param=None):
    """2-row × 4-column figure: slope vs n_σ (top) and ∂s/∂(n_σ) vs n_σ (bottom).

    ref_n_sigma_by_param: dict mapping each sweep param to its reference n_σ
    for the axvline.  Falls back to 1.0 for any param not in the dict.

    Saved to output/<run>/ellipse_sweep.png.
    """
    _SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]
    n_params = len(_SWEEP_PARAMS)
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))

    for col, p in enumerate(_SWEEP_PARAMS):
        records  = sweep_results[p]
        ns_arr   = np.array([r[0] for r in records])
        slopes   = np.array([r[2] if r[2] is not None else np.nan
                             for r in records])
        cut_vals = np.array([r[1] for r in records])

        ref_ns = ref_n_sigma_by_param.get(p, 1.0) if ref_n_sigma_by_param else 1.0

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: slope vs n_σ  (log x-axis)
        valid = np.isfinite(slopes)
        ax_top.plot(ns_arr[valid], slopes[valid], "o-", color="steelblue",
                    linewidth=1.5, markersize=4)
        ax_top.set_xscale("log")
        ax_top.axvline(ref_ns, color="gray", linestyle="--", linewidth=1.0,
                       label=rf"$n_\sigma={ref_ns:.2f}$")
        ax_top.set_xlabel(r"$n_\sigma$", fontsize=10)
        ax_top.set_ylabel("MLE slope", fontsize=10)
        ax_top.set_title(_label(p), fontsize=11)
        ax_top.grid(True, alpha=0.3, which="both")
        ax_top.legend(fontsize=8)

        # Secondary top x-axis: cut value at each n_σ tick
        n_ticks  = min(5, len(ns_arr))
        tick_idx = np.round(np.linspace(0, len(ns_arr) - 1, n_ticks)).astype(int)
        ax_top2 = ax_top.twiny()
        ax_top2.set_xscale("log")
        ax_top2.set_xlim(ax_top.get_xlim())
        ax_top2.set_xticks(ns_arr[tick_idx])
        ax_top2.set_xticklabels([f"{cut_vals[i]:.2f}" for i in tick_idx],
                                fontsize=7, rotation=45)
        ax_top2.set_xlabel(_label(p) + " value", fontsize=8)

        # Bottom: ∂s/∂(n_σ) vs n_σ  (log x-axis)
        if valid.sum() >= 2:
            ns_v   = ns_arr[valid]
            sl_v   = slopes[valid]
            derivs = np.gradient(sl_v, ns_v)
            ax_bot.plot(ns_v, derivs, "o-", color="purple",
                        linewidth=1.5, markersize=4)
        ax_bot.set_xscale("log")
        ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_bot.axvline(ref_ns, color="gray", linestyle="--", linewidth=1.0,
                       label=rf"$n_\sigma={ref_ns:.2f}$")
        ax_bot.set_xlabel(r"$n_\sigma$", fontsize=10)
        ax_bot.set_ylabel(r"$\partial s / \partial (n_\sigma)$", fontsize=10)
        ax_bot.grid(True, alpha=0.3, which="both")
        ax_bot.legend(fontsize=8)

    fig.suptitle(
        r"Ellipse sweep: $\partial s / \partial n_\sigma$ for each cut parameter",
        fontsize=13)
    plt.tight_layout()
    out_file = os.path.join(run_dir, "ellipse_sweep.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6b: MAG-SPLIT GRID
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
                        gmm_slope, slope_tol, run_dir):
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
    all_diffs = [s - gmm_slope for rs in results_by_perp.values()
                 for _, _, s, _ in rs if np.isfinite(s)]
    all_N     = [float(N) for rs in results_by_perp.values()
                 for _, _, _, N in rs if N > 0]
    abs_max   = max(abs(d) for d in all_diffs) if all_diffs else 1.0
    N_vmin    = min(all_N) if all_N else 0
    N_vmax    = max(all_N) if all_N else 1

    # Diverging symlog norm: linear within ±linthresh, log outside
    linthresh = slope_tol / 4.0
    diff_norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-abs_max, vmax=abs_max,
                                   base=10)

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
        diff_grid = slope_grid - gmm_slope

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
                cb.set_ticklabels([f"{gmm_slope + t:.2f}" for t in ticks])
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

    fig.suptitle(
        f"Mag-split grid  (GMM slope={gmm_slope:.3f}, tol={slope_tol})",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(run_dir, "mag_split_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved mag-split heatmap → {out_path}")


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
    parser.add_argument("--n_sigma_min", type=float, default=0.7,
                        help="Lower end of sweep n_σ grid")
    parser.add_argument("--n_sigma_max", type=float, default=1.7,
                        help="Upper end of sweep n_σ grid")
    parser.add_argument("--n_sigma_n",   type=int,   default=21,
                        help="Number of log-spaced sweep grid points")
    parser.add_argument("--z_obs_min", type=float, default=0.03,
                        help="Minimum redshift cut")
    parser.add_argument("--z_obs_max", type=float, default=0.10,
                        help="Maximum redshift cut")
    parser.add_argument("--n_sweep_objects", type=int, default=10000,
                        help="Subsample raw data to this many objects (0 = use all)")
    parser.add_argument("--slope_tol", type=float, default=0.5,
                        help="Tolerance |MLE slope - GMM slope| for fiducial search")
    # 2D fiducial search grid
    parser.add_argument("--n_sigma_perp_min", type=float, default=1.0,
                        help="Lower end of n_sigma_perp grid for fiducial search")
    parser.add_argument("--n_sigma_perp_max", type=float, default=4.0,
                        help="Upper end of n_sigma_perp grid for fiducial search")
    parser.add_argument("--n_sigma_perp_n",   type=int,   default=7,
                        help="Number of n_sigma_perp grid points")
    parser.add_argument("--n_sigma_mag_min",  type=float, default=0.5,
                        help="Lower end of n_sigma_mag grid for fiducial search")
    parser.add_argument("--n_sigma_mag_max",  type=float, default=3.0,
                        help="Upper end of n_sigma_mag grid for fiducial search")
    parser.add_argument("--n_sigma_mag_n",    type=int,   default=6,
                        help="Number of n_sigma_mag grid points")
    parser.add_argument("--fiducial_contraction", type=float, default=0.9,
                        help="Factor applied to best (n_sigma_perp, n_sigma_mag) to get fiducial (< 1 pulls slightly inward)")
    parser.add_argument("--n_fine_perp", type=int, default=10,
                        help="Number of fine-grid points along n_sigma_perp")
    parser.add_argument("--n_fine_mag",  type=int, default=8,
                        help="Number of fine-grid points along n_sigma_mag")
    parser.add_argument("--coarse_only", action="store_true",
                        help="Run only the coarse grid; skip fine-pass MLE")
    # Mag-split mode: vary haty_min and haty_max independently
    parser.add_argument("--mag_split", action="store_true",
                        help="Run mag-split 2D grid (vary n_σ_lo and n_σ_hi independently)")
    parser.add_argument("--mag_split_plot", action="store_true",
                        help="Replot mag-split grid from saved mag_split_grid.json (no Stan calls)")
    parser.add_argument("--n_sigma_perp", type=float, default=None,
                        help="Fixed n_σ_perp for --mag_split mode")
    parser.add_argument("--n_sigma_mag_lo_min", type=float, default=1.0,
                        help="Lower end of n_σ_lo grid (haty_min) for --mag_split")
    parser.add_argument("--n_sigma_mag_lo_max", type=float, default=4.0,
                        help="Upper end of n_σ_lo grid (haty_min) for --mag_split")
    parser.add_argument("--n_sigma_mag_lo_n",   type=int,   default=4,
                        help="Number of n_σ_lo grid points for --mag_split")
    parser.add_argument("--n_sigma_mag_hi_min", type=float, default=1.0,
                        help="Lower end of n_σ_hi grid (haty_max) for --mag_split")
    parser.add_argument("--n_sigma_mag_hi_max", type=float, default=4.0,
                        help="Upper end of n_σ_hi grid (haty_max) for --mag_split")
    parser.add_argument("--n_sigma_mag_hi_n",   type=int,   default=4,
                        help="Number of n_σ_hi grid points for --mag_split")

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

    n_sigma_vals = np.geomspace(args.n_sigma_min, args.n_sigma_max, args.n_sigma_n)
    extra_cuts   = {}
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

    # ── Mag-split mode ───────────────────────────────────────────────────────
    if args.mag_split_plot:
        results_by_perp, perp_vals, lo_vals, hi_vals = load_mag_split_results(run_dir)
        plot_mag_split_grid(results_by_perp, perp_vals, lo_vals, hi_vals,
                            gmm_slope, args.slope_tol, run_dir)
        import sys; sys.exit(0)

    if args.mag_split:
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
        plot_mag_split_grid(
            mag_split_results, perp_vals, lo_vals, hi_vals,
            gmm_slope, args.slope_tol, run_dir)
        import sys; sys.exit(0)

    # ── Standard fiducial search ─────────────────────────────────────────────
    print(f"Running ellipse sweep: {args.n_sigma_n} n_σ values in "
          f"[{args.n_sigma_min}, {args.n_sigma_max}], exe={exe_file}")

    # 2D fiducial search: find best (n_sigma_perp, n_sigma_mag) pair
    n_sigma_perp_vals = np.linspace(
        args.n_sigma_perp_min, args.n_sigma_perp_max, args.n_sigma_perp_n)
    n_sigma_mag_vals  = np.linspace(
        args.n_sigma_mag_min, args.n_sigma_mag_max, args.n_sigma_mag_n)

    (n_sigma_perp, n_sigma_mag, ref_cuts, ref_slope,
     grid_info, raw_best_perp, raw_best_mag) = find_fiducial_cuts(
        raw_data, mu, sigma, extra_cuts, exe_file, gmm_slope,
        n_sigma_perp_vals, n_sigma_mag_vals, slope_tol=args.slope_tol,
        contraction=args.fiducial_contraction, coarse_only=args.coarse_only,
        n_fine_perp=args.n_fine_perp, n_fine_mag=args.n_fine_mag)

    plot_fiducial_search(
        grid_info, gmm_slope, args.slope_tol,
        raw_best_perp, raw_best_mag,
        n_sigma_perp, n_sigma_mag,
        run_dir)

    if args.coarse_only:
        print("  (--coarse_only: skipping 1D ellipse sweep)")
    else:
        # Ensure both fiducial n_σ values are in the sweep grid
        for fiducial_ns in (n_sigma_perp, n_sigma_mag):
            if not np.any(np.isclose(n_sigma_vals, fiducial_ns)):
                n_sigma_vals = np.sort(np.append(n_sigma_vals, fiducial_ns))

        sweep_results = run_ellipse_sweep(
            raw_data, mu, sigma, n_sigma_vals, extra_cuts, exe_file,
            ref_cuts=ref_cuts)

        save_sweep_results(sweep_results, run_dir)

        ref_n_sigma_by_param = {
            "haty_min":         n_sigma_mag,
            "haty_max":         n_sigma_mag,
            "intercept_plane":  n_sigma_perp,
            "intercept_plane2": n_sigma_perp,
        }
        plot_ellipse_partial_deriv(sweep_results, run_dir,
                                   ref_n_sigma_by_param=ref_n_sigma_by_param)

    # Summary table
    _SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]
    print("\n" + "=" * 60)
    print(f"ELLIPSE SWEEP SUMMARY")
    print(f"  fiducial: n_sigma_perp={n_sigma_perp:.2f}  n_sigma_mag={n_sigma_mag:.2f}")
    print("=" * 60)
    print(f"  GMM slope (Gaussian fit): {gmm_slope:.4f}")
    if ref_slope is not None:
        print(f"  MLE slope at fiducial:    {ref_slope:.4f}")
    if not args.coarse_only:
        for p in _SWEEP_PARAMS:
            records = sweep_results[p]
            # Find record whose cut_value is closest to the fiducial cut value
            ref_cut_val = ref_cuts[p]
            idx_ref = int(np.argmin(np.abs(np.array([r[1] for r in records]) - ref_cut_val)))
            _, cut_val, slope_ref, N_ref = records[idx_ref]
            ns_arr = np.array([r[0] for r in records])
            sl_arr = np.array([r[2] if r[2] is not None else np.nan
                               for r in records])
            ref_ns = ref_n_sigma_by_param[p]
            valid  = np.isfinite(sl_arr)
            if valid.sum() >= 2:
                deriv_at_ref = float(np.gradient(sl_arr[valid], ns_arr[valid])[
                    np.argmin(np.abs(ns_arr[valid] - ref_ns))])
            else:
                deriv_at_ref = float("nan")
            print(f"  {p:22s}  cut={cut_val:7.3f}  "
                  f"slope={slope_ref if slope_ref is not None else float('nan'):7.4f}  "
                  f"ds/dn_σ={deriv_at_ref:+.4f}  N={N_ref}")
    print("=" * 60)
