#!/usr/bin/env python3
"""
mle_slope_sweep.py — MLE-slope-centred selection-cut sweep for TFR fitting.

The oblique strip is centred on the MLE line (using mle_slope / mle_intercept
from output/<run>/mle_nsigma.json) rather than on the GMM ellipse orientation.
The sweep varies strip width (n_sigma_perp) and magnitude limits (n_sigma_lo,
n_sigma_hi) independently over a 3-D grid.

Operating modes
───────────────
Default (no flag):
  Run the full grid → save output/<run>/mle_slope_grid.json → plot.

--plot_only:
  Reload saved JSON and replot without any Stan calls.

Outputs
───────
  output/<run>/mle_slope_grid.json   — grid results
  output/<run>/mle_slope_grid.png    — (n_perp × 2) heatmap

Prerequisites
─────────────
  output/<run>/selection_ellipse.json   (from selection_ellipse.py)
  output/<run>/mle_nsigma.json          (from selection_ellipse.py --exe …)

Usage
─────
  # Run sweep
  python mle_slope_sweep.py --source DESI --run DESI --exe ./tophat \\
      --n_sigma_perp_min 2 --n_sigma_perp_max 8 --n_sigma_perp_n 4 \\
      --n_sigma_mag_lo_min 2 --n_sigma_mag_lo_max 5 --n_sigma_mag_lo_n 4 \\
      --n_sigma_mag_hi_min 2 --n_sigma_mag_hi_max 5 --n_sigma_mag_hi_n 4

  # Replot from saved JSON (no Stan calls)
  python mle_slope_sweep.py --source DESI --run DESI --exe ./tophat --plot_only
"""

import argparse
import glob
import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ellipse_sweep import (
    load_fullmocks,
    load_desi,
    _build_stan_dicts,
    _stan_mle_slope,
    _cuts_at_nsigma,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: GEOMETRY — MLE-line-centred strip
# ─────────────────────────────────────────────────────────────────────────────

def _cuts_mle_strip(mu, sigma, mle_slope, mle_intercept,
                    n_sigma_perp, n_sigma_lo, n_sigma_hi):
    """Selection cuts with the oblique strip centred on the MLE line.

    Strip geometry:
      slope_plane = mle_slope  (MLE TFR slope — this is the defining
                                difference from ellipse_sweep, which uses
                                the GMM multimodal-fit slope)
      The two bounding lines are at perpendicular distance
        n_sigma_perp * sigma_minor
      from the MLE line, giving an intercept half-separation of
        n_sigma_perp * sigma_minor * sqrt(1 + mle_slope^2).

    Magnitude limits use the GMM y-extent (1σ ellipse), scaled independently:
      haty_min = mu[1] - n_sigma_lo * y_extent
      haty_max = mu[1] + n_sigma_hi * y_extent

    Returns dict: haty_min, haty_max, slope_plane, intercept_plane,
                  intercept_plane2.
    """
    vals, vecs = np.linalg.eigh(sigma)
    sigma_minor = np.sqrt(vals[0])
    sigma_major = np.sqrt(vals[1])
    angle_rad   = np.arctan2(vecs[1, -1], vecs[0, -1])

    # y-extent of 1σ ellipse (for magnitude limits)
    y_extent = np.sqrt(sigma_major**2 * np.sin(angle_rad)**2
                       + sigma_minor**2 * np.cos(angle_rad)**2)

    haty_min = float(mu[1]) - n_sigma_lo * float(y_extent)
    haty_max = float(mu[1]) + n_sigma_hi * float(y_extent)

    # Perpendicular half-separation in intercept units:
    #   perp_dist = |Δintercept| / sqrt(1 + mle_slope^2)
    ip_half_sep = n_sigma_perp * float(sigma_minor) * np.sqrt(1.0 + mle_slope**2)
    intercept_plane  = mle_intercept - ip_half_sep
    intercept_plane2 = mle_intercept + ip_half_sep

    return dict(
        haty_min=haty_min,
        haty_max=haty_max,
        slope_plane=float(mle_slope),
        intercept_plane=float(intercept_plane),
        intercept_plane2=float(intercept_plane2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GRID RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_mle_slope_grid(raw_data, mu, sigma, mle_slope, mle_intercept,
                       extra_cuts, exe_file, ref_slope,
                       perp_vals, lo_vals, hi_vals):
    """3-D grid over (n_sigma_perp, n_sigma_lo, n_sigma_hi).

    Returns dict {n_sigma_perp: [(nlo, nhi, slope, N), ...]}.
    ref_slope is used only for the difference column in the progress table.
    """
    results_by_perp = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for n_sigma_perp in perp_vals:
            results = []
            print(f"\n  [mle-slope-sweep]  n_σ_perp={n_sigma_perp:.3f}")
            print(f"  {'n_σ_lo':>7}  {'n_σ_hi':>7}  {'MLE slope':>10}  "
                  f"{'diff':>8}  {'N':>6}")
            print(f"  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*8}  {'-'*6}")
            for nlo in lo_vals:
                for nhi in hi_vals:
                    cuts = _cuts_mle_strip(
                        mu, sigma, mle_slope, mle_intercept,
                        float(n_sigma_perp), float(nlo), float(nhi),
                    )
                    cuts.update(extra_cuts)
                    data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
                    if data_dict is None:
                        print(f"  {nlo:7.2f}  {nhi:7.2f}  {'—':>10}  "
                              f"{'—':>8}  {'<30':>6}")
                        results.append((float(nlo), float(nhi), float("nan"), 0))
                        continue
                    slope = _stan_mle_slope(data_dict, init_dict, exe_file, tmp_dir)
                    N = data_dict["N_total"]
                    if slope is None:
                        print(f"  {nlo:7.2f}  {nhi:7.2f}  {'failed':>10}  "
                              f"{'—':>8}  {N:6d}")
                        results.append((float(nlo), float(nhi), float("nan"), N))
                        continue
                    diff = slope - ref_slope
                    print(f"  {nlo:7.2f}  {nhi:7.2f}  {slope:10.4f}  "
                          f"  {diff:+8.4f}  {N:6d}")
                    results.append((float(nlo), float(nhi), slope, N))
            results_by_perp[float(n_sigma_perp)] = results
    return results_by_perp


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_mle_slope_results(results_by_perp, perp_vals, lo_vals, hi_vals, run_dir):
    """Save grid results to output/<run>/mle_slope_grid.json."""
    out = {
        "perp_vals": list(perp_vals),
        "lo_vals":   list(lo_vals),
        "hi_vals":   list(hi_vals),
        "results_by_perp": {
            str(k): v for k, v in results_by_perp.items()
        },
    }
    out_path = os.path.join(run_dir, "mle_slope_grid.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved mle-slope results → {out_path}")


def load_mle_slope_results(run_dir):
    """Load grid results from output/<run>/mle_slope_grid.json."""
    in_path = os.path.join(run_dir, "mle_slope_grid.json")
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
# SECTION 4: PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_mle_slope_grid(results_by_perp, perp_vals, lo_vals, hi_vals,
                        mle_slope, mle_intercept, slope_tol, run_dir,
                        mu=None, sigma=None):
    """Save (n_perp × 2) heatmap to output/<run>/mle_slope_grid.png.

    Each row corresponds to a fixed n_σ_perp.
    x-axis: n_σ_hi (haty_max), y-axis: n_σ_lo (haty_min).
    Left column: MLE slope with tolerance contour. Right column: N.
    """
    import matplotlib.colors as mcolors

    lo_arr = np.asarray(lo_vals)
    hi_arr = np.asarray(hi_vals)
    n_perp = len(perp_vals)

    all_slopes = [s for rs in results_by_perp.values()
                  for _, _, s, _ in rs if np.isfinite(s)]
    all_N      = [float(N) for rs in results_by_perp.values()
                  for _, _, _, N in rs if N > 0]
    N_vmin = min(all_N) if all_N else 0
    N_vmax = max(all_N) if all_N else 1

    # Centre diverging cmap on mean of cells within slope_tol of mle_slope
    good_slopes = [s for s in all_slopes if abs(s - mle_slope) <= slope_tol]
    center      = float(np.mean(good_slopes)) if good_slopes else mle_slope

    all_diffs = [s - center for s in all_slopes]
    abs_max   = min(max(abs(d) for d in all_diffs) if all_diffs else 1.0,
                    2.0 * slope_tol)

    linthresh = slope_tol / 4.0
    diff_norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-abs_max, vmax=abs_max,
                                   base=10)

    # Secondary axis labels (actual magnitude values)
    if mu is not None and sigma is not None:
        unit      = _cuts_at_nsigma(mu, sigma, 1.0)
        y_extent  = unit["haty_max"] - float(mu[1])
        haty_max_vals = float(mu[1]) + hi_arr * y_extent
        haty_min_vals = float(mu[1]) - lo_arr * y_extent
    else:
        haty_max_vals = None
        haty_min_vals = None

    fig, axes = plt.subplots(n_perp, 2, figsize=(11, 4.5 * n_perp),
                             squeeze=False)

    for row, n_sigma_perp in enumerate(perp_vals):
        results    = results_by_perp[float(n_sigma_perp)]
        slope_grid = np.full((len(lo_arr), len(hi_arr)), np.nan)
        N_grid     = np.full((len(lo_arr), len(hi_arr)), np.nan)
        for nlo, nhi, slope, N in results:
            il = int(np.argmin(np.abs(lo_arr - nlo)))
            ih = int(np.argmin(np.abs(hi_arr - nhi)))
            slope_grid[il, ih] = slope
            N_grid[il, ih]     = float(N) if N > 0 else np.nan
        diff_grid = slope_grid - center

        for col, (data_grid, cmap_name, norm, cbar_label) in enumerate([
                (diff_grid, "RdBu",    diff_norm,
                 "MLE slope"),
                (N_grid,    "viridis", mcolors.Normalize(vmin=N_vmin, vmax=N_vmax),
                 "N")]):
            ax = axes[row, col]
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("lightgray")
            pcm = ax.pcolormesh(hi_arr, lo_arr, data_grid, cmap=cmap, norm=norm,
                                shading="nearest")
            cb = fig.colorbar(pcm, ax=ax, label=cbar_label)
            if col == 0:
                ticks = list(cb.get_ticks())
                cb.set_ticks(ticks)
                cb.set_ticklabels([f"{center + t:.2f}" for t in ticks])
                try:
                    ax.contour(hi_arr, lo_arr, np.abs(diff_grid),
                               levels=[slope_tol], colors="white", linewidths=1.2)
                except Exception:
                    pass
            ax.set_xlabel(r"$n_{\sigma,\hat{y}_\text{max}}$")
            ax.set_ylabel(r"$n_{\sigma,\hat{y}_\text{min}}$")
            col_title = "MLE slope" if col == 0 else "N"
            ax.set_title(f"$n_{{\\sigma,\\perp}}={n_sigma_perp:.1f}$  —  {col_title}")
            if haty_max_vals is not None:
                ax_top = ax.twiny()
                ax_top.set_xlim(ax.get_xlim())
                ax_top.set_xticks(hi_arr)
                ax_top.set_xticklabels([f"{v:.1f}" for v in haty_max_vals],
                                       fontsize=7, rotation=45, ha="left")
                ax_top.set_xlabel(r"$\hat{y}_\text{max}$", fontsize=8)
            if haty_min_vals is not None:
                ax_right = ax.twinx()
                ax_right.set_ylim(ax.get_ylim())
                ax_right.set_yticks(lo_arr)
                ax_right.set_yticklabels([f"{v:.1f}" for v in haty_min_vals],
                                         fontsize=7)
                ax_right.set_ylabel(r"$\hat{y}_\text{min}$", fontsize=8)

    fig.suptitle(
        f"MLE-slope grid  "
        f"(mle_slope={mle_slope:.3f}, mle_intercept={mle_intercept:.3f}, "
        f"color center={center:.3f}, tol={slope_tol})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = os.path.join(run_dir, "mle_slope_grid.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved mle-slope heatmap → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sweep (n_σ_perp, n_σ_lo, n_σ_hi) with the oblique strip centred "
            "on the MLE line (mle_slope from mle_nsigma.json)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["fullmocks", "DESI"], default="fullmocks",
                        help="Data source")
    parser.add_argument("--fits_file", default=None,
                        help="Path to a single FITS file; auto-detected from --dir if omitted")
    parser.add_argument("--dir", default="data/",
                        help="Directory searched for FITS files when --fits_file is omitted")
    parser.add_argument("--run", required=True,
                        help="Run name; reads output/<run>/selection_ellipse.json "
                             "and output/<run>/mle_nsigma.json")
    parser.add_argument("--exe", default="tophat",
                        help="Path to compiled Stan tophat executable")
    parser.add_argument("--z_obs_min", type=float, default=0.03,
                        help="Minimum redshift cut")
    parser.add_argument("--z_obs_max", type=float, default=0.10,
                        help="Maximum redshift cut")
    parser.add_argument("--n_sweep_objects", type=int, default=10000,
                        help="Subsample raw data to this many objects (0 = use all)")
    parser.add_argument("--slope_tol", type=float, default=0.5,
                        help="Tolerance |MLE slope - ref slope| for contour and colour centre")
    parser.add_argument("--plot_only", action="store_true",
                        help="Reload saved mle_slope_grid.json and replot without Stan calls")
    # Grid range
    parser.add_argument("--n_sigma_perp_min", type=float, default=2.0)
    parser.add_argument("--n_sigma_perp_max", type=float, default=8.0)
    parser.add_argument("--n_sigma_perp_n",   type=int,   default=4)
    parser.add_argument("--n_sigma_mag_lo_min", type=float, default=2.0)
    parser.add_argument("--n_sigma_mag_lo_max", type=float, default=5.0)
    parser.add_argument("--n_sigma_mag_lo_n",   type=int,   default=4)
    parser.add_argument("--n_sigma_mag_hi_min", type=float, default=2.0)
    parser.add_argument("--n_sigma_mag_hi_max", type=float, default=5.0)
    parser.add_argument("--n_sigma_mag_hi_n",   type=int,   default=4)

    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # ── Load ellipse JSON ─────────────────────────────────────────────────────
    ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
    if not os.path.exists(ellipse_path):
        raise FileNotFoundError(
            f"{ellipse_path} not found — run selection_ellipse.py first.")
    with open(ellipse_path) as f:
        ell = json.load(f)
    mu    = np.array(ell["mean"])
    sigma = np.array(ell["covariance"])

    # ── Load MLE JSON ─────────────────────────────────────────────────────────
    mle_path = os.path.join(run_dir, "mle_nsigma.json")
    if not os.path.exists(mle_path):
        raise FileNotFoundError(
            f"{mle_path} not found — run selection_ellipse.py --exe … first.")
    with open(mle_path) as f:
        mle_data = json.load(f)
    mle_slope     = float(mle_data["mle_slope"])
    mle_intercept = float(mle_data["mle_intercept"])
    print(f"Loaded MLE: slope={mle_slope:.4f}  intercept={mle_intercept:.4f}"
          f"  (from n_sigma={mle_data.get('n_sigma', '?')})")

    # ── Plot-only mode ────────────────────────────────────────────────────────
    if args.plot_only:
        results_by_perp, perp_vals, lo_vals, hi_vals = load_mle_slope_results(run_dir)
        plot_mle_slope_grid(
            results_by_perp, perp_vals, lo_vals, hi_vals,
            mle_slope, mle_intercept, args.slope_tol, run_dir,
            mu=mu, sigma=sigma,
        )
        import sys; sys.exit(0)

    # ── Load raw data ─────────────────────────────────────────────────────────
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

    # ── Build grid ────────────────────────────────────────────────────────────
    perp_vals = np.linspace(args.n_sigma_perp_min, args.n_sigma_perp_max,
                            args.n_sigma_perp_n)
    lo_vals   = np.linspace(args.n_sigma_mag_lo_min, args.n_sigma_mag_lo_max,
                            args.n_sigma_mag_lo_n)
    hi_vals   = np.linspace(args.n_sigma_mag_hi_min, args.n_sigma_mag_hi_max,
                            args.n_sigma_mag_hi_n)

    print(f"Running MLE-slope grid:")
    print(f"  n_σ_perp : {perp_vals}")
    print(f"  n_σ_lo   : {lo_vals}")
    print(f"  n_σ_hi   : {hi_vals}")
    print(f"  exe      : {exe_file}")

    results_by_perp = run_mle_slope_grid(
        raw_data, mu, sigma, mle_slope, mle_intercept,
        extra_cuts, exe_file, mle_slope,
        perp_vals, lo_vals, hi_vals,
    )
    save_mle_slope_results(results_by_perp, perp_vals, lo_vals, hi_vals, run_dir)
    plot_mle_slope_grid(
        results_by_perp, perp_vals, lo_vals, hi_vals,
        mle_slope, mle_intercept, args.slope_tol, run_dir,
        mu=mu, sigma=sigma,
    )
