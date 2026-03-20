#!/usr/bin/env python3
"""
selection_criteria.py — Automatically identify the stability plateau in each
ellipse-sweep profile and choose selection-cut values near its large-n_σ edge.

Reads:
  output/<run>/ellipse_sweep.json     — sweep slopes and derivatives per parameter
  output/<run>/selection_ellipse.json — for slope_plane (fixed, not swept)

Writes:
  output/<run>/selection_criteria.json — chosen cut values
  output/<run>/selection_criteria.png  — annotated sweep plot

Usage:
  python selection_criteria.py --run <name> [--d_threshold 0.3] [--frac 0.8]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (shared with ellipse_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────

_SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]

_PARAM_LABELS = {
    "haty_max":         r"$\hat{y}_{\max}$",
    "haty_min":         r"$\hat{y}_{\min}$",
    "slope_plane":      r"$\bar{s}$ (plane slope)",
    "intercept_plane":  r"$c_1$ (lower intercept)",
    "intercept_plane2": r"$c_2$ (upper intercept)",
}

# Stan model hard bounds on slope
_SLOPE_BOUND_LO = -9.0
_SLOPE_BOUND_HI = -4.0
_BOUND_TOL = 0.05


def _label(p) -> str:
    return _PARAM_LABELS.get(p) or p


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM: choose n_sigma for one parameter
# ─────────────────────────────────────────────────────────────────────────────

def _contiguous_groups(indices):
    """Split a sorted integer array into lists of consecutive index runs."""
    if len(indices) == 0:
        return []
    groups, cur = [], [indices[0]]
    for idx in indices[1:]:
        if idx == cur[-1] + 1:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)
    return groups


def choose_nsigma(ns_arr, cut_vals, slopes, derivs, d_threshold=0.3, frac=0.8):
    """Choose n_sigma for one sweep parameter based on derivative plateau.

    Two-level plateau detection:
      1. Broad plateau: valid points with |deriv| < d_threshold * max|deriv|.
      2. Select the contiguous group containing the global minimum |deriv|.
      3. Within that group, a point is "close to zero" if
         |deriv| < d_threshold * IQR(derivs in group), where IQR measures the
         width of the derivative distribution.  Take the RIGHTMOST contiguous
         sub-run of such points.
      4. edge_ns = rightmost point of that sub-run; chosen_ns interpolated via frac.

    Parameters
    ----------
    ns_arr     : 1-D array of n_sigma grid points
    cut_vals   : 1-D array of cut values at each grid point
    slopes     : 1-D array of MLE slopes (NaN where unavailable)
    derivs     : 1-D array of ∂s/∂n_σ (NaN where unavailable)
    d_threshold: fraction of max|deriv| for Level 1; fraction of IQR(deriv)
                 for Level 3 "close to zero" criterion
    frac       : interpolation fraction between preceding sub-run point and edge

    Returns
    -------
    dict with keys:
      chosen_ns   : float
      cut_value   : float
      slope       : float or NaN
      status      : str  ('ok' | 'plateau_at_edge' | 'no_plateau' | 'insufficient_data')
      threshold   : float or NaN  (global threshold, for plotting)
    """
    ns_arr   = np.asarray(ns_arr,   dtype=float)
    cut_vals = np.asarray(cut_vals, dtype=float)
    slopes   = np.asarray(slopes,   dtype=float)
    derivs   = np.asarray(derivs,   dtype=float)

    # Valid: finite slope, finite derivative, slope not hitting Stan bounds
    valid_mask = (
        np.isfinite(slopes)
        & np.isfinite(derivs)
        & (slopes > _SLOPE_BOUND_LO + _BOUND_TOL)
        & (slopes < _SLOPE_BOUND_HI - _BOUND_TOL)
    )

    if valid_mask.sum() < 2:
        chosen_ns = 1.0
        cut_value = float(np.interp(chosen_ns, ns_arr, cut_vals))
        idx_near  = int(np.argmin(np.abs(ns_arr - chosen_ns)))
        return dict(
            chosen_ns=chosen_ns,
            cut_value=cut_value,
            slope=float(slopes[idx_near]) if np.isfinite(slopes[idx_near]) else float("nan"),
            status="insufficient_data",
            threshold=float("nan"),
        )

    abs_derivs_valid = np.abs(derivs[valid_mask])
    threshold = d_threshold * float(np.max(abs_derivs_valid))

    # ── Level 1: broad plateau ──────────────────────────────────────────────
    plateau_mask = valid_mask & (np.abs(derivs) < threshold)

    if not plateau_mask.any():
        valid_ns  = ns_arr[valid_mask]
        chosen_ns = float(valid_ns[np.argmin(np.abs(valid_ns - 1.0))])
        cut_value = float(np.interp(chosen_ns, ns_arr, cut_vals))
        idx_near  = int(np.argmin(np.abs(ns_arr - chosen_ns)))
        return dict(
            chosen_ns=chosen_ns,
            cut_value=cut_value,
            slope=float(slopes[idx_near]) if np.isfinite(slopes[idx_near]) else float("nan"),
            status="no_plateau",
            threshold=threshold,
        )

    # ── Level 2: select the contiguous group containing global min |deriv| ──
    penalized    = np.where(valid_mask, np.abs(derivs), np.inf)
    global_min_i = int(np.argmin(penalized))

    broad_groups = _contiguous_groups(np.where(plateau_mask)[0].tolist())
    selected     = broad_groups[-1]                  # fallback: last group
    for g in broad_groups:
        if global_min_i in g:
            selected = g
            break

    # ── Level 3: local threshold within selected group ──────────────────────
    # If the global min sits at the leftmost valid point the derivative profile
    # is monotonically drifting (no interior stable region).  Applying a local
    # threshold would collapse the selection to the boundary; instead fall back
    # to the Level-1 edge (rightmost point of the selected broad-plateau group).
    valid_indices   = np.where(valid_mask)[0]
    boundary_effect = (global_min_i == int(valid_indices[0]))

    if not boundary_effect:
        sel_derivs     = derivs[selected]
        sel_abs_derivs = np.abs(sel_derivs)
        iqr = float(np.percentile(sel_derivs, 75) - np.percentile(sel_derivs, 25))
        if iqr < 1e-10:                              # degenerate: all derivs identical
            iqr = float(np.median(sel_abs_derivs)) or 1.0
        local_threshold = d_threshold * iqr
        local_plateau   = (sel_abs_derivs < local_threshold).tolist()

        local_groups = _contiguous_groups(
            [j for j, v in enumerate(local_plateau) if v]
        )
        # Use the rightmost sub-group: loosest cut still close to zero
        sub = local_groups[-1] if local_groups else [int(np.argmin(sel_abs_derivs))]

        edge_global_i = selected[sub[-1]]
        edge_ns       = float(ns_arr[edge_global_i])
        use_sub       = sub
    else:
        # Boundary fallback: use right edge of selected broad-plateau group
        edge_global_i = selected[-1]
        edge_ns       = float(ns_arr[edge_global_i])
        use_sub       = list(range(len(selected)))   # treat whole group as sub

    # Check if edge is at the rightmost valid point
    valid_ns = ns_arr[valid_mask]
    at_edge  = (edge_ns >= float(np.max(valid_ns)) - 1e-10)
    status   = "plateau_at_edge" if at_edge else "ok"

    # Interpolate chosen_ns between the sub-run's last two points
    if len(use_sub) == 1:
        chosen_ns = edge_ns
    else:
        prev_global_i = selected[use_sub[-2]]
        prev_ns       = float(ns_arr[prev_global_i])
        chosen_ns     = prev_ns + frac * (edge_ns - prev_ns)

    cut_value = float(np.interp(chosen_ns, ns_arr, cut_vals))
    idx_near  = int(np.argmin(np.abs(ns_arr - chosen_ns)))
    slope_at  = float(slopes[idx_near]) if np.isfinite(slopes[idx_near]) else float("nan")

    return dict(
        chosen_ns=chosen_ns,
        cut_value=cut_value,
        slope=slope_at,
        status=status,
        threshold=threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_selection_criteria(sweep_data, results, run_dir):
    """2-row × 4-column annotated sweep figure saved to selection_criteria.png."""
    n_params = len(_SWEEP_PARAMS)
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))

    for col, p in enumerate(_SWEEP_PARAMS):
        entry    = sweep_data[p]
        ns_arr   = np.array(entry["n_sigma"],          dtype=float)
        cut_vals = np.array(entry["cut_values"],        dtype=float)
        slopes   = np.array([v if v is not None else np.nan
                             for v in entry["slopes"]], dtype=float)
        derivs   = np.array([v if v is not None else np.nan
                             for v in entry["d_slope_d_nsigma"]], dtype=float)
        res      = results[p]

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        valid = np.isfinite(slopes)

        # ── Top row: slope vs n_σ ──────────────────────────────────────────
        ax_top.plot(ns_arr[valid], slopes[valid], "o-", color="steelblue",
                    linewidth=1.5, markersize=4)
        ax_top.set_xscale("log")
        ax_top.axvline(1.0, color="gray", linestyle="--", linewidth=1.0,
                       label=r"$n_\sigma=1$")
        ax_top.axvline(res["chosen_ns"], color="orange", linestyle="--",
                       linewidth=1.5, label=f"chosen $n_\\sigma$={res['chosen_ns']:.2f}")
        ax_top.set_xlabel(r"$n_\sigma$", fontsize=10)
        ax_top.set_ylabel("MLE slope", fontsize=10)

        title = _label(p)
        if res["status"] != "ok":
            warning_map = {
                "plateau_at_edge":    "plateau at sweep edge — increase n_sigma_max",
                "no_plateau":         "no plateau found",
                "insufficient_data":  "insufficient data",
            }
            title += f"\n[{warning_map.get(res['status']) or res['status']}]"
        ax_top.set_title(title, fontsize=10)
        ax_top.grid(True, alpha=0.3, which="both")
        ax_top.legend(fontsize=7)

        # Secondary top x-axis: cut value labels
        n_ticks  = min(5, len(ns_arr))
        tick_idx = np.round(np.linspace(0, len(ns_arr) - 1, n_ticks)).astype(int)
        ax_top2  = ax_top.twiny()
        ax_top2.set_xscale("log")
        ax_top2.set_xlim(ax_top.get_xlim())
        ax_top2.set_xticks(ns_arr[tick_idx])
        ax_top2.set_xticklabels([f"{cut_vals[i]:.2f}" for i in tick_idx],
                                fontsize=7, rotation=45)

        # ── Bottom row: derivative vs n_σ ─────────────────────────────────
        valid_d = np.isfinite(derivs)
        if valid_d.any():
            ax_bot.plot(ns_arr[valid_d], derivs[valid_d], "o-", color="purple",
                        linewidth=1.5, markersize=4)

        ax_bot.set_xscale("log")
        ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_bot.axvline(1.0, color="gray", linestyle="--", linewidth=1.0,
                       label=r"$n_\sigma=1$")
        ax_bot.axvline(res["chosen_ns"], color="orange", linestyle="--",
                       linewidth=1.5, label=f"chosen $n_\\sigma$={res['chosen_ns']:.2f}")

        ax_bot.set_xlabel(r"$n_\sigma$", fontsize=10)
        ax_bot.set_ylabel(r"$\partial s / \partial (n_\sigma)$", fontsize=10)
        ax_bot.grid(True, alpha=0.3, which="both")
        ax_bot.legend(fontsize=7)

    fig.suptitle(
        r"Selection criteria: plateau detection for each cut parameter",
        fontsize=13)
    plt.tight_layout()
    out_file = os.path.join(run_dir, "selection_criteria.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA OVERPLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_cuts_on_data(raw_data, ell, criteria, run_dir):
    """Reproduce selection_ellipse.png with selected cuts overplotted.

    Galaxy scatter is coloured by the core-component Gaussian probability
    (exp(-½ χ²)) — a proxy for P(core) computed from the saved covariance
    without refitting the full GMM.

    1σ cuts from selection_ellipse.json: cyan / lime  (same as selection_ellipse.py)
    Selected cuts from selection_criteria.json:        magenta / yellow

    Saved to output/<run>/selection_criteria_data.png.
    """
    x  = raw_data["x"]
    y  = raw_data["y"]
    xy = np.column_stack([x, y])

    mu        = np.array(ell["mean"])
    sigma     = np.array(ell["covariance"])
    semi_axes = np.array(ell["semi_axes"])   # [minor, major]
    angle_deg = float(ell["angle_deg"])
    x_hi      = float(ell["x_trunc"])
    y_lo      = float(ell["y_trunc"])

    # Proxy for P(core): exp(-½ χ²) using the core-component covariance
    diff      = xy - mu
    sigma_inv = np.linalg.inv(sigma)
    chi2      = np.einsum("ij,jk,ik->i", diff, sigma_inv, diff)
    core_prob = np.exp(-0.5 * chi2)

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(x, y, c=core_prob, cmap="coolwarm",
                    s=1, alpha=0.4, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label="exp(−½χ²) [proxy for P(core)]")

    # Ellipses: 1σ / 2σ / 3σ  (same palette as selection_ellipse.py)
    for n_sigma, color, ls in zip([1, 2, 3],
                                  ["gold", "orange", "red"],
                                  ["-",    "--",     ":"  ]):
        ell_patch = mpatches.Ellipse(
            (float(mu[0]), float(mu[1])),
            width=2 * n_sigma * semi_axes[1],
            height=2 * n_sigma * semi_axes[0],
            angle=angle_deg,
            edgecolor=color, facecolor="none",
            linewidth=1.5, linestyle=ls,
            label=f"{n_sigma}σ ellipse", zorder=5,
        )
        ax.add_patch(ell_patch)

    # Truncation boundaries
    ax.axvline(x_hi, color="white", lw=1.0, ls="-",
               label=f"x_trunc={x_hi:.3f}")
    ax.axhline(y_lo, color="white", lw=1.0, ls="-",
               label=f"y_trunc={y_lo:.2f}")

    x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 300)

    # Selected cuts (cyan / lime — same palette as selection_ellipse.py)
    hm_s  = float(criteria["haty_max"])
    hmi_s = float(criteria["haty_min"])
    sp_s  = float(criteria["slope_plane"])
    ip1_s = float(criteria["intercept_plane"])
    ip2_s = float(criteria["intercept_plane2"])
    ax.axhline(hm_s,  color="deepskyblue", lw=2.5, ls="--",
               label=f"$\\hat{{y}}_{{\\max}}$={hm_s:.2f}")
    ax.axhline(hmi_s, color="deepskyblue", lw=2.5, ls=":",
               label=f"$\\hat{{y}}_{{\\min}}$={hmi_s:.2f}")
    ax.plot(x_line, sp_s * x_line + ip1_s, color="limegreen", lw=2.5, ls="--",
            label=f"$c_1$={ip1_s:.2f}")
    ax.plot(x_line, sp_s * x_line + ip2_s, color="limegreen", lw=2.5, ls=":",
            label=f"$c_2$={ip2_s:.2f}")

    ax.set_xlabel(r"$x = \log_{10}(V/100\,\mathrm{km\,s}^{-1})$")
    ax.set_ylabel(r"$y = R\mathrm{-band\ absolute\ magnitude}$")
    ax.set_title("Selected cuts overlaid on galaxy distribution")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.invert_yaxis()

    fig.tight_layout()
    out_file = os.path.join(run_dir, "selection_criteria_data.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-select TFR cut values from ellipse sweep plateau.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run",         required=True,
                        help="Run name; reads/writes output/<run>/")
    parser.add_argument("--d_threshold", type=float, default=0.3,
                        help="Plateau threshold as fraction of max|∂s/∂n_σ|")
    parser.add_argument("--frac",        type=float, default=0.8,
                        help="Interpolation fraction between prev plateau point and edge")
    # Optional: load raw data to produce selection_criteria_data.png
    parser.add_argument("--source",    choices=["fullmocks", "DESI"], default=None,
                        help="Data source for overplot (omit to skip data plot)")
    parser.add_argument("--fits_file", default=None,
                        help="Path to FITS file (auto-detected from --dir if omitted)")
    parser.add_argument("--dir",       default="data/",
                        help="Directory searched for FITS files when --fits_file is omitted")
    args = parser.parse_args()

    run_dir = os.path.join("output", args.run)

    # Load sweep data
    sweep_path = os.path.join(run_dir, "ellipse_sweep.json")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(
            f"{sweep_path} not found — run ellipse_sweep.py first.")
    with open(sweep_path) as f:
        sweep_data = json.load(f)

    # Load slope_plane from selection_ellipse.json
    ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
    if not os.path.exists(ellipse_path):
        raise FileNotFoundError(
            f"{ellipse_path} not found — run selection_ellipse.py first.")
    with open(ellipse_path) as f:
        ell = json.load(f)
    slope_plane = float(ell["slope_plane"])

    # Run plateau detection for each parameter
    results = {}
    for p in _SWEEP_PARAMS:
        entry  = sweep_data[p]
        ns_arr   = np.array(entry["n_sigma"],   dtype=float)
        cut_vals = np.array(entry["cut_values"], dtype=float)
        slopes   = np.array([v if v is not None else np.nan
                             for v in entry["slopes"]], dtype=float)
        derivs   = np.array([v if v is not None else np.nan
                             for v in entry["d_slope_d_nsigma"]], dtype=float)
        results[p] = choose_nsigma(
            ns_arr, cut_vals, slopes, derivs,
            d_threshold=args.d_threshold, frac=args.frac,
        )

    # Build output JSON
    out_json = {
        "haty_min":          results["haty_min"]["cut_value"],
        "haty_max":          results["haty_max"]["cut_value"],
        "slope_plane":       slope_plane,
        "intercept_plane":   results["intercept_plane"]["cut_value"],
        "intercept_plane2":  results["intercept_plane2"]["cut_value"],
        "n_sigma_chosen": {
            p: results[p]["chosen_ns"] for p in _SWEEP_PARAMS
        },
    }

    out_json_path = os.path.join(run_dir, "selection_criteria.json")
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved: {out_json_path}")

    # Sweep plot
    plot_selection_criteria(sweep_data, results, run_dir)

    # Summary table
    print()
    print("=" * 78)
    print(f"{'SELECTION CRITERIA SUMMARY':^78}")
    print(f"  run={args.run}  d_threshold={args.d_threshold}  frac={args.frac}")
    print("=" * 78)
    print(f"  {'parameter':22s}  {'n_σ chosen':>10}  {'cut value':>10}  "
          f"{'slope':>8}  status")
    print("-" * 78)
    for p in _SWEEP_PARAMS:
        r = results[p]
        slope_str = f"{r['slope']:8.4f}" if np.isfinite(r["slope"]) else "     nan"
        print(f"  {p:22s}  {r['chosen_ns']:10.4f}  {r['cut_value']:10.4f}  "
              f"{slope_str}  {r['status']}")
    print("-" * 78)
    print(f"  {'slope_plane':22s}  {'(fixed)':>10}  {slope_plane:10.4f}")
    print("=" * 78)

    # Optional: overplot cuts on the galaxy scatter
    if args.source is not None:
        import glob as _glob
        from ellipse_sweep import load_fullmocks, load_desi

        fits_file = args.fits_file
        if args.source == "fullmocks":
            if fits_file is None:
                pattern = os.path.join(args.dir, "TF_extended_AbacusSummit_*.fits")
                matches = sorted(_glob.glob(pattern))
                if not matches:
                    raise FileNotFoundError(f"No FITS files found: {pattern}")
                fits_file = matches[0]
                print(f"Auto-selected: {fits_file}")
            raw_data = load_fullmocks(fits_file)
        else:
            fits_file = fits_file or "data/DESI-DR1_TF_pv_cat_v15.fits"
            raw_data = load_desi(fits_file)

        with open(os.path.join(run_dir, "selection_criteria.json")) as f:
            criteria = json.load(f)

        plot_cuts_on_data(raw_data, ell, criteria, run_dir)
