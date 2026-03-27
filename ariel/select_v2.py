#!/usr/bin/env python3
"""
select_v2.py — TFR selection pipeline (v2).

Two modes:

  Diagnostic (Step 2 of Selection_v2.md)
  ────────────────────────────────────────
  Run Stan MLE on the 3-sigma ellipse selection, then compute and plot the
  pull residual over ALL catalog objects.  Use the pull plot to choose the
  final magnitude window.

    python select_v2.py --run RUN --fits_file FITS --exe ./tophat

  Set fiducial (Step 3 of Selection_v2.md)
  ─────────────────────────────────────────
  Record the user-chosen selection parameters as the fiducial for downstream
  scripts (desi_data.py).  No Stan call is made.

    python select_v2.py --run RUN --fits_file FITS \\
        --set_fiducial --haty_min -22 --haty_max -19.5
"""

import argparse
import json
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ellipse_sweep import load_desi, apply_cuts, _build_stan_dicts, _cuts_at_nsigma
from predict import ystar_pp_mean_sd_tophat_vectorized


# ─────────────────────────────────────────────────────────────────────────────
# Stan MLE helper
# ─────────────────────────────────────────────────────────────────────────────

def _stan_mle_params(data_dict, init_dict, exe_file, tmp_dir):
    """Run Stan optimize and return dict of MLE parameters, or None on failure.

    Parsed keys: slope, intercept.1, sigma_int_x, sigma_int_y.
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
        print(f"Stan optimize failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
        return None

    try:
        with open(output_path) as f:
            lines = [ln.strip() for ln in f
                     if not ln.startswith("#") and ln.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        values = lines[1].split(",")
        row    = dict(zip(header, values))
        params = {}
        for key in ("slope", "intercept.1", "sigma_int_x", "sigma_int_y"):
            if key not in row:
                print(f"Key '{key}' missing from Stan output; "
                      f"available: {list(row.keys())[:15]}")
                return None
            params[key] = float(row[key])
        return params
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"Error parsing Stan output: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pull-plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pull_stats(raw_data, params, y_min, y_max, n_bins):
    """Compute per-galaxy residuals and equal-occupancy binned pull statistics.

    Returns (bin_centers, bin_widths, pulls, wt_means, wt_uncs).
    """
    draws  = pd.DataFrame([params])
    x_all  = raw_data["x"]
    sx_all = raw_data["sigma_x"]
    y_all  = raw_data["y"]
    sy_all = raw_data["sigma_y"]

    mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
        draws, x_all, sx_all,
        y_min=y_min, y_max=y_max,
        on_bad_Z="floor", Z_floor=1e-300,
    )

    delta       = mean_pred - y_all
    sigma_delta = np.sqrt(sd_pred**2 + sy_all**2)

    quantiles   = np.linspace(0, 100, n_bins + 1)
    bin_edges   = np.percentile(y_all, quantiles)
    bin_edges   = np.unique(bin_edges)
    n_bins_act  = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths  = np.diff(bin_edges)

    wt_means = np.full(n_bins_act, np.nan)
    wt_uncs  = np.full(n_bins_act, np.nan)
    pulls    = np.full(n_bins_act, np.nan)

    for i in range(n_bins_act):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_all >= lo) & (y_all < hi) if i < n_bins_act - 1 else (y_all >= lo) & (y_all <= hi)
        n = mask.sum()
        if n < 2:
            continue
        w           = 1.0 / sigma_delta[mask] ** 2
        wt_means[i] = np.sum(w * delta[mask]) / np.sum(w)
        wt_uncs[i]  = 1.0 / np.sqrt(np.sum(w))
        pulls[i]    = wt_means[i] / wt_uncs[i]

    return bin_centers, bin_widths, pulls, wt_means, wt_uncs


def _save_pull_plot(run_dir, run_name, n_all, n_sel, params,
                   bin_centers, bin_widths, pulls, wt_means, wt_uncs,
                   haty_lines=None, filename="select_v2_pull.png"):
    """Draw and save select_v2_pull.png.

    haty_lines: optional dict mapping label → M_abs value drawn as a vertical
                line on both panels (e.g. {"haty_min": -22, "haty_max": -19.5}).
    filename:   output filename (default "select_v2_pull.png").
    """
    valid = np.isfinite(pulls)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Top panel: pull
    ax0    = axes[0]
    colors = np.where(pulls[valid] >= 0, "steelblue", "tomato")
    ax0.bar(bin_centers[valid], pulls[valid],
            width=bin_widths[valid] * 0.8, color=colors, alpha=0.75)
    ax0.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if haty_lines:
        for label, val in haty_lines.items():
            ax0.axvline(val, color="darkorange", linewidth=1.2,
                        linestyle="--", label=label)
        ax0.legend(fontsize=8)
    ax0.set_ylabel("Pull  (weighted mean / unc)")
    ax0.set_title(
        f"Residual pull — {run_name}  (N_all={n_all}, N_sel={n_sel}, "
        f"slope={params['slope']:.3f}, "
        f"intercept={params['intercept.1']:.3f})"
    )
    p_lo, p_hi = np.nanpercentile(pulls[valid], [2, 98])
    pad = 0.15 * max(p_hi - p_lo, 1e-6)
    ax0.set_ylim(p_lo - pad, p_hi + pad)

    # Bottom panel: weighted mean ± uncertainty
    ax1         = axes[1]
    colors_mean = np.where(wt_means[valid] >= 0, "steelblue", "tomato")
    ax1.bar(bin_centers[valid], wt_means[valid],
            width=bin_widths[valid] * 0.8,
            yerr=wt_uncs[valid],
            color=colors_mean, alpha=0.75,
            error_kw=dict(ecolor="black", capsize=3))
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if haty_lines:
        for label, val in haty_lines.items():
            ax1.axvline(val, color="darkorange", linewidth=1.2,
                        linestyle="--", label=label)
        ax1.legend(fontsize=8)
    ax1.set_xlabel(r"$M_\mathrm{abs}$ bin center")
    ax1.set_ylabel(r"$\langle\Delta M\rangle_w$  (weighted mean)")
    lo_vals = wt_means[valid] - wt_uncs[valid]
    hi_vals = wt_means[valid] + wt_uncs[valid]
    p_lo = float(np.nanpercentile(lo_vals, 2))
    p_hi = float(np.nanpercentile(hi_vals, 98))
    pad = 0.15 * max(p_hi - p_lo, 1e-6)
    ax1.set_ylim(p_lo - pad, p_hi + pad)

    fig.tight_layout()
    pull_path = os.path.join(run_dir, filename)
    fig.savefig(pull_path, dpi=150)
    plt.close(fig)
    print(f"Saved pull profile → {pull_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direct 1-sigma ellipse selection pipeline (select_v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run",       required=True,
                        help="Run name (output/<run>/)")
    parser.add_argument("--fits_file", required=True,
                        help="Path to DESI FITS file")
    parser.add_argument("--exe",       default="tophat",
                        help="Path to compiled tophat Stan binary")
    parser.add_argument("--source",    default="DESI", choices=["DESI"],
                        help="Data source")
    parser.add_argument("--n_bins",      type=int, default=20,
                        help="Number of M_abs bins for pull plot (diagnostic mode only)")
    parser.add_argument("--z_obs_min",  type=float, default=0.03,
                        help="Minimum redshift for Stan MLE sample (default: 0.03); "
                             "not applied to pull-plot prediction")
    parser.add_argument("--set_fiducial", action="store_true",
                        help="Write select_v2_fiducial.json from --haty_min/max; skip Stan")
    parser.add_argument("--haty_min",  type=float, default=None,
                        help="Fiducial bright-end magnitude limit (required with --set_fiducial)")
    parser.add_argument("--haty_max",  type=float, default=None,
                        help="Fiducial dim-end magnitude limit (required with --set_fiducial)")
    args = parser.parse_args()

    if args.set_fiducial and (args.haty_min is None or args.haty_max is None):
        parser.error("--set_fiducial requires both --haty_min and --haty_max")

    run_dir = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # ── Step 1: Load 1-sigma ellipse cuts ─────────────────────────────────────
    ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
    if not os.path.exists(ellipse_path):
        raise FileNotFoundError(
            f"{ellipse_path} not found — run selection_ellipse.py first.")
    with open(ellipse_path) as f:
        ell = json.load(f)

    mu    = np.array(ell["mean"])
    sigma = np.array(ell["covariance"])
    cuts3 = _cuts_at_nsigma(mu, sigma, 3.0)

    # ── Set-fiducial mode: record user-chosen cuts, then regenerate pull plot ──
    if args.set_fiducial:
        fiducial = {
            "haty_min":         args.haty_min,
            "haty_max":         args.haty_max,
            "slope_plane":      cuts3["slope_plane"],
            "intercept_plane":  cuts3["intercept_plane"],
            "intercept_plane2": cuts3["intercept_plane2"],
        }
        fiducial_path = os.path.join(run_dir, "select_v2_fiducial.json")
        with open(fiducial_path, "w") as f:
            json.dump(fiducial, f, indent=2)
        print(f"Saved fiducial criteria → {fiducial_path}")
        for k, v in fiducial.items():
            print(f"  {k} = {v}")

        # Regenerate pull plot with fiducial cut lines marked
        mle_path = os.path.join(run_dir, "select_v2_mle.json")
        if not os.path.exists(mle_path):
            print(f"Warning: {mle_path} not found — skipping pull plot.")
            return
        with open(mle_path) as f:
            params = json.load(f)

        cuts3_no_z = {k: cuts3[k] for k in
                      ("haty_min", "haty_max", "slope_plane",
                       "intercept_plane", "intercept_plane2")}
        raw_data = load_desi(args.fits_file)
        x_sel, _, _, _ = apply_cuts(raw_data, cuts3_no_z)
        data_dict, _ = _build_stan_dicts(raw_data, cuts3_no_z)
        if data_dict is None:
            print("Warning: could not build Stan data dict — skipping pull plot.")
            return
        y_min = float(data_dict["y_min"])
        y_max = float(data_dict["y_max"])

        bin_centers, bin_widths, pulls, wt_means, wt_uncs = \
            _compute_pull_stats(raw_data, params, y_min, y_max, args.n_bins)
        _save_pull_plot(
            run_dir, args.run,
            n_all=len(raw_data["x"]), n_sel=len(x_sel),
            params=params,
            bin_centers=bin_centers, bin_widths=bin_widths,
            pulls=pulls, wt_means=wt_means, wt_uncs=wt_uncs,
            haty_lines={"haty_min": args.haty_min, "haty_max": args.haty_max},
            filename="select_v2_fiducial_pull.png",
        )
        return

    # ── Diagnostic mode: 3-sigma cuts for MLE ────────────────────────────────
    cuts = {
        "haty_min":         cuts3["haty_min"],
        "haty_max":         cuts3["haty_max"],
        "slope_plane":      cuts3["slope_plane"],
        "intercept_plane":  cuts3["intercept_plane"],
        "intercept_plane2": cuts3["intercept_plane2"],
    }
    if args.z_obs_min is not None:
        cuts["z_obs_min"] = args.z_obs_min

    print("Diagnostic cuts (3-sigma):")
    print(f"  haty_min={cuts['haty_min']:.4f}  haty_max={cuts['haty_max']:.4f}")
    print(f"  slope_plane={cuts['slope_plane']:.4f}")
    print(f"  intercept_plane={cuts['intercept_plane']:.4f}  "
          f"intercept_plane2={cuts['intercept_plane2']:.4f}")

    # ── Step 2: Load data and apply cuts ──────────────────────────────────────
    raw_data = load_desi(args.fits_file)
    x, _, _, _ = apply_cuts(raw_data, cuts)
    print(f"After cuts: N = {len(x)}")

    # ── Step 3: Build Stan data dict and run MLE ───────────────────────────────
    data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
    if data_dict is None:
        raise RuntimeError(
            "Sample too small or geometrically invalid cuts; cannot run Stan MLE.")

    # Resolve executable path
    exe_file = args.exe
    if not os.path.isabs(exe_file) and not os.path.exists(exe_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate  = os.path.join(script_dir, exe_file)
        if os.path.exists(candidate):
            exe_file = candidate

    print(f"Running Stan MLE: exe={exe_file}  N={data_dict['N_total']}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        params = _stan_mle_params(data_dict, init_dict, exe_file, tmp_dir)

    if params is None:
        raise RuntimeError("Stan MLE failed; cannot continue.")

    print("MLE parameters:")
    for k, v in params.items():
        print(f"  {k} = {v:.6f}")

    mle_path = os.path.join(run_dir, "select_v2_mle.json")
    with open(mle_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved MLE parameters → {mle_path}")

    # ── Compute per-galaxy residual and pull profile over full catalog ─────────
    y_min = float(data_dict["y_min"])
    y_max = float(data_dict["y_max"])

    bin_centers, bin_widths, pulls, wt_means, wt_uncs = \
        _compute_pull_stats(raw_data, params, y_min, y_max, args.n_bins)

    pull_stats = {
        "n_all":        int(len(raw_data["x"])),
        "n_sel":        int(len(x)),
        "bin_centers":  bin_centers.tolist(),
        "bin_widths":   bin_widths.tolist(),
        "pulls":        pulls.tolist(),
        "wt_means":     wt_means.tolist(),
        "wt_uncs":      wt_uncs.tolist(),
    }
    pull_stats_path = os.path.join(run_dir, "select_v2_pull_stats.json")
    with open(pull_stats_path, "w") as f:
        json.dump(pull_stats, f)
    print(f"Saved pull stats → {pull_stats_path}")

    _save_pull_plot(
        run_dir, args.run,
        n_all=pull_stats["n_all"], n_sel=pull_stats["n_sel"],
        params=params,
        bin_centers=bin_centers, bin_widths=bin_widths,
        pulls=pulls, wt_means=wt_means, wt_uncs=wt_uncs,
    )


if __name__ == "__main__":
    main()
