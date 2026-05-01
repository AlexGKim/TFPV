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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ellipse_sweep import load_desi, apply_cuts, _build_stan_dicts, _cuts_at_nsigma
from predict import ystar_pp_mean_sd_tophat_vectorized, create_average_grid_image


# ─────────────────────────────────────────────────────────────────────────────
# Stan MLE helper
# ─────────────────────────────────────────────────────────────────────────────


def _stan_mle_params(data_dict, init_dict, exe_file, tmp_dir):
    """Run Stan optimize and return dict of MLE parameters, or None on failure.

    Parsed keys: slope, intercept.1, sigma_int_x, sigma_int_y.
    """
    input_path = os.path.join(tmp_dir, "input.json")
    init_path = os.path.join(tmp_dir, "init.json")
    output_path = os.path.join(tmp_dir, "optimize.csv")

    with open(input_path, "w") as f:
        json.dump(data_dict, f)
    with open(init_path, "w") as f:
        json.dump(init_dict, f)

    result = subprocess.run(
        [
            exe_file,
            "optimize",
            "data",
            f"file={input_path}",
            f"init={init_path}",
            "output",
            f"file={output_path}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"Stan optimize failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
        return None

    try:
        with open(output_path) as f:
            lines = [ln.strip() for ln in f if not ln.startswith("#") and ln.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].split(",")
        values = lines[1].split(",")
        row = dict(zip(header, values))
        params = {}
        for key in ("slope", "intercept.1", "sigma_int_x", "sigma_int_y"):
            if key not in row:
                print(
                    f"Key '{key}' missing from Stan output; "
                    f"available: {list(row.keys())[:15]}"
                )
                return None
            params[key] = float(row[key])
        return params
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"Error parsing Stan output: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scatter plot
# ─────────────────────────────────────────────────────────────────────────────


def _save_scatter_plot(run_dir, run_name, raw_data, cuts, params, ell):
    """Scatter plot of selected galaxies coloured by GMM core probability.

    Shows the 1σ/2σ/3σ GMM ellipses, the 3σ selection boundary, and the
    MLE TFR line.  Saves select_v2_scatter.png.
    """
    x_all = raw_data["x"]
    y_all = raw_data["y"]

    # Select only the points that entered the MLE fit
    haty_min = cuts["haty_min"]
    haty_max = cuts["haty_max"]
    slope_p = cuts["slope_plane"]
    ip = cuts["intercept_plane"]
    ip2 = cuts["intercept_plane2"]
    mask = (y_all > haty_min) & (y_all < haty_max)
    lb = np.maximum(haty_min, slope_p * x_all + ip)
    ub = np.minimum(haty_max, slope_p * x_all + ip2)
    mask &= (y_all >= lb) & (y_all <= ub)
    x_sel = x_all[mask]
    y_sel = y_all[mask]

    # GMM core ellipse geometry from selection_ellipse.json
    mu = np.array(ell["mean"])
    cov = np.array(ell["covariance"])
    vals, vecs = np.linalg.eigh(cov)
    semi_axes = np.sqrt(vals)  # [sigma_minor, sigma_major]
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))

    # Colour by Mahalanobis distance from the core GMM mean
    cov_inv = np.linalg.inv(cov)
    dxy = np.column_stack([x_sel - mu[0], y_sel - mu[1]])
    mahal = np.sqrt(np.einsum("ni,ij,nj->n", dxy, cov_inv, dxy))

    mle_slope = params["slope"]
    mle_intercept = params["intercept.1"]

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        x_sel,
        y_sel,
        c=mahal,
        cmap="viridis_r",
        s=1,
        alpha=0.4,
        vmin=0,
        vmax=4,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Mahalanobis distance from core")

    # 1σ/2σ/3σ GMM ellipses
    for n, color, ls in zip([1, 2, 3], ["gold", "orange", "red"], ["-", "--", ":"]):
        ax.add_patch(
            mpatches.Ellipse(
                mu,
                width=2 * n * semi_axes[1],
                height=2 * n * semi_axes[0],
                angle=angle,
                edgecolor=color,
                facecolor="none",
                linewidth=1.5,
                linestyle=ls,
                label=f"{n}σ ellipse",
                zorder=5,
            )
        )

    # 3σ selection boundary
    x_line = np.linspace(x_sel.min() - 0.1, x_sel.max() + 0.1, 300)
    ax.axhline(
        haty_max, color="white", lw=2.0, ls="--", label=f"haty_max={haty_max:.2f}"
    )
    ax.axhline(
        haty_min, color="white", lw=2.0, ls=":", label=f"haty_min={haty_min:.2f}"
    )
    ax.plot(
        x_line,
        slope_p * x_line + ip,
        color="white",
        lw=2.0,
        ls="--",
        label=f"plane1: slope={slope_p:.2f}, b={ip:.2f}",
    )
    ax.plot(
        x_line,
        slope_p * x_line + ip2,
        color="white",
        lw=2.0,
        ls=":",
        label=f"plane2: slope={slope_p:.2f}, b={ip2:.2f}",
    )

    # MLE TFR line
    ax.plot(
        x_line,
        mle_slope * x_line + mle_intercept,
        color="red",
        lw=2.0,
        label=f"MLE slope={mle_slope:.3f}",
    )

    ax.set_xlabel(r"$x = \log_{10}(V/100\,\mathrm{km\,s}^{-1})$")
    ax.set_ylabel(r"$y = R$-band absolute magnitude")
    ax.set_title(f"TFR selection + MLE — {run_name}  (N={mask.sum()})")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(x_sel.min(), x_sel.max())
    ax.set_ylim(y_sel.min(), y_sel.max())
    ax.invert_yaxis()
    fig.tight_layout()

    out_path = os.path.join(run_dir, "select_v2_scatter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pull-plot helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_pull_stats(raw_data, params, y_min, y_max, n_bins):
    """Compute per-galaxy residuals and equal-occupancy binned pull statistics.

    Returns (bin_centers, bin_widths, pulls, wt_means, wt_uncs).
    """
    draws = pd.DataFrame([params])
    x_all = raw_data["x"]
    sx_all = raw_data["sigma_x"]
    y_all = raw_data["y"]
    sy_all = raw_data["sigma_y"]

    mean_pred, sd_pred = ystar_pp_mean_sd_tophat_vectorized(
        draws,
        x_all,
        sx_all,
        y_min=y_min,
        y_max=y_max,
        on_bad_Z="floor",
        Z_floor=1e-300,
    )

    delta = mean_pred - y_all
    sigma_delta = np.sqrt(sd_pred**2 + sy_all**2)

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_all, quantiles)
    bin_edges = np.unique(bin_edges)
    n_bins_act = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    wt_means = np.full(n_bins_act, np.nan)
    wt_uncs = np.full(n_bins_act, np.nan)
    pulls = np.full(n_bins_act, np.nan)

    for i in range(n_bins_act):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (
            (y_all >= lo) & (y_all < hi)
            if i < n_bins_act - 1
            else (y_all >= lo) & (y_all <= hi)
        )
        n = mask.sum()
        if n < 2:
            continue
        w = 1.0 / sigma_delta[mask] ** 2
        wt_means[i] = np.sum(w * delta[mask]) / np.sum(w)
        wt_uncs[i] = 1.0 / np.sqrt(np.sum(w))
        pulls[i] = wt_means[i] / wt_uncs[i]

    return bin_centers, bin_widths, pulls, wt_means, wt_uncs, delta


def _save_grid_plot(
    run_dir, run_name, x, y, delta, haty_lines=None, filename="select_v2_grid.png"
):
    """Draw and save a phase-space residual grid plot.

    Analogous to the grid plots generated in predict.py.
    """
    fig, ax, img = create_average_grid_image(
        x,
        y,
        delta,
        grid_resolution_x=50,
        grid_resolution_y=50,
    )
    ax.set_xlabel(r"$\log_{10}(V/100\,\mathrm{km\,s}^{-1})$")
    ax.set_ylabel(r"$R$-band absolute magnitude")
    ax.set_title(f"Average Magnitude Difference — {run_name}")
    fig.colorbar(img, ax=ax, label="Average Magnitude Difference")

    if haty_lines:
        for label, val in haty_lines.items():
            ax.axhline(
                val,
                color="darkorange",
                linewidth=1.2,
                linestyle="--",
                label=f"{label} = {val:.2f}",
            )
        ax.legend(fontsize=8)

    out_path = os.path.join(run_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved grid plot → {out_path}")


def _save_pull_plot(
    run_dir,
    run_name,
    n_all,
    n_sel,
    params,
    bin_centers,
    bin_widths,
    wt_means,
    wt_uncs,
    haty_lines=None,
    filename="select_v2_pull.png",
):
    """Draw and save select_v2_pull.png.

    haty_lines: optional dict mapping label → M_abs value drawn as a vertical
                line on both panels (e.g. {"haty_min": -22, "haty_max": -19.5}).
    filename:   output filename (default "select_v2_pull.png").
    """
    valid = np.isfinite(wt_means)

    lo_vals = wt_means[valid] - wt_uncs[valid]
    hi_vals = wt_means[valid] + wt_uncs[valid]
    all_vals = np.concatenate([lo_vals, hi_vals])
    p_lo = float(np.nanpercentile(all_vals, 2))
    p_hi = float(np.nanpercentile(all_vals, 98))
    pad = 0.22 * max(p_hi - p_lo, 1e-6)

    colors_mean = np.where(wt_means[valid] >= 0, "steelblue", "tomato")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(
        bin_centers[valid],
        wt_means[valid],
        width=bin_widths[valid] * 0.8,
        yerr=wt_uncs[valid],
        color=colors_mean,
        alpha=0.75,
        error_kw=dict(ecolor="black", capsize=3),
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(p_lo - pad, p_hi + pad)

    if haty_lines:
        for label, val in haty_lines.items():
            ax.axvline(
                val,
                color="darkorange",
                linewidth=1.2,
                linestyle="--",
                label=f"{label} = {val:.2f}",
            )
        ax.legend(fontsize=8)

    ax.set_xlabel(r"$M_\mathrm{abs}$ bin center")
    ax.set_ylabel(r"$\langle\Delta M\rangle_w$  (weighted mean)")
    ax.set_title(
        f"Weighted mean residual — {run_name}  "
        f"(N_all={n_all}, N_sel={n_sel}, "
        f"slope={params['slope']:.3f}, "
        f"intercept={params['intercept.1']:.3f})"
    )

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
    parser.add_argument(
        "--config", help="Path to JSON config file (e.g., configs/dr1_default.json)"
    )
    parser.add_argument(
        "--run", help="Run name (output/<run>/) (required if not using config)"
    )
    parser.add_argument(
        "--fits_file", help="Path to DESI FITS file (required if not using config)"
    )
    parser.add_argument(
        "--exe",
        default=None,
        help="Path to compiled tophat Stan binary (default: tophat)",
    )
    parser.add_argument(
        "--source", default=None, choices=["DESI"], help="Data source (default: DESI)"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=None,
        help="Number of M_abs bins for pull plot (default: 20)",
    )
    parser.add_argument(
        "--z_obs_min",
        type=float,
        default=None,
        help="Minimum redshift for Stan MLE sample (default: 0.03)",
    )
    parser.add_argument(
        "--z_obs_max",
        type=float,
        default=None,
        help="Maximum redshift for Stan MLE sample (default: 0.1)",
    )
    parser.add_argument(
        "--set_fiducial",
        action="store_true",
        help="Write select_v2_fiducial.json from --haty_min/max; skip Stan",
    )
    parser.add_argument(
        "--haty_min",
        type=float,
        default=None,
        help="Fiducial bright-end magnitude limit (required with --set_fiducial)",
    )
    parser.add_argument(
        "--haty_max",
        type=float,
        default=None,
        help="Fiducial dim-end magnitude limit (required with --set_fiducial)",
    )
    args = parser.parse_args()

    from config_utils import apply_config

    cfg = apply_config(args)
    if cfg.get("run") and not args.run:
        args.run = cfg["run"]
    if cfg.get("fits_file") and not args.fits_file:
        args.fits_file = cfg["fits_file"]

    if not args.run or not args.fits_file:
        parser.error(
            "The following arguments are required: --run, --fits_file (or provide them via --config)"
        )

    if args.set_fiducial and (args.haty_min is None or args.haty_max is None):
        parser.error(
            "--set_fiducial requires both --haty_min and --haty_max (either via CLI or --config)"
        )

    run_dir = os.path.join("output", args.run)
    os.makedirs(run_dir, exist_ok=True)

    # ── Step 1: Load 1-sigma ellipse cuts ─────────────────────────────────────
    ellipse_path = os.path.join(run_dir, "selection_ellipse.json")
    if not os.path.exists(ellipse_path):
        raise FileNotFoundError(
            f"{ellipse_path} not found — run selection_ellipse.py first."
        )
    with open(ellipse_path) as f:
        ell = json.load(f)

    mu = np.array(ell["mean"])
    sigma = np.array(ell["covariance"])
    cuts3 = _cuts_at_nsigma(mu, sigma, 3.0)

    # ── Set-fiducial mode: record user-chosen cuts, then regenerate pull plot ──
    if args.set_fiducial:
        fiducial = {
            "haty_min": args.haty_min,
            "haty_max": args.haty_max,
            "slope_plane": cuts3["slope_plane"],
            "intercept_plane": cuts3["intercept_plane"],
            "intercept_plane2": cuts3["intercept_plane2"],
            "z_obs_min": args.z_obs_min,
            "z_obs_max": args.z_obs_max,
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

        cuts3_no_z = {
            k: cuts3[k]
            for k in (
                "haty_min",
                "haty_max",
                "slope_plane",
                "intercept_plane",
                "intercept_plane2",
            )
        }
        raw_data = load_desi(args.fits_file)
        x_sel, _, _, _ = apply_cuts(raw_data, cuts3_no_z)
        data_dict, _ = _build_stan_dicts(raw_data, cuts3_no_z)
        if data_dict is None:
            print("Warning: could not build Stan data dict — skipping pull plot.")
            return
        y_min = float(data_dict["y_min"])
        y_max = float(data_dict["y_max"])

        bin_centers, bin_widths, pulls, wt_means, wt_uncs, delta = _compute_pull_stats(
            raw_data, params, y_min, y_max, args.n_bins
        )
        _save_pull_plot(
            run_dir,
            args.run,
            n_all=len(raw_data["x"]),
            n_sel=len(x_sel),
            params=params,
            bin_centers=bin_centers,
            bin_widths=bin_widths,
            wt_means=wt_means,
            wt_uncs=wt_uncs,
            haty_lines={"haty_min": args.haty_min, "haty_max": args.haty_max},
            filename="select_v2_fiducial_pull.png",
        )

        _save_grid_plot(
            run_dir,
            args.run,
            raw_data["x"],
            raw_data["y"],
            delta,
            haty_lines={"haty_min": args.haty_min, "haty_max": args.haty_max},
            filename="select_v2_fiducial_grid.png",
        )

        return

    # ── Diagnostic mode: 3-sigma cuts for MLE ────────────────────────────────
    cuts = {
        "haty_min": cuts3["haty_min"],
        "haty_max": cuts3["haty_max"],
        "slope_plane": cuts3["slope_plane"],
        "intercept_plane": cuts3["intercept_plane"],
        "intercept_plane2": cuts3["intercept_plane2"],
    }
    if args.z_obs_min is not None:
        cuts["z_obs_min"] = args.z_obs_min
    if args.z_obs_max is not None:
        cuts["z_obs_max"] = args.z_obs_max

    print("Diagnostic cuts (3-sigma):")
    print(f"  haty_min={cuts['haty_min']:.4f}  haty_max={cuts['haty_max']:.4f}")
    print(f"  slope_plane={cuts['slope_plane']:.4f}")
    print(
        f"  intercept_plane={cuts['intercept_plane']:.4f}  "
        f"intercept_plane2={cuts['intercept_plane2']:.4f}"
    )

    # ── Step 2: Load data and apply cuts ──────────────────────────────────────
    raw_data = load_desi(args.fits_file)
    x, _, _, _ = apply_cuts(raw_data, cuts)
    print(f"After cuts: N = {len(x)}")

    # ── Step 3: Build Stan data dict and run MLE ───────────────────────────────
    data_dict, init_dict = _build_stan_dicts(raw_data, cuts)
    if data_dict is None:
        raise RuntimeError(
            "Sample too small or geometrically invalid cuts; cannot run Stan MLE."
        )

    # Resolve executable path
    exe_file = args.exe
    if not os.path.isabs(exe_file) and not os.path.exists(exe_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, exe_file)
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

    _save_scatter_plot(run_dir, args.run, raw_data, cuts, params, ell)

    # ── Compute per-galaxy residual and pull profile over z-restricted catalog ──
    y_min = float(data_dict["y_min"])
    y_max = float(data_dict["y_max"])

    # Filter to the same z window used for the MLE fit so the pull plot reflects
    # the actual calibration sample, not galaxies outside the redshift range.
    _z_raw = raw_data.get("z_obs", np.zeros(len(raw_data["x"])))
    _z_mask = np.ones(len(_z_raw), dtype=bool)
    if cuts.get("z_obs_min") is not None:
        _z_mask &= _z_raw > cuts["z_obs_min"]
    if cuts.get("z_obs_max") is not None:
        _z_mask &= _z_raw <= cuts["z_obs_max"]
    raw_data_z = {k: v[_z_mask] for k, v in raw_data.items()}

    bin_centers, bin_widths, pulls, wt_means, wt_uncs, delta = _compute_pull_stats(
        raw_data_z, params, y_min, y_max, args.n_bins
    )

    pull_stats = {
        "n_all": int(len(raw_data_z["x"])),
        "n_sel": int(len(x)),
        "bin_centers": bin_centers.tolist(),
        "bin_widths": bin_widths.tolist(),
        "pulls": pulls.tolist(),
        "wt_means": wt_means.tolist(),
        "wt_uncs": wt_uncs.tolist(),
        "x": raw_data_z["x"].tolist(),
        "y": raw_data_z["y"].tolist(),
        "delta": delta.tolist(),
    }
    pull_stats_path = os.path.join(run_dir, "select_v2_pull_stats.json")
    with open(pull_stats_path, "w") as f:
        json.dump(pull_stats, f)
    print(f"Saved pull stats → {pull_stats_path}")

    _save_pull_plot(
        run_dir,
        args.run,
        n_all=pull_stats["n_all"],
        n_sel=pull_stats["n_sel"],
        params=params,
        bin_centers=bin_centers,
        bin_widths=bin_widths,
        wt_means=wt_means,
        wt_uncs=wt_uncs,
    )

    _save_grid_plot(
        run_dir,
        args.run,
        raw_data_z["x"],
        raw_data_z["y"],
        delta,
        filename="select_v2_mle_grid.png",
    )


if __name__ == "__main__":
    main()
