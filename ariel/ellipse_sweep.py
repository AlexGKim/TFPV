#!/usr/bin/env python3
"""
ellipse_sweep.py — Parametric sweep of TFR selection cuts along the GMM ellipse scale.

For each of the four selection-cut parameters (haty_min, haty_max, intercept_plane,
intercept_plane2), varies the ellipse scale n_σ over a log-spaced grid while keeping
the other three parameters fixed at their 1σ values.  At each grid point the TFR slope
is estimated by Stan MAP optimisation (tophat model).  The resulting slope profiles and
their numerical derivatives ∂s/∂(n_σ) are plotted.

Requires output/<run>/selection_ellipse.json (produced by selection_ellipse.py).

Usage:
  python ellipse_sweep.py \\
      --source fullmocks \\
      --fits_file data/TF_extended_AbacusSummit_base_c000_ph000_r001_z0.11.fits \\
      --run c000_ph000_r001

Output:
  output/<run>/ellipse_sweep.png
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
    for col in ("ZOBS", "Z_OBS", "zobs", "Z", "ZHELIO"):
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

def run_ellipse_sweep(raw_data, mu, sigma, n_sigma_vals, extra_cuts, exe_file):
    """Sweep each ellipse cut parameter over n_sigma_vals, others fixed at 1σ.

    For each of the four cut parameters (haty_min, haty_max, intercept_plane,
    intercept_plane2), varies the ellipse scale n_σ while keeping the others
    fixed at their 1σ values.  slope_plane is always fixed at the 1σ value.

    Returns dict mapping parameter name → list of (n_sigma, cut_value, slope, N).
    """
    _SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]
    ref1 = _cuts_at_nsigma(mu, sigma, 1.0)
    results = {p: [] for p in _SWEEP_PARAMS}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for p in _SWEEP_PARAMS:
            print(f"  Sweeping {p} …")
            for n_sigma in n_sigma_vals:
                cuts_nsigma = _cuts_at_nsigma(mu, sigma, n_sigma)
                cuts = dict(ref1)
                cuts[p] = cuts_nsigma[p]
                cuts["slope_plane"] = ref1["slope_plane"]
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


def plot_ellipse_partial_deriv(sweep_results, n_sigma_vals, run_dir):
    """2-row × 4-column figure: slope vs n_σ (top) and ∂s/∂(n_σ) vs n_σ (bottom).

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

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        # Top: slope vs n_σ  (log x-axis)
        valid = np.isfinite(slopes)
        ax_top.plot(ns_arr[valid], slopes[valid], "o-", color="steelblue",
                    linewidth=1.5, markersize=4)
        ax_top.set_xscale("log")
        ax_top.axvline(1.0, color="gray", linestyle="--", linewidth=1.0,
                       label=r"$n_\sigma=1$")
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
        ax_bot.axvline(1.0, color="gray", linestyle="--", linewidth=1.0,
                       label=r"$n_\sigma=1$")
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
                        help="Lower end of n_σ grid")
    parser.add_argument("--n_sigma_max", type=float, default=1.7,
                        help="Upper end of n_σ grid")
    parser.add_argument("--n_sigma_n",   type=int,   default=21,
                        help="Number of log-spaced grid points")
    parser.add_argument("--z_obs_min", type=float, default=0.03,
                        help="Minimum redshift cut")
    parser.add_argument("--z_obs_max", type=float, default=0.10,
                        help="Maximum redshift cut")
    parser.add_argument("--n_sweep_objects", type=int, default=10000,
                        help="Subsample raw data to this many objects (0 = use all)")

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

    print(f"Running ellipse sweep: {args.n_sigma_n} n_σ values in "
          f"[{args.n_sigma_min}, {args.n_sigma_max}], exe={exe_file}")

    sweep_results = run_ellipse_sweep(
        raw_data, mu, sigma, n_sigma_vals, extra_cuts, exe_file)

    save_sweep_results(sweep_results, run_dir)
    plot_ellipse_partial_deriv(sweep_results, n_sigma_vals, run_dir)

    # Summary table
    _SWEEP_PARAMS = ["haty_min", "haty_max", "intercept_plane", "intercept_plane2"]
    ref1 = _cuts_at_nsigma(mu, sigma, 1.0)
    print("\n" + "=" * 60)
    print("ELLIPSE SWEEP SUMMARY  (at n_σ = 1.0)")
    print("=" * 60)
    for p in _SWEEP_PARAMS:
        records = sweep_results[p]
        # Find the record closest to n_sigma = 1.0
        idx1 = int(np.argmin(np.abs(np.array([r[0] for r in records]) - 1.0)))
        _, cut_val, slope1, N1 = records[idx1]
        ns_arr = np.array([r[0] for r in records])
        sl_arr = np.array([r[2] if r[2] is not None else np.nan
                           for r in records])
        valid  = np.isfinite(sl_arr)
        if valid.sum() >= 2:
            deriv_at1 = float(np.gradient(sl_arr[valid], ns_arr[valid])[
                np.argmin(np.abs(ns_arr[valid] - 1.0))])
        else:
            deriv_at1 = float("nan")
        print(f"  {p:22s}  cut={cut_val:7.3f}  "
              f"slope={slope1 if slope1 is not None else float('nan'):7.4f}  "
              f"ds/dn_σ={deriv_at1:+.4f}  N={N1}")
    print("=" * 60)
