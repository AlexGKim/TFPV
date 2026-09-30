#!/usr/bin/env python
"""Pathfinder-seeded dense metric for the v7 model.

make_pf_metric.py is hardcoded to the 2color parameter list (w.1, w.2, w.3).
v7 replaces those with sigma_ach (lower-bounded, so log-transformed) and the
free Cartesian pair v_chrom.1, v_chrom.2.  Parameter order below must match the
declaration order in 2color_v7.stan, since the metric is a covariance on Stan's
unconstrained scale.

mean_x and sd_x are transformed data in the model (mean(x), sd(x) with Stan's
N-1 convention), so they are recomputed from input.json rather than read from
init_MAP.json -- v7's init does not carry them.
"""
import argparse
import csv
import json

import numpy as np

SLOPE_LO_COEF, SLOPE_HI_COEF = -9.0, -4.0
INTER_LO_OFF, INTER_HI_OFF = -24.0, -14.0


def _logit_unit_interval(x, lo, hi):
    u = (np.asarray(x, float) - lo) / (hi - lo)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return np.log(u / (1.0 - u))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--stepsize", type=float, default=0.1)
    args = ap.parse_args()

    run = args.run
    pf_csv = f"output/{run}/pathfinder.csv"
    out_json = f"output/{run}/pf_metric.json"

    x = np.asarray(json.load(open(f"output/{run}/input.json"))["x"], float)
    mean_x, sd_x = float(x.mean()), float(x.std(ddof=1))

    rows = [r for r in csv.reader(open(pf_csv)) if r and not r[0].startswith("#")]
    hdr = rows[0]
    if "intercept_std.2" in hdr:
        raise SystemExit("This script assumes N_bins==1; found intercept_std.2.")
    col = lambda n: np.array([float(r[hdr.index(n)]) for r in rows[1:]])

    slope = col("slope_std")
    inter = col("intercept_std.1")

    U = []
    U.append(_logit_unit_interval(slope, SLOPE_LO_COEF * sd_x, SLOPE_HI_COEF * sd_x))
    U.append(_logit_unit_interval(inter, INTER_LO_OFF + slope * mean_x / sd_x,
                                         INTER_HI_OFF + slope * mean_x / sd_x))
    U.append(_logit_unit_interval(col("sigma_int_x"), 0.0, 1.0))
    U.append(np.log(np.clip(col("sigma_ach"), 1e-12, None)))   # <lower=0> -> log
    for n in ("v_chrom.1", "v_chrom.2",                        # native unconstrained
              "delta_c", "mu_c", "delta_g", "mu_g",
              "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g"):
        U.append(col(n))

    X = np.vstack(U).T
    if X.shape[0] < 20:
        raise SystemExit(f"Only {X.shape[0]} Pathfinder draws; need >=20.")

    C = np.cov(X.T)
    eig = np.linalg.eigvalsh(C)
    if eig.min() <= 0:
        C = C + np.eye(C.shape[0]) * (abs(eig.min()) + 1e-8)
    ev = np.linalg.eigvalsh(C)
    json.dump({"stepsize": args.stepsize, "inv_metric": C.tolist()}, open(out_json, "w"))
    print(f"Pathfinder metric written to {out_json}")
    print(f"  full {X.shape[1]}x{X.shape[1]} dense_e metric from {X.shape[0]} draws; "
          f"cond {ev.max()/ev.min():.1f}, PD={ev.min() > 0}")


if __name__ == "__main__":
    main()
