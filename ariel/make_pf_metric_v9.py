#!/usr/bin/env python
"""Pathfinder-seeded dense metric for v9 (3 state variables, full 2x2 S).

Parameter order must match the declaration order in sb_v9.stan, since the metric
is a covariance on Stan's unconstrained scale. Lower-bounded parameters (L11, L22) are log-transformed; the interval-bounded ones use the logit.
"""
import argparse
import csv
import json

import numpy as np

SLOPE_LO_COEF, SLOPE_HI_COEF = -9.0, -4.0
INTER_LO_OFF, INTER_HI_OFF = -24.0, -14.0


def _logit(x, lo, hi):
    u = np.clip((np.asarray(x, float) - lo) / (hi - lo), 1e-12, 1 - 1e-12)
    return np.log(u / (1.0 - u))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--stepsize", type=float, default=0.1)
    args = ap.parse_args()

    run = args.run
    x = np.asarray(json.load(open(f"output/{run}/input.json"))["x"], float)
    mean_x, sd_x = float(x.mean()), float(x.std(ddof=1))

    rows = [r for r in csv.reader(open(f"output/{run}/pathfinder.csv"))
            if r and not r[0].startswith("#")]
    hdr = rows[0]
    if "intercept_std.2" in hdr:
        raise SystemExit("assumes N_bins == 1; found intercept_std.2")
    col = lambda n: np.array([float(r[hdr.index(n)]) for r in rows[1:]])

    slope = col("slope_std")
    inter = col("intercept_std.1")
    U = [
        _logit(slope, SLOPE_LO_COEF * sd_x, SLOPE_HI_COEF * sd_x),
        _logit(inter, INTER_LO_OFF + slope * mean_x / sd_x,
                      INTER_HI_OFF + slope * mean_x / sd_x),
        _logit(col("sigma_int_x"), 0.0, 1.0),
        np.log(np.clip(col("L11"), 1e-12, None)),
        col("L21"),
        np.log(np.clip(col("L22"), 1e-12, None)),
        col("delta_s"), col("mu_s"),
        col("gamma_inc"), col("lambda_s"),
        col("alpha_kcorr_r"), col("alpha_kcorr_s"),
    ]
    X = np.vstack(U).T
    if X.shape[0] < 20:
        raise SystemExit(f"only {X.shape[0]} Pathfinder draws; need >= 20")
    C = np.cov(X.T)
    ev = np.linalg.eigvalsh(C)
    if ev.min() <= 0:
        C = C + np.eye(C.shape[0]) * (abs(ev.min()) + 1e-8)
        ev = np.linalg.eigvalsh(C)
    json.dump({"stepsize": args.stepsize, "inv_metric": C.tolist()},
              open(f"output/{run}/pf_metric.json", "w"))
    print(f"wrote output/{run}/pf_metric.json")
    print(f"  {X.shape[1]}x{X.shape[1]} dense_e metric from {X.shape[0]} draws; "
          f"cond {ev.max()/ev.min():.1f}, PD={ev.min() > 0}")


if __name__ == "__main__":
    main()
