#!/usr/bin/env python
"""Pathfinder-seeded dense metric for 2color_v10 (v5 plus a redshift-dependent slope).

Parameter order must match the declaration order in 2color_v10.stan, since the metric
is a covariance on Stan's unconstrained scale. slope_std_0 and intercept_std are
interval-bounded (logit); sigma_int_x is bounded [0,1] (logit); slope_std_1 and the
rest are unconstrained.
"""
import argparse, csv, json
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
    s0 = col("slope_std_0")
    inter = col("intercept_std.1")
    U = [
        _logit(s0, SLOPE_LO_COEF * sd_x, SLOPE_HI_COEF * sd_x),
        col("slope_std_1"),                       # [V10] unconstrained
        _logit(inter, INTER_LO_OFF + s0 * mean_x / sd_x,
                      INTER_HI_OFF + s0 * mean_x / sd_x),
        _logit(col("sigma_int_x"), 0.0, 1.0),
        col("w.1"), col("w.2"), col("w.3"),
        col("delta_c"), col("mu_c"), col("delta_g"), col("mu_g"),
        col("alpha_kcorr_r"), col("alpha_kcorr_z"), col("alpha_kcorr_g"),
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
