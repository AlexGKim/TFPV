#!/usr/bin/env python3
"""Build a dense_e warmup metric for 2color.stan from a Pathfinder run.

Step 5e of DR2_2COLOR.md. Pathfinder (Stan's L-BFGS variational method) gives a
cheap approximate posterior in ~1 min; its draws estimate the posterior scale
and correlations of the sampling parameters. The 2color posterior is badly
conditioned, so HMC started from the identity metric spends warmup at max
treedepth with a tiny stepsize. Seeding `metric=dense_e` with the Pathfinder
covariance removes that mis-conditioning from iteration 1.

The rank-1 model (S = w w^T) has 13 sampling dimensions, and — unlike the earlier
rank-2 unit_vector parameterization — ALL of them transform cleanly to Stan's
unconstrained scale, so the dense metric is exact (no placeholder for a
sphere-constrained direction). Unconstrained order Stan uses for 2color.stan:

  slope_std, intercept_std.1, sigma_int_x,
  w.1, w.2, w.3,                 (vector[3], unconstrained-native)
  delta_c, mu_c, delta_g, mu_g,
  alpha_kcorr_r, alpha_kcorr_z, alpha_kcorr_g

Bounded parameters are transformed before taking the covariance (logit for
lower/upper bounds). Emits a full 13x13 dense `inv_metric` to
output/<run>/pf_metric.json. Assumes N_bins == 1 (the DR2 2color workflow).
"""
import argparse
import csv
import json
import numpy as np

# Bound constants hardcoded in 2color.stan's parameters block (keep in sync):
#   slope_std     : <lower=-9*sd_x,               upper=-4.0*sd_x>
#   intercept_std : <lower=-24+slope_std*mean_x/sd_x, upper=-14+slope_std*mean_x/sd_x>
#   sigma_int_x   : <lower=0, upper=1>
#   w             : vector[3], unconstrained
SLOPE_LO_COEF, SLOPE_HI_COEF = -9.0, -4.0
INTER_LO_OFF, INTER_HI_OFF = -24.0, -14.0

# The 13 sampling params in Stan declaration order (constrained CSV column names).
CSV_COLS = [
    "slope_std", "intercept_std.1", "sigma_int_x",
    "w.1", "w.2", "w.3",
    "delta_c", "mu_c", "delta_g", "mu_g",
    "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g",
]


def _logit_unit_interval(x, lo, hi):
    """Stan's unconstrained value for a <lower=lo, upper=hi> variable."""
    u = np.clip((x - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return np.log(u / (1 - u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run name (output/<run>/)")
    ap.add_argument("--stepsize", type=float, default=0.1,
                    help="Initial stepsize recorded in the metric file (default 0.1)")
    args = ap.parse_args()

    run = args.run
    pf_csv = f"output/{run}/pathfinder.csv"
    map_json = f"output/{run}/init_MAP.json"
    out_json = f"output/{run}/pf_metric.json"

    mp = json.load(open(map_json))
    mean_x, sd_x = float(mp["mean_x"]), float(mp["sd_x"])

    rows = [r for r in csv.reader(open(pf_csv)) if r and not r[0].startswith("#")]
    hdr = rows[0]
    if "intercept_std.2" in hdr:
        raise SystemExit("This script assumes N_bins==1; found intercept_std.2.")
    col = lambda n: np.array([float(r[hdr.index(n)]) for r in rows[1:]])

    slope = col("slope_std")
    inter = col("intercept_std.1")

    # Transform each param to Stan's unconstrained scale, in declaration order.
    U = []
    U.append(_logit_unit_interval(slope, SLOPE_LO_COEF * sd_x, SLOPE_HI_COEF * sd_x))
    U.append(_logit_unit_interval(inter, INTER_LO_OFF + slope * mean_x / sd_x,
                                          INTER_HI_OFF + slope * mean_x / sd_x))
    U.append(_logit_unit_interval(col("sigma_int_x"), 0.0, 1.0))
    for n in ("w.1", "w.2", "w.3",                       # native unconstrained
              "delta_c", "mu_c", "delta_g", "mu_g",
              "alpha_kcorr_r", "alpha_kcorr_z", "alpha_kcorr_g"):
        U.append(col(n))

    X = np.vstack(U).T                                   # (draws, 13)
    if X.shape[0] < 20:
        raise SystemExit(f"Only {X.shape[0]} Pathfinder draws; need >=20 for a covariance.")

    C = np.cov(X.T)
    # Guarantee positive-definiteness (small ridge if a direction collapsed).
    eig = np.linalg.eigvalsh(C)
    if eig.min() <= 0:
        C = C + np.eye(C.shape[0]) * (abs(eig.min()) + 1e-8)
    w = np.linalg.eigvalsh(C)

    json.dump({"stepsize": args.stepsize, "inv_metric": C.tolist()}, open(out_json, "w"))
    print(f"Pathfinder metric written to {out_json}")
    print(f"  full {C.shape[0]}x{C.shape[0]} dense_e metric from {X.shape[0]} draws; "
          f"cond {w.max()/w.min():.1f}, PD={bool(w.min() > 0)}")


if __name__ == "__main__":
    main()
