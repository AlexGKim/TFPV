#!/usr/bin/env python
"""Analysis (prediction) step for a v9 run: catalog + posterior-predictive covariance.

color_predict.py's __main__ also calls DESI_color(), which builds colour-model
diagnostic GRIDS and hard-requires delta_c/mu_c/alpha_kcorr_z/g. Those do not exist
in sb_v9.stan and the grids are plots, not science output, so this driver calls the
two writers that produce the actual products and skips the diagnostics.

The x-only predictor is the one that produced the published v5b catalog. For v9 it
additionally carries the inclination term: y_TF = intercept + gamma_inc*q + slope*x,
with q = 2.5 log10(b/a), applied inside color_predict.py via q_star.
"""
import argparse
import json

import color_predict as cp


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default="v9")
    p.add_argument("--no-catalog", action="store_true")
    p.add_argument("--no-cov", action="store_true")
    args = p.parse_args()

    cfg = json.load(open(f"{args.run_dir}/config.json"))
    fits_path = cfg["fits_file"]
    print(f"run_dir  {args.run_dir}\nmodel    {args.model}\nfits     {fits_path}\n")

    if not args.no_catalog:
        cp.write_desi_catalog_color_xonly(args.run_dir, fits_path, cfg=cfg,
                                          model=args.model)
    if not args.no_cov:
        cp.write_cov_color_xonly(args.run_dir, fits_path, cfg=cfg, model=args.model)


if __name__ == "__main__":
    main()
