#!/usr/bin/env python
"""Extract sample covariance from short MCMC run as inverse mass matrix."""
import argparse, json
import numpy as np
import pandas as pd

SAMPLING_PARAMS = [
    'slope_std', 'intercept_std.1', 'sigma_int_x', 'sigma_int_y',
    'log_sigma_int_z', 'gamma_tau_c', 'delta_c', 'mu_c', 'log_tau_c',
    'gamma_tau_g', 'delta_g', 'mu_g', 'log_tau_g', 'log_sigma_int_g',
    'alpha_kcorr_r', 'alpha_kcorr_z', 'alpha_kcorr_g'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Run name (output/<run>/)')
    parser.add_argument('--csv', help='Override input CSV path')
    parser.add_argument('--out', help='Override output metric.json path')
    args = parser.parse_args()

    csv_path = args.csv or f'output/{args.run}/2color_metric_build.csv'
    out_path = args.out or f'output/{args.run}/metric.json'

    df = pd.read_csv(csv_path, comment='#')
    X   = df[SAMPLING_PARAMS].to_numpy(dtype=float)
    cov = np.cov(X.T)

    metric = {'inv_metric': cov.tolist()}
    with open(out_path, 'w') as f:
        json.dump(metric, f)

    print(f'Metric written to {out_path}')
    print(f'Condition number: {np.linalg.cond(cov):.1f}')

if __name__ == '__main__':
    main()
