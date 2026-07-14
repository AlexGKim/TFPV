#!/usr/bin/env python
"""Convert optimizer output CSV to MCMC warm-start init file."""
import argparse, json, math
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Run name (output/<run>/)')
    parser.add_argument('--config', help='Config JSON (alternative to --run)')
    parser.add_argument(
        '--sigma-floor', type=float, default=0.01,
        help='Minimum starting value for sigma_int_* / exp(log_sigma_int_*) '
             'parameters (default: 0.01). MAP often drives these to ~0, which '
             'starts HMC in the degenerate boundary regime and slows warmup; '
             'flooring away from 0 gives the sampler room to move. Set to 0 '
             'to disable.'
    )
    args = parser.parse_args()

    run = args.run
    if args.config and not args.run:
        run = json.load(open(args.config))['run']

    optimize_csv = f'output/{run}/optimize.csv'
    init_json    = f'output/{run}/init.json'
    out_json     = f'output/{run}/init_MAP.json'

    df  = pd.read_csv(optimize_csv, comment='#')
    row = df.iloc[0]
    old = json.load(open(init_json))

    new = {}
    for k in old.keys():
        if k == 'intercept_std':
            cols = sorted([c for c in df.columns if c.startswith('intercept_std.')],
                          key=lambda s: int(s.split('.')[1]))
            new[k] = [float(row[c]) for c in cols]
        elif k == 'S_scale' and 'S_scale.1' in df.columns:
            # [2COLOR] free intrinsic-covariance scales -> length-3 vector
            new[k] = [float(row[f'S_scale.{i}']) for i in (1, 2, 3)]
        elif k == 'S_Lcorr' and 'S_Lcorr.1.1' in df.columns:
            # [2COLOR] intrinsic-correlation Cholesky -> 3x3 (row-major) matrix
            new[k] = [[float(row[f'S_Lcorr.{i}.{j}']) for j in (1, 2, 3)]
                      for i in (1, 2, 3)]
        elif k in df.columns:
            new[k] = float(row[k])
        else:
            new[k] = old[k]

    # MAP frequently drives sigma_int_* parameters to (near) their zero
    # boundary; starting HMC there puts it in a degenerate funnel that's
    # slow to escape. Floor them away from 0 so warmup has room to explore.
    floor = args.sigma_floor
    if floor > 0:
        for k in list(new.keys()):
            if k.startswith('sigma_int_') and isinstance(new[k], (int, float)):
                if new[k] < floor:
                    print(f'  floor: {k} {new[k]:.6g} -> {floor}')
                    new[k] = floor
            elif k.startswith('log_sigma_int_') and isinstance(new[k], (int, float)):
                log_floor = math.log(floor)
                if new[k] < log_floor:
                    print(f'  floor: {k} {new[k]:.6g} -> {log_floor:.6g}  (sigma={floor})')
                    new[k] = log_floor

    with open(out_json, 'w') as f:
        json.dump(new, f, indent=2)
    print(f'MAP init written to {out_json}')

if __name__ == '__main__':
    main()
