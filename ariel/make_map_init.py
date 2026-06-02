#!/usr/bin/env python
"""Convert optimizer output CSV to MCMC warm-start init file."""
import argparse, json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='Run name (output/<run>/)')
    parser.add_argument('--config', help='Config JSON (alternative to --run)')
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
        elif k in df.columns:
            new[k] = float(row[k])
        else:
            new[k] = old[k]

    with open(out_json, 'w') as f:
        json.dump(new, f, indent=2)
    print(f'MAP init written to {out_json}')

if __name__ == '__main__':
    main()
