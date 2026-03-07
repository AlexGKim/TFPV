#!/usr/bin/env python3
"""
Create corner plots from Stan MCMC output files using ChainConsumer.

Reads one or more sets of Stan CSV output files (matched by glob patterns) and
produces a corner plot of selected parameters, automatically skipping any
parameters that are not present in the CSV output.

Usage
-----
As a script (command line)::

    python corner.py 'DESI_tophat_?.csv' --output DESI_tophat.png
    python corner.py 'DESI_tophat_?.csv' 'ariel_n10000_tophat_?.csv' --output compare.png

As a module (edit the __main__ block at the bottom)::

    python corner.py
"""

import argparse
import glob

import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer, Truth


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_stan_csv(filename):
    """Load a single Stan CSV output file, skipping comment lines."""
    return pd.read_csv(filename, comment='#')


def _first_existing_column(df, candidates):
    """Return the first column name in *candidates* that exists in df.columns, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_params(file_pattern):
    """
    Load and combine Stan CSV files matching *file_pattern* and extract a set of
    parameters if present. Missing parameters are skipped (not an error).
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")

    print(f"Found {len(files)} file(s) for pattern '{file_pattern}':")
    all_dfs = []
    for f in files:
        df = load_stan_csv(f)
        all_dfs.append(df)
        print(f"  Loaded {f}: {len(df)} samples")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  → {len(combined)} total samples")

    # Desired plot parameters (key = label used in corner plot)
    # Values are candidate CSV column names to search for.
    wanted = {
        "slope":        ["slope"],
        "intercept":    ["intercept.1", "intercept[1]", "intercept"],  # common Stan CSV variants
        "sigma_int_x":  ["sigma_int_x"],
        "sigma_int_y":  ["sigma_int_y"],
        "mu_{y_TF}":    ["mu_y_TF"],
        "tau":          ["tau"],
        # Add more here if you like; they will be included only if present.
        # "theta_int":  ["theta_int"],
    }

    params = {}
    missing = []
    found_cols = {}

    for label, candidates in wanted.items():
        col = _first_existing_column(combined, candidates)
        if col is None:
            missing.append(label)
            continue
        found_cols[label] = col
        params[label] = combined[col].to_numpy()

    if missing:
        print("  Skipping missing parameter(s): " + ", ".join(missing))
    if not params:
        raise ValueError(
            f"No requested parameters were found in files matching pattern: {file_pattern}\n"
            f"Available columns include (first file): {list(all_dfs[0].columns)[:30]} ..."
        )

    return pd.DataFrame(params)


def _print_stats(df, label):
    """Print mean / std / median for every column of *df*."""
    print(f"\nParameter statistics ({label}):")
    for col in df.columns:
        v = df[col].values
        print(f"  {col}:  mean={np.mean(v):.6f}  std={np.std(v):.6f}  median={np.median(v):.6f}")


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def create_corner_plot(file_patterns, output_file='corner_plot.png',
                       truth_values=None,
                       names=None):
    """
    Create a corner plot from one or more sets of Stan output files.

    If multiple chains are overlaid, only the *intersection* of parameters
    available in all chains is plotted (so ChainConsumer always gets consistent
    columns).
    """
    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    # Build / fill-in names list
    if names is None:
        names = [None] * len(file_patterns)
    else:
        names = list(names) + [None] * (len(file_patterns) - len(names))

    resolved_names = [
        (n if n is not None else pat.replace('.csv', ''))
        for pat, n in zip(file_patterns, names)
    ]

    # Load all chains first (so we can take common columns)
    dfs = []
    for i, (pat, name) in enumerate(zip(file_patterns, resolved_names), start=1):
        print(f"\n=== Chain {i}: {pat} ===")
        df = load_params(pat)
        _print_stats(df, name)
        dfs.append(df)

    # Compute common columns across all chains (required for an overlay plot)
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)

    if not common:
        raise ValueError(
            "No common parameters across the provided chains after checking for existence.\n"
            "Tip: run each pattern alone to see what parameters it contains."
        )

    # Keep a nice, stable plotting order
    preferred_order = ["slope", "intercept", "sigma_int_x", "sigma_int_y", "mu_{y_TF}", "tau"]
    ordered = [p for p in preferred_order if p in common] + sorted(common - set(preferred_order))

    # Reduce each df to the common ordered set
    dfs = [df[ordered].copy() for df in dfs]

    # Filter truth values to plotted params only
    if truth_values is not None:
        truth_values = {k: v for k, v in truth_values.items() if k in ordered}
        if not truth_values:
            truth_values = None

    c = ChainConsumer()
    for df, name in zip(dfs, resolved_names):
        c.add_chain(Chain(samples=df, name=name))

    if truth_values is not None:
        c.add_truth(Truth(location=truth_values))

    fig = c.plotter.plot()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCorner plot saved to: {output_file}")

    # Summary (for the plotted params)
    print("\nSummary statistics (plotted parameters):")
    summary = c.analysis.get_summary()
    for param in ordered:
        if param in summary:
            print(f"\n{param}:")
            for key, value in summary[param].items():
                print(f"  {key}: {value}")

    return fig


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_truth(items):
    """Parse a list of ``key=value`` strings into a float-valued dict."""
    if not items:
        return None
    result = {}
    for item in items:
        key, _, val = item.partition('=')
        result[key.strip()] = float(val.strip())
    return result


def _build_parser():
    p = argparse.ArgumentParser(
        description="Create a corner plot from Stan MCMC CSV output files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        'infiles',
        nargs='+',
        metavar='PATTERN',
        help="One or more glob patterns for Stan CSV files "
             "(quote each pattern to prevent shell expansion).",
    )
    p.add_argument(
        '--output', '-o',
        default='corner_plot.png',
        metavar='FILE',
        help="Output PNG filename (default: corner_plot.png).",
    )
    p.add_argument(
        '--name',
        dest='names',
        action='append',
        default=None,
        metavar='LABEL',
        help="Legend label for a chain (repeat once per infile, in order).",
    )
    p.add_argument(
        '--truth',
        nargs='+',
        metavar='PARAM=VALUE',
        default=None,
        help="True parameter values as key=value pairs, e.g. "
             "--truth slope=-8.0 intercept=-20.0 sigma_int_x=0.03 sigma_int_y=0.03",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        args = _build_parser().parse_args()
        create_corner_plot(
            file_patterns=args.infiles,
            output_file=args.output,
            truth_values=_parse_truth(args.truth),
            names=args.names,
        )
    else:

        truth = {
            "slope":       -8.0,
            "intercept":   -20.0,
            "sigma_int_x":  0.03,
            "sigma_int_y":  0.03,
        }
        infiles = [
            # "ariel_tophat_?.csv",
            # "ariel_normal_?.csv",
            "DESI_tophat_?.csv",
            "DESI_normal_?.csv",
        ]
        outfile = "temp.png"
        truth = None


        create_corner_plot(
            file_patterns=infiles,
            output_file=outfile,
            truth_values=truth,
        )