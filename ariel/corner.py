#!/usr/bin/env python3
"""
Create corner plots from Stan MCMC output files using ChainConsumer.

Reads one or more sets of Stan CSV output files (matched by glob patterns) and
produces a corner plot of the key parameters: slope, intercept, sigma_int_x,
sigma_int_y (and optionally theta_int).

Usage
-----
As a script (command line)::

    # Single chain
    python corner.py 'DESI_base_?.csv' --output DESI_base.png

    # Two chains overlaid
    python corner.py 'DESI_base_?.csv' 'MOCK_n10000_base_?.csv' --output compare.png

    # Three chains with truth lines
    python corner.py 'DESI_base_?.csv' 'DESI_normal_?.csv' 'MOCK_n10000_base_?.csv' \\
        --output compare.png --theta-int \\
        --truth slope=-8.0 intercept=-20.0 sigma_int_x=0.03 sigma_int_y=0.03

    # Override legend names (one --name per infile, in order)
    python corner.py 'DESI_base_?.csv' 'MOCK_n10000_base_?.csv' \\
        --name 'DESI' --name 'Mock' --output compare.png

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
    """
    Load a single Stan CSV output file, skipping comment lines.

    Parameters
    ----------
    filename : str
        Path to the Stan CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the MCMC samples.
    """
    return pd.read_csv(filename, comment='#')


def load_params(file_pattern, include_theta_int=False):
    """
    Load and combine Stan CSV files matching *file_pattern* and extract the
    standard TF parameters.

    Parameters
    ----------
    file_pattern : str
        Glob pattern to match Stan output files.
    include_theta_int : bool
        Whether to include ``theta_int`` in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame of extracted parameter samples with columns
        ``slope``, ``intercept``, ``sigma_int_x``, ``sigma_int_y``
        (and optionally ``theta_int``).

    Raises
    ------
    FileNotFoundError
        If no files match *file_pattern*.
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

    params = {
        'slope':       combined['slope'].values,
        'intercept':   combined['intercept.1'].values,
        'sigma_int_x': combined['sigma_int_x'].values,
        'sigma_int_y': combined['sigma_int_y'].values,
    }

    if include_theta_int:
        params['theta_int'] = combined['theta_int'].values

    return pd.DataFrame(params)


def _print_stats(df, label):
    """Print mean / std / median for every column of *df*."""
    print(f"\nParameter statistics ({label}):")
    for col in df.columns:
        v = df[col].values
        print(f"  {col}:  mean={np.mean(v):.6f}  std={np.std(v):.6f}  "
              f"median={np.median(v):.6f}")


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def create_corner_plot(file_patterns, output_file='corner_plot.png',
                       include_theta_int=False, truth_values=None,
                       names=None):
    """
    Create a corner plot from one or more sets of Stan output files.

    Parameters
    ----------
    file_patterns : str or list of str
        Glob pattern(s) for the Stan CSV files.  A single string is treated as
        a list of one element.
    output_file : str
        Path to save the output PNG (default: ``'corner_plot.png'``).
    include_theta_int : bool
        Include ``theta_int`` in the plot (default: ``False``).
    truth_values : dict, optional
        Mapping of parameter name → true value, drawn as reference lines.
        Example: ``{"slope": -8.0, "intercept": -20.0}``.
    names : list of str, optional
        Legend labels, one per entry in *file_patterns*.  Any missing entries
        default to the corresponding pattern with ``.csv`` stripped.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Normalise to a list
    if isinstance(file_patterns, str):
        file_patterns = [file_patterns]

    # Build / fill-in names list
    if names is None:
        names = [None] * len(file_patterns)
    else:
        # Pad with None if fewer names than patterns were supplied
        names = list(names) + [None] * (len(file_patterns) - len(names))

    resolved_names = [
        (n if n is not None else pat.replace('.csv', ''))
        for pat, n in zip(file_patterns, names)
    ]

    c = ChainConsumer()

    first_df = None
    for i, (pat, name) in enumerate(zip(file_patterns, resolved_names), start=1):
        print(f"\n=== Chain {i}: {pat} ===")
        df = load_params(pat, include_theta_int=include_theta_int)
        _print_stats(df, name)
        c.add_chain(Chain(samples=df, name=name))
        if first_df is None:
            first_df = df

    # --- truth lines ---
    if truth_values is not None:
        c.add_truth(Truth(location=truth_values))

    # --- plot ---
    fig = c.plotter.plot()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCorner plot saved to: {output_file}")

    # --- summary (based on first chain's columns) ---
    print("\nSummary statistics:")
    summary = c.analysis.get_summary()
    for param in first_df.columns:
        if param in summary:
            print(f"\n{param}:")
            for key, value in summary[param].items():
                print(f"  {key}: {value}")

    return fig


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_truth(items):
    """
    Parse a list of ``key=value`` strings into a float-valued dict.

    Parameters
    ----------
    items : list of str or None

    Returns
    -------
    dict or None
    """
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
             "(quote each pattern to prevent shell expansion, "
             "e.g. 'DESI_base_?.csv' 'MOCK_n10000_base_?.csv').",
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
        help="Legend label for a chain (repeat once per infile, in order). "
             "Defaults to the pattern with .csv stripped.",
    )
    p.add_argument(
        '--theta-int',
        action='store_true',
        default=False,
        help="Include theta_int in the corner plot.",
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

    # ------------------------------------------------------------------
    # If arguments are passed on the command line, use the CLI.
    # Otherwise fall through to the hard-coded __main__ configuration.
    # ------------------------------------------------------------------
    if len(sys.argv) > 1:
        args = _build_parser().parse_args()
        create_corner_plot(
            file_patterns=args.infiles,
            output_file=args.output,
            include_theta_int=args.theta_int,
            truth_values=_parse_truth(args.truth),
            names=args.names,
        )
    else:
        # ------------------------------------------------------------------
        # Hard-coded configuration — edit these variables as needed.
        # ------------------------------------------------------------------
        infiles = [
            # "MOCK_n10000_base_?.csv",
            # "MOCK_n10000_max0.5_base_?.csv",
            # "MOCK_n10000_min0.5_base_?.csv",
            "MOCK_normal_?.csv",
        ]
        # infiles = [
        #     "DESI_base_?.csv",
        #     "DESI_normal_?.csv",
        # ]
        outfile = "MOCK_normal.png"
        # outfile = "DESI_base_vs_normal.png"

        truth = {
            "slope":       -8.0,
            "intercept":   -20.0,
            "sigma_int_x":  0.03,
            "sigma_int_y":  0.03,
        }

        create_corner_plot(
            file_patterns=infiles,
            output_file=outfile,
            truth_values=truth,
        )
