#!/usr/bin/env python3
"""check_rhat.py — convergence gate on rank-normalized split R-hat.

CmdStan's own `diagnose` reports R-hat above 1.01, but it is advisory: it
exits 0 even when it warns, so `set -e` never trips and step8 runs on the
`afterok` regardless. This script turns R-hat into an explicit, machine-
readable verdict at a threshold we choose.

Scope: GLOBAL parameters only. The per-galaxy latents are not saved in these
models' CSV output, so in practice "global" is every sampled quantity except
the sampler's own diagnostics (`lp__`, `accept_stat__`, `stepsize__`,
`treedepth__`, `n_leapfrog__`, `divergent__`, `energy__`). Indexed parameters
(`intercept[1]`, `w[3]`, ...) count as global while their dimension stays at
or below --max-dim, which keeps any future per-object array out of the gate
instead of silently drowning it in thousands of rows.

Precision matters here: stansummary defaults to 2 significant figures, which
prints R-hat as "1.0" or "1.1" and cannot resolve a 1.02 threshold. Callers
must generate the summary with --sig_figs >= 4; this script refuses to judge
a summary whose R_hat column is too coarse to compare.

Exit status is 0 by default even on FAIL, so a failing fit still produces its
catalogue in step8. Pass --strict to make a FAIL exit 1 (and, under `set -e`,
hold the dependent jobs).
"""
import argparse
import json
import math
import os
import re
import sys

_SAMPLER_COLS = {
    "lp__", "accept_stat__", "stepsize__", "treedepth__",
    "n_leapfrog__", "divergent__", "energy__",
}
_INDEX_RE = re.compile(r"^(?P<base>[^\[]+)\[(?P<idx>[0-9,]+)\]$")


def parse_summary(path):
    """Yield (name, base, max_index, r_hat) for every parameter row."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            # A parameter row: name plus the numeric columns ending in R_hat.
            if len(parts) < 10 or parts[0].endswith("__") or parts[0] in _SAMPLER_COLS:
                continue
            name = parts[0]
            try:
                r_hat = float(parts[-1])
            except ValueError:
                continue  # header or prose line
            m = _INDEX_RE.match(name)
            if m:
                base = m.group("base")
                max_idx = max(int(i) for i in m.group("idx").split(","))
            else:
                base, max_idx = name, 1
            rows.append((name, base, max_idx, r_hat))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="run name; reads output/<run>/stansummary.txt")
    p.add_argument("--summary", help="explicit path to a stansummary file "
                                     "(overrides --run)")
    p.add_argument("--threshold", type=float, default=1.02,
                   help="fail if any global parameter exceeds this R-hat "
                        "(default 1.02)")
    p.add_argument("--max-dim", type=int, default=10,
                   help="an indexed parameter counts as global while its "
                        "dimension is <= this (default 10)")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 on FAIL instead of 0")
    p.add_argument("--out", help="where to write the JSON verdict "
                                 "(default output/<run>/rhat_check.json)")
    args = p.parse_args()

    if not args.summary and not args.run:
        p.error("one of --run or --summary is required")
    summary = args.summary or os.path.join("output", args.run, "stansummary.txt")
    if not os.path.exists(summary):
        print(f"check_rhat: ERROR no such summary: {summary}", file=sys.stderr)
        return 2

    rows = parse_summary(summary)
    if not rows:
        print(f"check_rhat: ERROR no parameter rows parsed from {summary}",
              file=sys.stderr)
        return 2

    # Refuse to judge a summary printed at stansummary's 2-sig-fig default:
    # every R_hat would read as 1.0 / 1.1 and a 1.02 test would be meaningless.
    finite = [r for *_, r in rows if math.isfinite(r)]
    if finite and all(round(r, 1) == r for r in finite):
        print("check_rhat: ERROR the R_hat column is quantized to 1 decimal "
              "place, so it cannot be compared against "
              f"{args.threshold}. Regenerate with: stansummary --sig_figs=6",
              file=sys.stderr)
        return 2

    glob = [(n, b, i, r) for (n, b, i, r) in rows if i <= args.max_dim]
    skipped = sorted({b for (_, b, i, _) in rows if i > args.max_dim})

    bad = sorted([(n, r) for (n, _, _, r) in glob if math.isfinite(r)
                  and r > args.threshold], key=lambda t: -t[1])
    nan = sorted([n for (n, _, _, r) in glob if not math.isfinite(r)])
    worst = max((r for (_, _, _, r) in glob if math.isfinite(r)), default=float("nan"))
    ok = not bad and not nan

    verdict = {
        "pass": ok,
        "threshold": args.threshold,
        "summary": summary,
        "n_global_parameters": len(glob),
        "worst_r_hat": worst,
        "violations": [{"parameter": n, "r_hat": r} for n, r in bad],
        "non_finite": nan,
        "skipped_high_dimension": skipped,
    }

    out = args.out or (os.path.join("output", args.run, "rhat_check.json")
                       if args.run else None)
    if out:
        with open(out, "w") as fh:
            json.dump(verdict, fh, indent=2)

    tag = "PASS" if ok else "FAIL"
    print(f"check_rhat: {tag}  worst R_hat = {worst:.4f} over "
          f"{len(glob)} global parameters (threshold {args.threshold})")
    for n, r in bad:
        print(f"  R_hat {r:.4f}  {n}")
    for n in nan:
        print(f"  R_hat non-finite  {n}")
    if skipped:
        print(f"  (excluded as high-dimension: {', '.join(skipped)})")
    if out:
        print(f"check_rhat: wrote {out}")

    return 1 if (args.strict and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())
