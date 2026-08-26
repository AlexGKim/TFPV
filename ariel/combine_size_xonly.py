#!/usr/bin/env python3
"""
combine_size_xonly.py — Combine two populations' color_xonly outputs
(catalog + covariance) into one self-consistent pair of files, with the
population labels supplied on the command line.

This is combine_color_xonly.py with the 'spiral'/'irregular' labels lifted out
into arguments, so the size chain (large / small, see DR2_SIZESPLIT.md) can use
it. The physics is unchanged and is not reimplemented: the per-population
loading, the d_err_r resolution and the systematic off-diagonal terms are
imported from combine_color_xonly.py, so a fix there is a fix here too.

Why the cross-terms exist: each population is a fully independent Stan run, so
the cross-population *statistical* covariance is legitimately zero -- but the
per-galaxy systematic terms (PHOTSYS N/S calibration floor, internal-dust slope
uncertainty) are shared survey-wide systematics. Naive block-diagonal
concatenation would silently drop them.

Usage:
  python combine_size_xonly.py \
      --run-a DR2_TF_disks_v5_2color_large --label-a large \
      --run-b DR2_TF_disks_v5_2color_small --label-b small \
      --out-run DR2_TF_disks_v5_2color_combined \
      --catalog-name DESI-DR2_TF_pv_cat_v5_size.fits
"""

import argparse
import os

import h5py
import numpy as np
from astropy.table import vstack

from color_predict import _systematic_offdiag_terms
from combine_color_xonly import load_population_main


def combine(run_a, run_b, out_run, label_a="a", label_b="b",
            catalog_name="color_xonly_catalog.fits",
            config_a=None, config_b=None):
    out_dir = os.path.join("output", out_run)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {label_a} population from output/{run_a} ...")
    t_a, cov_a, ana_a, d_err_r_a = load_population_main(
        os.path.join("output", run_a), pipeline_config=config_a
    )
    print(f"  MAIN rows: {len(t_a)}, d_err_r={d_err_r_a:.8f}")

    print(f"Loading {label_b} population from output/{run_b} ...")
    t_b, cov_b, ana_b, d_err_r_b = load_population_main(
        os.path.join("output", run_b), pipeline_config=config_b
    )
    print(f"  MAIN rows: {len(t_b)}, d_err_r={d_err_r_b:.8f}")

    if not np.isclose(d_err_r_a, d_err_r_b, rtol=1e-10, atol=0):
        raise ValueError(
            "d_err_r mismatch between populations — the cross-population "
            "dust systematic term is only physically valid if both runs "
            f"share the same dust-slope uncertainty. Got {label_a}={d_err_r_a!r} "
            f"vs {label_b}={d_err_r_b!r}. Check 'dust_pickle' in "
            f"output/{run_a}/config.json and output/{run_b}/config.json."
        )
    d_err_r = d_err_r_a

    n_a = len(t_a)
    n_b = len(t_b)
    n_tot = n_a + n_b

    # --- Combined catalog ---
    t_a = t_a.copy()
    t_b = t_b.copy()
    dt = f"U{max(len(label_a), len(label_b))}"
    t_a["POPULATION"] = np.full(n_a, label_a, dtype=dt)
    t_b["POPULATION"] = np.full(n_b, label_b, dtype=dt)
    combined_table = vstack([t_a, t_b], metadata_conflicts="silent")

    cat_out = os.path.join(out_dir, catalog_name)
    combined_table.write(cat_out, overwrite=True)
    print(f"Written {n_tot} rows to {cat_out}")

    # --- Combined covariance ---
    ba_col = "BA" if "BA" in combined_table.colnames else "BA_RATIO"
    ba_all = np.asarray(combined_table[ba_col], dtype=float)
    photsys_all = np.asarray(combined_table["PHOTSYS"])
    # PHOTSYS_ERR, when the catalogs carry it, is the per-galaxy calibration
    # uncertainty and takes precedence over the 'N'/'S' flag. DESI catalogs (the
    # only ones this script can combine — it needs SGA_ID) do not have it, so
    # this is normally None; carried through so the cross-population block is
    # built from the same v_phot the per-population blocks used.
    photsys_err_all = (
        np.asarray(combined_table["PHOTSYS_ERR"], dtype=float)
        if "PHOTSYS_ERR" in combined_table.colnames else None
    )
    sga_id_all = np.asarray(combined_table["SGA_ID"], dtype=float)

    v_dust_all, v_phot_all = _systematic_offdiag_terms(
        ba_all, photsys_all, d_err_r=d_err_r, photsys_err=photsys_err_all
    )

    cov_combined = np.zeros((n_tot, n_tot), dtype=np.float32)
    cov_combined[:n_a, :n_a] = cov_a
    cov_combined[n_a:, n_a:] = cov_b

    v_dust_a, v_dust_b = v_dust_all[:n_a], v_dust_all[n_a:]
    v_phot_a, v_phot_b = v_phot_all[:n_a], v_phot_all[n_a:]

    cross = (
        np.outer(v_dust_a, v_dust_b) + np.outer(v_phot_a, v_phot_b)
    ).astype(np.float32)
    cov_combined[:n_a, n_a:] = cross
    cov_combined[n_a:, :n_a] = cross.T

    analysis_combined = np.concatenate([ana_a, ana_b])
    population_combined = np.concatenate(
        [np.zeros(n_a, dtype=np.int8), np.ones(n_b, dtype=np.int8)]
    )

    # --- Sanity checks ---
    diag_expected = np.concatenate([np.diag(cov_a), np.diag(cov_b)])
    diag_actual = np.diag(cov_combined)
    assert np.array_equal(diag_actual, diag_expected.astype(np.float32)), (
        "Diagonal of combined cov does not exactly match the concatenated "
        "input diagonals — cross-terms must be off-diagonal only."
    )
    print("PASS: diagonal preserved exactly")

    assert np.allclose(cov_combined, cov_combined.T, atol=0), (
        "Combined covariance is not symmetric."
    )
    print("PASS: combined covariance is symmetric")

    assert cov_combined.shape == (n_tot, n_tot)
    assert len(combined_table) == n_tot
    assert np.array_equal(sga_id_all, np.concatenate([
        np.asarray(t_a["SGA_ID"], dtype=float),
        np.asarray(t_b["SGA_ID"], dtype=float),
    ]))
    print(f"PASS: shapes/row-order consistent (G={n_tot})")

    # Spot-check: first North-flagged row in each population, if any
    n_idx_a = np.flatnonzero(photsys_all[:n_a] == "N")
    n_idx_b = np.flatnonzero(photsys_all[n_a:] == "N")
    if len(n_idx_a) and len(n_idx_b):
        i = int(n_idx_a[0])
        j = int(n_idx_b[0]) + n_a
        expected = v_dust_all[i] * v_dust_all[j] + v_phot_all[i] * v_phot_all[j]
        actual = cov_combined[i, j]
        assert np.isclose(actual, expected, rtol=1e-5), (
            f"Cross-term spot-check failed at (i={i}, j={j}): "
            f"expected {expected}, got {actual}"
        )
        print(
            f"PASS: cross-term spot-check at ({label_a} row {i}, {label_b} row "
            f"{j - n_a}) matches independent computation ({actual:.6e})"
        )
    else:
        print("SKIP: cross-term spot-check (no North-flagged pair found)")

    cov_name = os.path.splitext(catalog_name)[0] + "_cov.h5"
    cov_out = os.path.join(out_dir, cov_name)
    with h5py.File(cov_out, "w") as hf:
        hf.create_dataset(
            "cov", data=cov_combined, compression="gzip", compression_opts=3,
            chunks=(min(512, n_tot), n_tot),
        )
        hf.create_dataset("analysis", data=analysis_combined)
        hf.create_dataset("population", data=population_combined)
        hf.create_dataset("sga_id", data=sga_id_all)
        # Record the dust value the whole matrix was built with, matching the
        # per-population inputs' own attribute — this delivered product is the
        # end of the chain and must carry the same provenance.
        hf.attrs["d_err_r"] = float(d_err_r)
        hf.attrs["population_labels"] = f"0={label_a}, 1={label_b}"
        hf.attrs["row_order"] = (
            f"MAIN=True rows of {catalog_name} in this directory, "
            f"{label_a} rows first (in {label_a} catalog order), then "
            f"{label_b} rows (in {label_b} catalog order)"
        )
    print(f"Written combined covariance ({n_tot}, {n_tot}) to {cov_out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-a", default="DR2_TF_disks_v5_2color_large")
    parser.add_argument("--run-b", default="DR2_TF_disks_v5_2color_small")
    parser.add_argument("--label-a", default="large",
                        help="population label for --run-a (goes in the POPULATION "
                             "column and the HDF5 attributes)")
    parser.add_argument("--label-b", default="small", help="population label for --run-b")
    parser.add_argument("--out-run", default="DR2_TF_disks_v5_2color_combined")
    parser.add_argument("--catalog-name", default="DESI-DR2_TF_pv_cat_v5_size.fits")
    parser.add_argument(
        "--config-a", default=None,
        help="Pipeline config for --run-a. Only consulted if that run's "
             "color_xonly_cov.h5 predates the d_err_r attribute, in which case "
             "it is overlaid on output/<run>/config.json the way color_predict.py "
             "does — without it the fallback lands on the iron dust default.",
    )
    parser.add_argument("--config-b", default=None, help="Pipeline config for --run-b; see --config-a.")
    args = parser.parse_args()

    combine(args.run_a, args.run_b, args.out_run,
            label_a=args.label_a, label_b=args.label_b,
            catalog_name=args.catalog_name,
            config_a=args.config_a, config_b=args.config_b)


if __name__ == "__main__":
    main()
