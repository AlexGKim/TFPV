#!/usr/bin/env python3
"""
combine_color_xonly.py — Combine two populations' color_xonly outputs
(catalog + covariance) into one self-consistent pair of files.

Each population (spiral, irregular) is fit as a fully independent Stan run,
so the cross-population *statistical* covariance is legitimately zero — but
color_predict.py's per-galaxy systematic terms (PHOTSYS N/S calibration
floor, internal-dust slope uncertainty) are shared systematics that apply
across the whole survey, not just within one population. Naive block-
diagonal concatenation of the two color_xonly_cov.h5 files would silently
drop these cross-population correlations. This script fills them in.

Usage:
  python combine_color_xonly.py \
      --spiral-run DR2_TF_spirals_v5_2color_spiral \
      --irregular-run DR2_TF_irrs_v5_2color_irregular \
      --out-run DR2_TF_v5_2color_combined
"""

import argparse
import json
import os

import h5py
import numpy as np
from astropy.table import Table, vstack

from color_predict import _systematic_offdiag_terms, resolve_d_err_r


def _resolve_d_err_r(run_dir, cov_path=None, pipeline_config=None):
    """Return the d_err_r that step 8 actually used for this population.

    Prefers the ``d_err_r`` attribute step 8 records on the covariance HDF5.
    That is authoritative: the per-population blocks of this combined matrix
    already have that value baked in, so the cross-population terms computed
    here must match it exactly. Re-deriving it independently is what left the
    v5b product with loa dust (0.2173) in the population blocks and iron dust
    (0.1768) in the cross terms.

    Falls back to re-resolving from the run's config/FITS header for
    covariances written before the attribute existed, and says so. That fallback
    overlays ``pipeline_config`` the same way ``color_predict.py`` does, because
    ``output/<run>/config.json`` records only what step 4 knew: all three DR2 run
    dirs have ``dust_pickle: null``, and the DR2 FITS files carry no dust
    keywords, so without the overlay this path lands on the iron default 0.1768
    -- exactly the drift this function exists to prevent.
    """
    if cov_path and os.path.exists(cov_path):
        with h5py.File(cov_path, "r") as hf:
            if "d_err_r" in hf.attrs:
                val = float(hf.attrs["d_err_r"])
                print(f"  d_err_r={val:.8f} (from {os.path.basename(cov_path)} attrs — "
                      f"as used by step 8)")
                return val
        print(f"  WARNING: {os.path.basename(cov_path)} has no 'd_err_r' attribute "
              f"(written before step 8 recorded it) — re-deriving, which may not "
              f"match what its blocks were built with. Re-run step 8 to be sure.")
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    if pipeline_config and os.path.exists(pipeline_config):
        with open(pipeline_config) as f:
            _pcfg = json.load(f)
        for k, v in _pcfg.items():
            if cfg.get(k) is None:
                cfg[k] = v
    return resolve_d_err_r(cfg, cfg.get("fits_file"))


def load_population_main(run_dir, pipeline_config=None):
    """Load one population's MAIN-row catalog subset + full covariance.

    Returns (table, cov, analysis_mask, d_err_r). `table` rows, `cov`
    rows/cols, and `analysis_mask` entries are all in the same order:
    MAIN=True rows of color_xonly_catalog.fits, in catalog order (the
    invariant color_predict.py relies on for its own FITS-cov output,
    verified to hold identically for the HDF5 path).
    """
    cat_path = os.path.join(run_dir, "color_xonly_catalog.fits")
    cov_path = os.path.join(run_dir, "color_xonly_cov.h5")

    table = Table.read(cat_path)
    # astropy's FITS reader yields bytes (|Sn) for string columns. Decode to
    # native unicode so the *written* combined catalog carries str, not bytes.
    # _systematic_offdiag_terms also decodes defensively now, but this in-place
    # fix is what keeps the output column's dtype consistent with the inputs.
    if table["PHOTSYS"].dtype.kind == "S":
        table["PHOTSYS"] = np.char.decode(
            np.asarray(table["PHOTSYS"]), "ascii"
        )
    main_mask = np.asarray(table["MAIN"], dtype=bool)
    table_main = table[main_mask]

    with h5py.File(cov_path, "r") as hf:
        cov = hf["cov"][...]
        analysis = hf["analysis"][...]

    if cov.shape[0] != len(table_main):
        raise ValueError(
            f"{run_dir}: cov shape {cov.shape} does not match "
            f"MAIN row count {len(table_main)} in {cat_path}"
        )
    if len(analysis) != len(table_main):
        raise ValueError(
            f"{run_dir}: analysis length {len(analysis)} does not match "
            f"MAIN row count {len(table_main)}"
        )

    d_err_r = _resolve_d_err_r(run_dir, cov_path, pipeline_config=pipeline_config)
    return table_main, cov, analysis, d_err_r


def combine(spiral_run, irregular_run, out_run, catalog_name="color_xonly_catalog.fits",
            spiral_config=None, irregular_config=None):
    out_dir = os.path.join("output", out_run)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading spiral population from output/{spiral_run} ...")
    t_sp, cov_sp, ana_sp, d_err_r_sp = load_population_main(
        os.path.join("output", spiral_run), pipeline_config=spiral_config
    )
    print(f"  MAIN rows: {len(t_sp)}, d_err_r={d_err_r_sp:.8f}")

    print(f"Loading irregular population from output/{irregular_run} ...")
    t_ir, cov_ir, ana_ir, d_err_r_ir = load_population_main(
        os.path.join("output", irregular_run), pipeline_config=irregular_config
    )
    print(f"  MAIN rows: {len(t_ir)}, d_err_r={d_err_r_ir:.8f}")

    if not np.isclose(d_err_r_sp, d_err_r_ir, rtol=1e-10, atol=0):
        raise ValueError(
            "d_err_r mismatch between populations — the cross-population "
            "dust systematic term is only physically valid if both runs "
            f"share the same dust-slope uncertainty. Got spiral={d_err_r_sp!r} "
            f"vs irregular={d_err_r_ir!r}. Check 'dust_pickle' in "
            f"output/{spiral_run}/config.json and output/{irregular_run}/config.json."
        )
    d_err_r = d_err_r_sp

    n_sp = len(t_sp)
    n_ir = len(t_ir)
    n_tot = n_sp + n_ir

    # --- Combined catalog ---
    t_sp = t_sp.copy()
    t_ir = t_ir.copy()
    t_sp["POPULATION"] = np.full(n_sp, "spiral", dtype="U9")
    t_ir["POPULATION"] = np.full(n_ir, "irregular", dtype="U9")
    combined_table = vstack([t_sp, t_ir], metadata_conflicts="silent")

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
    cov_combined[:n_sp, :n_sp] = cov_sp
    cov_combined[n_sp:, n_sp:] = cov_ir

    v_dust_sp, v_dust_ir = v_dust_all[:n_sp], v_dust_all[n_sp:]
    v_phot_sp, v_phot_ir = v_phot_all[:n_sp], v_phot_all[n_sp:]

    cross = (
        np.outer(v_dust_sp, v_dust_ir) + np.outer(v_phot_sp, v_phot_ir)
    ).astype(np.float32)
    cov_combined[:n_sp, n_sp:] = cross
    cov_combined[n_sp:, :n_sp] = cross.T

    analysis_combined = np.concatenate([ana_sp, ana_ir])
    population_combined = np.concatenate(
        [np.zeros(n_sp, dtype=np.int8), np.ones(n_ir, dtype=np.int8)]
    )

    # --- Sanity checks ---
    diag_expected = np.concatenate([np.diag(cov_sp), np.diag(cov_ir)])
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
        np.asarray(t_sp["SGA_ID"], dtype=float),
        np.asarray(t_ir["SGA_ID"], dtype=float),
    ]))
    print(f"PASS: shapes/row-order consistent (G={n_tot})")

    # Spot-check: first North-flagged row in each population, if any
    n_idx_sp = np.flatnonzero(photsys_all[:n_sp] == "N")
    n_idx_ir = np.flatnonzero(photsys_all[n_sp:] == "N")
    if len(n_idx_sp) and len(n_idx_ir):
        i = int(n_idx_sp[0])
        j = int(n_idx_ir[0]) + n_sp
        expected = v_dust_all[i] * v_dust_all[j] + v_phot_all[i] * v_phot_all[j]
        actual = cov_combined[i, j]
        assert np.isclose(actual, expected, rtol=1e-5), (
            f"Cross-term spot-check failed at (i={i}, j={j}): "
            f"expected {expected}, got {actual}"
        )
        print(
            f"PASS: cross-term spot-check at (spiral row {i}, irregular row "
            f"{j - n_sp}) matches independent computation ({actual:.6e})"
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
        # Record the dust value the whole matrix was built with. The
        # per-population inputs carry this attribute and it is what the
        # cross-population blocks above were computed from; omitting it here left
        # the delivered product as the only file in the chain without the
        # provenance the mechanism exists to preserve.
        hf.attrs["d_err_r"] = float(d_err_r)
        hf.attrs["population_labels"] = "0=spiral, 1=irregular"
        hf.attrs["row_order"] = (
            f"MAIN=True rows of {catalog_name} in this directory, "
            "spiral rows first (in spiral catalog order), then irregular "
            "rows (in irregular catalog order)"
        )
    print(f"Written combined covariance ({n_tot}, {n_tot}) to {cov_out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiral-run", default="DR2_TF_spirals_v5_2color_spiral")
    parser.add_argument("--irregular-run", default="DR2_TF_irrs_v5_2color_irregular")
    parser.add_argument("--out-run", default="DR2_TF_v5_2color_combined")
    parser.add_argument("--catalog-name", default="DESI-DR2_TF_pv_cat_v5b.fits")
    parser.add_argument(
        "--spiral-config", default=None,
        help="Pipeline config for the spiral run. Only consulted if that run's "
             "covariance lacks a d_err_r attribute; overlaid onto the run-dir "
             "config.json so keys added after step 4 last ran (notably "
             "dust_pickle) are still seen. Same overlay color_predict.py does.",
    )
    parser.add_argument(
        "--irregular-config", default=None,
        help="Pipeline config for the irregular run; see --spiral-config.",
    )
    args = parser.parse_args()
    combine(args.spiral_run, args.irregular_run, args.out_run, args.catalog_name,
            spiral_config=args.spiral_config,
            irregular_config=args.irregular_config)


if __name__ == "__main__":
    main()
