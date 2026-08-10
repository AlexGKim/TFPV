#!/usr/bin/env python3
"""Regression test: SGA_ID must be mapped into valid-row space using the SAME
validity mask ``load_xyz_and_uncertainties_from_desi`` applies.

The bug this pins down: ``color_predict`` derived that mapping twice, from two
independent and NON-equivalent filters. The authoritative loader drops a row when
``sigma_y >= 0`` fails (among other conditions); the second derivation
(``_sga_ids_valid_for_mask``) never checked sigma_y at all. Any catalog with a
negative R_ABSMAG_SB26_ERR therefore produced two different row counts and step 8
died with

    ValueError: _sga_ids_valid_for_mask: SGA_ID array length (177184) does not
    match main_mask length (177171)

Found on the real v2.0.8 mock TF_AbacusSummit_base_c000_ph000_r000, which has
exactly 13 rows with R_ABSMAG_SB26_ERR < 0. Every file processed before it
happened to have none, so the duplication sat dormant.

Run: python3 tests/test_sga_id_valid_mask.py
"""

import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_predict import (  # noqa: E402
    load_xyz_and_uncertainties_from_desi,
    _sga_ids_for_valid_mask,
)


def _make_mock_fits(path, n=50, n_bad_sigma_y=3):
    """A minimal fullmocks-shaped catalog with n_bad_sigma_y negative sigma_y rows."""
    rng = np.random.default_rng(0)
    logv = rng.normal(2.2, 0.1, n)
    r_abs = rng.normal(-20.0, 0.5, n)
    r_app = r_abs + 35.0
    r_err = np.full(n, 0.05)
    # The trigger: a few rows carry a NEGATIVE r-band error.
    r_err[:n_bad_sigma_y] = -0.05

    cols = [
        fits.Column(name="LOGVROT", format="D", array=logv),
        fits.Column(name="LOGVROT_ERR", format="D", array=np.full(n, 0.02)),
        fits.Column(name="R_ABSMAG_SB26", format="D", array=r_abs),
        fits.Column(name="R_ABSMAG_SB26_ERR", format="D", array=r_err),
        fits.Column(name="R_MAG_SB26", format="D", array=r_app),
        fits.Column(name="Z_ABSMAG_SB26", format="D", array=r_abs - 0.4),
        fits.Column(name="Z_ABSMAG_SB26_ERR", format="D", array=np.full(n, 0.04)),
        fits.Column(name="Z_MAG_SB26", format="D", array=r_app - 0.4),
        fits.Column(name="Z_MAG_SB26_ERR", format="D", array=np.full(n, 0.04)),
        fits.Column(name="G_ABSMAG_SB26", format="D", array=r_abs + 0.6),
        fits.Column(name="G_ABSMAG_SB26_ERR", format="D", array=np.full(n, 0.06)),
        fits.Column(name="G_MAG_SB26", format="D", array=r_app + 0.6),
        fits.Column(name="G_MAG_SB26_ERR", format="D", array=np.full(n, 0.06)),
        fits.Column(name="ZOBS", format="D", array=rng.uniform(0.02, 0.06, n)),
        fits.Column(name="BA_RATIO", format="D", array=rng.uniform(0.2, 0.9, n)),
        fits.Column(name="PHOTSYS", format="D", array=np.zeros(n)),
        fits.Column(name="PHOTSYS_ERR", format="D", array=np.zeros(n)),
        fits.Column(name="MAIN", format="L", array=np.ones(n, bool)),
        fits.Column(name="DWARF", format="L", array=np.zeros(n, bool)),
    ]
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return n, n_bad_sigma_y


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mock_negative_sigma_y.fits")
        n, n_bad = _make_mock_fits(path)

        out = load_xyz_and_uncertainties_from_desi(
            path, with_gband=True, return_mask=True
        )
        mask = out[-1]
        xhat = out[0]

        n_valid = int(np.asarray(mask).sum())
        print(f"rows={n}  negative sigma_y={n_bad}")
        print(f"loader mask keeps {n_valid} (expected {n - n_bad})")
        if n_valid != n - n_bad:
            failures.append(
                f"loader kept {n_valid}, expected {n - n_bad} — the sigma_y>=0 "
                f"condition is not being applied as assumed"
            )
        if len(xhat) != n_valid:
            failures.append(
                f"returned arrays are {len(xhat)} long but mask sums to {n_valid}"
            )

        # The property under test: SGA_IDs must be mapped into valid-row space
        # using the loader's OWN mask, never a second filter reconstructed from
        # scratch. _sga_ids_for_valid_mask takes that mask as an argument, so
        # there is only one definition of "valid" in the codebase.
        sga_valid = _sga_ids_for_valid_mask(path, mask)
        print(f"SGA_ID mapped through the loader mask: {len(sga_valid)}")
        if len(sga_valid) != len(xhat):
            failures.append(
                f"SGA_ID length {len(sga_valid)} != loader array length {len(xhat)}"
            )

        # And the regression itself: a filter that omits sigma_y>=0 disagrees.
        with fits.open(path) as h:
            d = h[1].data
            lv = np.asarray(d["LOGVROT"], float)
            lve = np.asarray(d["LOGVROT_ERR"], float)
            zab = np.asarray(d["Z_ABSMAG_SB26"], float)
        naive = (np.isfinite(lv) & np.isfinite(lve) & (lve > 0) & np.isfinite(zab))
        print(f"a filter omitting sigma_y>=0 would keep {int(naive.sum())} "
              f"-> mismatch of {int(naive.sum()) - n_valid}")
        if int(naive.sum()) == n_valid:
            failures.append(
                "the synthetic file does not actually exercise the bug: the naive "
                "filter agrees with the loader"
            )

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS: SGA_ID maps through the loader's own validity mask consistently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
