"""Augment SGA-2020 v5 FITS with Z_ABSMAG_SB26 columns.

Computes the four z-band absolute magnitude columns that v5 lacks (and v6 has):
  Z_ABSMAG_SB26          = R_ABSMAG_SB26 - (R_MAG_SB26 - Z_MAG_SB26)
  Z_ABSMAG_SB26_ERR      = Z_MAG_SB26_ERR
  Z_ABSMAG_SB26_CORR     = R_ABSMAG_SB26_CORR - (R_MAG_SB26_CORR - Z_MAG_SB26_CORR)
  Z_ABSMAG_SB26_ERR_CORR = Z_MAG_SB26_ERR_CORR

Formula and error treatment match desi_data.py:127-128 (sigma_z is the
photometric noise on hat_z given latent z; combining with sigma_y would
double-count it in the B_i covariance — see desi_data.py:99-102).

Modifies the FITS file in place after writing a .bak backup.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

FITS_PATH = Path("data/SGA-2020_iron_Vrot_VI_corr_v5.fits")
BAK_PATH = FITS_PATH.with_suffix(FITS_PATH.suffix + ".bak")

NEW_COLS = [
    ("Z_ABSMAG_SB26",       "Z_ABSMAG_SB26_ERR",      "R_ABSMAG_SB26",       "R_MAG_SB26",       "Z_MAG_SB26",       "Z_MAG_SB26_ERR"),
    ("Z_ABSMAG_SB26_CORR",  "Z_ABSMAG_SB26_ERR_CORR", "R_ABSMAG_SB26_CORR",  "R_MAG_SB26_CORR",  "Z_MAG_SB26_CORR",  "Z_MAG_SB26_ERR_CORR"),
]


def main() -> int:
    if not FITS_PATH.exists():
        print(f"ERROR: {FITS_PATH} not found", file=sys.stderr)
        return 1

    with fits.open(FITS_PATH) as hdul:
        table_hdu = hdul[1]
        existing = set(table_hdu.columns.names)  # type: ignore[union-attr]
        already_present = [c for tup in NEW_COLS for c in tup[:2] if c in existing]
        if already_present:
            print(f"ERROR: column(s) already present, refusing to overwrite: {already_present}", file=sys.stderr)
            return 2

    if not BAK_PATH.exists():
        print(f"Backing up to {BAK_PATH}")
        shutil.copy2(FITS_PATH, BAK_PATH)
    else:
        print(f"Backup already exists at {BAK_PATH}, leaving it alone")

    with fits.open(FITS_PATH) as hdul:
        primary_hdu = hdul[0].copy()
        table_hdu = hdul[1]
        data = table_hdu.data  # type: ignore[union-attr]
        header = table_hdu.header.copy()  # type: ignore[union-attr]

        new_columns = list(table_hdu.columns)  # type: ignore[union-attr]
        added_names: list[str] = []
        for absname, errname, r_abs, r_app, z_app, z_err in NEW_COLS:
            R_ABS = np.asarray(data[r_abs], dtype=np.float32)
            R_APP = np.asarray(data[r_app], dtype=np.float32)
            Z_APP = np.asarray(data[z_app], dtype=np.float32)
            Z_ERR = np.asarray(data[z_err], dtype=np.float32)

            Z_ABS = R_ABS - (R_APP - Z_APP)

            new_columns.append(fits.Column(name=absname, format="E", array=Z_ABS))
            new_columns.append(fits.Column(name=errname, format="E", array=Z_ERR))
            added_names.extend([absname, errname])

            n_finite = int(np.isfinite(Z_ABS).sum())
            if n_finite > 0:
                med_zabs = float(np.nanmedian(Z_ABS))
                med_rabs = float(np.nanmedian(R_ABS))
                print(f"  {absname}: n_finite={n_finite}, median={med_zabs:.3f} (R median={med_rabs:.3f}, diff={med_zabs - med_rabs:+.3f})")

        new_table_hdu = fits.BinTableHDU.from_columns(new_columns, header=header)
        new_hdul = fits.HDUList([primary_hdu, new_table_hdu] + [hdu.copy() for hdu in hdul[2:]])

    new_hdul.writeto(FITS_PATH, overwrite=True)
    print(f"Wrote {FITS_PATH} with new columns: {added_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
