import os
from astropy.io import fits


def main():
    input_file = os.path.join("data", "SGA-2020_iron_Vrot_VI_corr_v2.fits")
    output_file = os.path.join("data", "SGA-2020_iron_Vrot_VI_corr_v3.fits")

    with fits.open(input_file) as hdul:
        data = hdul[1].data

        # 1. Rename the absolute magnitude column
        # Note: changing column names dynamically may be safer by creating a new ColDefs

        cols = []
        for col in hdul[1].columns:
            if col.name == "R_ABSMAG_SB26":
                # Create renamed column
                new_col = fits.Column(
                    name="R_ABSMAG_SB26_CORR", format=col.format, array=col.array
                )
                cols.append(new_col)
            else:
                cols.append(col)

        # 2. Add the absolute magnitude error column by copying the apparent magnitude error
        err_data = data["R_MAG_SB26_ERR_CORR"].copy()
        new_err_col = fits.Column(
            name="R_ABSMAG_SB26_ERR_CORR",
            format=hdul[1].columns["R_MAG_SB26_ERR_CORR"].format,
            array=err_data,
        )
        cols.append(new_err_col)

        # Create a new HDU with the old + new columns
        new_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        new_hdu.writeto(output_file, overwrite=True)
        print(f"Successfully created {output_file}")


if __name__ == "__main__":
    main()
