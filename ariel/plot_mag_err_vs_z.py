import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

fits_path = "data/SGA-2020_iron_Vrot_VI_corr.fits"

with fits.open(fits_path) as hdul:
    data = hdul[1].data
    z    = np.asarray(data["Z_DESI"],        dtype=float)
    err  = np.asarray(data["R_MAG_SB26_ERR"], dtype=float)

finite = np.isfinite(z) & np.isfinite(err)
z, err = z[finite], err[finite]

plt.scatter(z, err, marker=".", alpha=0.2, s=2)
plt.xlabel(r"$z_\mathrm{DESI}$")
plt.ylabel(r"R\_MAG\_SB26\_ERR (mag)")
plt.tight_layout()
plt.savefig("mag_err_vs_z.png", dpi=300)
print("Saved mag_err_vs_z.png")
