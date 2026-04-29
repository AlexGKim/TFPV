from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

data_file = 'data/SGA-2020_iron_Vrot_VI_corr_v5.fits'

hdul = fits.open(data_file)
data = hdul[1].data

z = data['Z_DESI']
kcorr = data['R_KCORR']

mask = np.isfinite(z) & np.isfinite(kcorr)
z = z[mask]
kcorr = kcorr[mask]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(z, kcorr, s=5, alpha=0.4, linewidths=0)
ax.axvline(0.065, color='k', linestyle=':', linewidth=1)
ax.set_xlim(0, 0.15)
ax.set_ylim(-0.15, 0.15)
ax.set_xlabel('Z_DESI')
ax.set_ylabel('K-correction (R band)')
ax.set_title('Z_DESI vs R-band K-correction')
plt.tight_layout()
plt.savefig('plot_zcorr_kcorr_r.png', dpi=150)
hdul.close()
