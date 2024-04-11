import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt

fn = "data/SGA-2020_iron_Vrot_sub.json"

with open(fn, 'r') as f:
    data = json.load(f)

z = numpy.array(data["Z_DESI"])
dv = 300
dm = (5/numpy.log(10)*(1+z)**2*dv/cosmo.H(z)/cosmo.luminosity_distance(z)).value

plt.errorbar(data["Z_DESI"], data["R_MAG_SB26"],yerr=data["R_MAG_SB26_ERR"], fmt=".")
plt.xlabel("Z_DESI")
plt.ylabel("R_MAG_SB26")
plt.show()

plt.errorbar(data["Z_DESI"], numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu"]),yerr=data["R_MAG_SB26_ERR"], fmt=".")
plt.xlabel("Z_DESI")
plt.ylabel("R_MAG_SB26-mu")
plt.show()

plt.errorbar(data["Z_DESI"], data["V_0p4R26"],yerr=data["V_0p4R26_err"], fmt=".")
plt.xlabel("Z_DESI")
plt.ylabel("V_0p4R26")
plt.show()

x=[10,800]
plt.errorbar(data["V_0p4R26"], numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu"]),yerr=numpy.sqrt(dm**2+numpy.array(data["R_MAG_SB26_ERR"])**2),xerr=data["V_0p4R26_err"], fmt=".")
plt.plot(x,-7.1 - 6.1* numpy.log10(x))
plt.xscale('log',base=10)
plt.show()