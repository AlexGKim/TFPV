import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt
import scipy.stats

fn = "data/SGA-2020_iron_Vrot_sub_0.10.json"

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


x=numpy.linspace(0.1,1e16,100)
ans = scipy.stats.lognorm.fit(numpy.array(data["V_0p4R26"]),floc=0) # (0.5521701970150247, 0, 135.42611665436885)


ans = scipy.stats.lognorm.fit(numpy.array(data["V_0p4R26"])**(1/numpy.cos(numpy.arctan(-6.1))),floc=0) # (3.413197988760724, 0, 15029899349402.74)
plt.plot(x, scipy.stats.lognorm.pdf(x, *ans))
plt.show()

plt.hist(numpy.array(data["V_0p4R26"]))
plt.show()
plt.hist(numpy.array(data["V_0p4R26"])**(1/numpy.cos(numpy.arctan(-6.1))),range=(0,1e10),bins=20)
plt.show()

plt.hist(1/numpy.cos(numpy.arctan(-6.1)) * numpy.log10(numpy.array(data["V_0p4R26"])))
plt.show()

ans = scipy.stats.norm.fit(1/numpy.cos(numpy.arctan(-6.1)) * numpy.log10(numpy.array(data["V_0p4R26"]))) # (13.176956072260538, 1.48233305216206)


x=[10,800]
MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu"])
plt.errorbar(data["V_0p4R26"], MR ,yerr=numpy.sqrt(dm**2+numpy.array(data["R_MAG_SB26_ERR"])**2),xerr=data["V_0p4R26_err"], fmt=".")
plt.plot(x,-7.979 -5.784* numpy.log10(x))
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()

