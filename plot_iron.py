import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt
import scipy.stats
import csv

fn = "data/SGA-2020_iron_Vrot_cuts_sub_0.02.json"
# fn="data/SGA-2020_fuji_Vrot.json"
# fn = "data/SGA_TFR_simtest_001.json"
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

plt.errorbar(data["V_0p4R26"], numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu"]),yerr=data["R_MAG_SB26_ERR"], xerr=data["V_0p4R26_err"], fmt=".")
plt.xlabel("V_0p4R26")
plt.ylabel("R_MAG_SB26-mu")
plt.show()


plt.errorbar(data["Z_DESI"], data["V_0p4R26"],yerr=data["V_0p4R26_err"], fmt=".")
plt.xlabel("Z_DESI")
plt.ylabel("V_0p4R26")
plt.show()



plt.hist(numpy.log10(data["V_0p33R26"]))
plt.xlabel(r"$\log{V}$")
plt.ylabel(r"$N$")
plt.savefig("temp.pdf")

plt.hist(numpy.log10(data["V_0p4R26"]))
plt.xlabel(r"$\log{V}$")
plt.ylabel(r"$N$")
plt.savefig("temp.pdf")


plt.plot(x, scipy.stats.lognorm.pdf(x, *ans))
plt.show()

MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu"])

plt.hist(MR)
plt.show()
ans = scipy.stats.lognorm.fit(MR) # (0.3203771381830672, -24.483252891972377, 3.8906505354308463)
x=numpy.linspace(-23,-14,100)
plt.plot(x, scipy.stats.lognorm.pdf(x, *ans))
plt.show()

ans = scipy.stats.norm.fit(numpy.log10(data["V_0p4R26"])/numpy.cos(numpy.arctan(-6.1)))
ans # (13.133570672711606, 1.5160651053079683)


ans = scipy.stats.skewnorm.fit(numpy.log10(data["V_0p4R26"])/numpy.cos(numpy.arctan(-6.1)))
ans # (-3.661245022462153, 14.913405242237685, 2.2831016215521247)
# Out[42]: (-2.4813505391290436, 14.628796578863792, 1.4880837674710605) for pruned set
--
#Fuji
x=[20,600]
# plt.plot(x,-6.9 -6.1* numpy.log10(x))
plt.plot(x,-3.88 -7.55* numpy.log10(x))
MR = numpy.array(data["R_MAG_SB26"]) - 34.7
plt.errorbar(data["V_0p33R26"], MR ,yerr=numpy.sqrt(numpy.array(data["R_MAG_SB26_ERR"])**2),xerr=data["V_0p33R26_err"], fmt=".")
plt.xscale('log',base=10)
plt.xlabel("V_0p33R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()



#iron

fn = "data/SGA-2020_iron_Vrot_cuts.json"
with open(fn, 'r') as f:
    data = json.load(f)

z = numpy.array(data["Z_DESI"])
dv = 300
dm = (5/numpy.log(10)*(1+z)**2*dv/cosmo.H(z)/cosmo.luminosity_distance(z)).value

x=[50,600]
plt.plot(x,-6.8 -6.1* numpy.log10(x))
MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu"])
plt.errorbar(data["V_0p4R26"], MR ,yerr=numpy.sqrt(numpy.array(data["R_MAG_SB26_ERR"])**2),xerr=data["V_0p4R26_err"], fmt=".")
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()

MR = numpy.array(data["Rhat"]) - numpy.array(data["mu"])
plt.errorbar(data["Vhat"], MR ,yerr=data["Rhat_noise"],xerr=data["Vhat_noise"], fmt=".")
plt.xscale('log',base=10)
plt.xlabel("V_0p33R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()

ans = scipy.stats.skewnorm.fit(numpy.log10(data["V_0p4R26"])/numpy.cos(numpy.arctan(-6.1)))
print(ans) # (-1.3565289337241162, 14.193371687903761, 1.0984767423119663)
plt.hist(numpy.log10(data["V_0p4R26"])/numpy.cos(numpy.arctan(-6.1)),density=True)
x=numpy.linspace(6,18,100)
plt.plot(x, scipy.stats.skewnorm.pdf(x, -1.3565289337241162, 14.193371687903761,1.0984767423119663))
# plt.plot(x, scipy.stats.skewnorm.pdf(x, -3.661245022462153, 14.913405242237685,2.2831016215521247))
plt.show()

fn = "data/iron_cluster.json"
with open(fn, 'r') as f:
    data = json.load(f)

with open("output/cluster_410_opt.csv", newline='') as csvfile:
    optimal = pandas.read_csv(csvfile,comment='#')

MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
brArr=numpy.array([optimal["bR.{}".format(i+1)][0] for i in range(0,data["N_cluster"])])
brArrmn= brArr-brArr.mean()
index = 0
for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
    if True:
        plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    index = index+data["N_per_cluster"][i]
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()


index = 0
for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
    if True:
        plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]]-brArrmn[i] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    index = index+data["N_per_cluster"][i]
print(plt.xlim())
plt.plot(plt.xlim(), brArr.mean()+numpy.log10(plt.xlim())*optimal['aR'][0])
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()


ans = scipy.stats.skewnorm.fit(numpy.log10(data["V_0p4R26"])/numpy.cos(numpy.arctan(-6.1)))
print(ans) # (-1.3565289337241162, 14.193371687903761, 1.0984767423119663)

plt.hist(numpy.log10(data["V_0p4R26"]),density=True)
x=numpy.linspace(1.7,2.6,100)
# plt.plot(x, scipy.stats.skewnorm.pdf(x/numpy.cos(numpy.arctan(-6.334)), 1.585, 13.35, 0.8958)/numpy.cos(numpy.arctan(-6.334)))
plt.plot(x, scipy.stats.skewnorm.pdf(x/numpy.cos(optimal["atanAR"][0]), optimal["alpha_dist"][0], optimal["xi_dist"][0], optimal["omega_dist"][0])/numpy.cos(optimal["atanAR"][0]))
plt.show()