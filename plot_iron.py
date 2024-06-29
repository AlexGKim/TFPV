import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt
import scipy.stats
import pandas
from chainconsumer import Chain, ChainConsumer, PlotConfig
import matplotlib

matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["lines.linewidth"] = 2


def cluster():
    infile = json.load(open("data/iron_cluster.json",))

    chains=[]
    c = ChainConsumer()

    for _ in [3,4]:
        if _ == 3:
            name = 'Inverse TF'
        elif _==4:
            name = 'Perpendicular'
        dum=[pandas.read_csv("output/cluster_{}11_{}.csv".format(_,i),comment='#') for i in range(1,5)]
        bRcols=["bR.{}".format(cin) for cin in range(1,12)]
        for df_ in dum:
            df_["bR_use"] = df_[bRcols].mean(axis=1) - df_["xi_dist"]*df_["aR"]
            df_["omega_dist_use"] = df_["omega_dist"] * numpy.cos(df_["atanAR"])
        dum=pandas.concat(dum)
        chains.append(dum)
        # dum=pandas.read_csv("output/temp_{}.csv".format(1),comment='#')

        lrmn=[]
        for cin in range(1,12):
            use = dum["bR.{}".format(cin)] - dum["xi_dist"]*dum["aR"]
            lrmn.append(numpy.percentile(use, (.32,.5,.68)))

        lrmn = numpy.array(lrmn).transpose()
        lrmn[0]=lrmn[1]-lrmn[0]
        lrmn[2]=lrmn[2]-lrmn[1]
        yerr=numpy.array((lrmn[0],lrmn[2]))
        plt.errorbar(infile["mu"],lrmn[1],fmt="+",yerr=yerr)
        plt.xlabel(r"$\mu$")
        plt.ylabel(r"$b$")
        plt.savefig("b_cluster.png")
        plt.clf()

        # c = ChainConsumer()
        c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use"]], name=name))

        # fig = c.plotter.plot()
        # plt.savefig("corner_cluster_{}.png".format(_))
        # # plt.show()
        # plt.clf()

        # print(c.analysis.get_latex_table())
    c.set_plot_config(
        PlotConfig(
            summary_font_size=20, label_font_size=20, legend_kwargs={'prop':{'size':20}}, labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$"},
        )
    )
    fig = c.plotter.plot()
    plt.savefig("corner_cluster.png")
    # plt.show()
    plt.clf()
    egr
    print(c.analysis.get_latex_table())

    fn="data/iron_cluster.json"
    fn_all="data/iron_cluster_all.json"


    with open(fn, 'r') as f:
        data = json.load(f)

    with open(fn_all, 'r') as f:
        data_all = json.load(f)

    MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
    MR_all  = numpy.array(data_all["R_MAG_SB26"]) - numpy.array(data_all["mu_all"])

    plt.errorbar(data_all["V_0p4R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p4R26_err"], fmt="+", label="cut",color='black')
    # plt.errorbar(data["V_0p4R26"], MR ,yerr=data["R_MAG_SB26_ERR"],xerr=data["V_0p4R26_err"], fmt="+", label="sample",color='black')

    index = 0
    for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
        if True:
            plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
        index = index+data["N_per_cluster"][i]

    mn = chains[1][["aR","bR_use"]].mean()
    cov = chains[1][["aR","bR_use"]].cov()
    
    dum = numpy.array(plt.xlim())
    if dum[0] <=0:
        dum[0]=10
    for i in range(1000):
        aR, bR = numpy.random.multivariate_normal(mn, cov)
        plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')    

    plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    plt.plot(dum, chains[0]["bR_use"].mean() + chains[0]["aR"].mean()*numpy.log10(dum),label="Inverse TF")
            
    plt.xscale('log',base=10)
    plt.xlabel("V_0p4R26")
    plt.ylabel(r"R_MAG_SB26-$\mu$")
    plt.ylim((MR.max()+.5,MR.min()-.5))
    plt.legend()
    plt.savefig("tf_cluster_all.png") 
    plt.clf()

    plt.hist(numpy.log10(data["V_0p4R26"]),density=True)
    x=numpy.linspace(1.8,2.5,100)
    plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[1]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[0]["xi_dist"].mean(),chains[0]["omega_dist_use"].mean()) ,label="Inverse TF")
    plt.xlabel(r"$\log{(V\_0p4R26)}$")
    plt.legend()
    plt.savefig("hist_cluster.png")
    plt.clf()

cluster()

wef

def fuji():
    chains=[]
    for _ in [3,4]:
        dum=[pandas.read_csv("output/fuji_{}11_{}.csv".format(_,i),comment='#') for i in range(1,5)]
        for df_ in dum:
            df_["bR_use"] = df_["bR"] - df_["xi_dist"]*df_["aR"]
            df_["omega_dist_use"] = df_["omega_dist"] * numpy.cos(df_["atanAR"])
        dum=pandas.concat(dum)
        chains.append(dum)
        # dum=pandas.read_csv("output/temp_{}.csv".format(1),comment='#')

        c = ChainConsumer()
        c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use"]], name="An Example Contour"))
        c.set_plot_config(
            PlotConfig(
                labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$"},
            )
        )
        fig = c.plotter.plot()
        plt.savefig("corner_fuji_{}.png".format(_))
        # plt.show()
        plt.clf()
    fn="data/SGA-2020_fuji_Vrot_cuts.json"
    fn_all="data/SGA-2020_fuji_Vrot.json"


    with open(fn, 'r') as f:
        data = json.load(f)

    with open(fn_all, 'r') as f:
        data_all = json.load(f)

    MR = numpy.array(data["R_MAG_SB26"]) - 34.7
    MR_all  = numpy.array(data_all["R_MAG_SB26"]) - 34.7
    plt.errorbar(data_all["V_0p33R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p33R26_err"], fmt="+", label="cut", color="black")

    mn = chains[1][["aR","bR_use"]].mean()
    cov = chains[1][["aR","bR_use"]].cov()
    
    dum = numpy.array(plt.xlim())
    if dum[0] <=0:
        dum[0]=10

    mn = chains[1][["aR","bR_use"]].mean()
    cov = chains[1][["aR","bR_use"]].cov()
    
    for i in range(2000):
        aR, bR = numpy.random.multivariate_normal(mn, cov)
        plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')  

    #     plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')    
    # plt.errorbar(data_all["V_0p33R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p33R26_err"], fmt=".", label="cut")
    plt.errorbar(data["V_0p33R26"], MR ,yerr=data["R_MAG_SB26_ERR"],xerr=data["V_0p33R26_err"], fmt=".",label="sample") 
    plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    plt.plot(dum, chains[0]["bR_use"].mean() + chains[0]["aR"].mean()*numpy.log10(dum),label="Inverse TF")
    plt.xscale('log',base=10)
    plt.xlabel("V_0p33R26")
    plt.ylabel(r"R_MAG_SB26-$\mu$")
    plt.ylim((MR_all.max()+.5,MR_all.min()-.5))
    # plt.xlim(dum)
    plt.legend()
    plt.savefig("tf_fuji.png") 
    plt.clf()
    # plt.show()

    plt.hist(numpy.log10(data["V_0p33R26"]),density=True)
    x=numpy.linspace(1.5,2.5,100)
    plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[1]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[0]["xi_dist"].mean(),chains[0]["omega_dist_use"].mean()) ,label="Inverse TF")
    plt.xlabel(r"$\log{(V\_0p33R26)}$")
    plt.legend()
    plt.savefig("hist_fuji.png")
    plt.clf()

fuji()
wef

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


MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])

index = 0
for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
    if True:
        plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    index = index+data["N_per_cluster"][i]
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
# plt.ylim((MR.max()+.5,MR.min()-.5))
plt.show()


index = 0
for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
    if True:
        plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]]-brArr4mn[i] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    index = index+data["N_per_cluster"][i]
plt.plot(plt.xlim(), brArr4.mean()+numpy.log10(plt.xlim())*optimal4['aR'][0])
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.title("Perpendicular Best Fit")
plt.show()


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
        plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]]-brArr3mn[i] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    index = index+data["N_per_cluster"][i]
plt.xscale('log',base=10)
plt.xlabel("V_0p4R26")
plt.ylabel(r"R_MAG_SB26-$\mu$")
plt.ylim((MR.max()+.5,MR.min()-.5))
plt.title("Inverse TF Best Fit")
plt.show()


ans = scipy.stats.skewnorm.fit(numpy.log10(data["V_0p4R26"])/numpy.cos(numpy.arctan(-6.1)))
print(ans) # (-1.3565289337241162, 14.193371687903761, 1.0984767423119663)

plt.hist(numpy.log10(data["V_0p4R26"]),density=True)
x=numpy.linspace(1.7,2.6,100)
plt.legend()
plt.xlabel(r"$\log{(V\_0p4R26)}$")
plt.show()




