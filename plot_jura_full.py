'''
This script is intended to be run with an entire sample that has been broken up into subsamples/chunks (for if we run on full Y1 or Y3 sample). To analyze just a single subsample or single data file, use "plot_jura.py".


This generates the collective sample diagnostics of the combined sample as well as individual corner plots for each subsample.

cluster_perp generates just the metrics for the perpendicular dispersion case
cluster_all reads in the output chains for all 3 dispersion cases

'''

import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt
import scipy.stats
import pandas
from chainconsumer import Chain, ChainConsumer, PlotConfig
import matplotlib
import glob
from scipy.stats import norm
import pickle

matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["lines.linewidth"] = 2

def cluster_perp(file, subdir, nocluster=False, V0=True):
    desi_sga_dir = "/global/homes/s/sgmoore1/DESI_SGA/"
    
    # ncluster  = len(glob.glob(desi_sga_dir+"/TF/Y3/output_*.txt"))
    # print(ncluster)
    # wef
    # infile = json.load(open("data/{}.json".format(file)))
    # logV0 = infile["logV0"]
    
    # if nocluster:
    #     ncluster=1
    # else:
    #     ncluster = infile["N_cluster"]

    # Load all JSONs and treat them as one cluster
    mu_all = []
    logV0_list = []
    
    # Add these lists to collect other arrays
    V_0p4R26_all = []
    R_MAG_SB26_all = []
    R_MAG_SB26_ERR_all = []
    V_0p4R26_err_all = []
    mu_per_subsample = []  # optional, if needed
    
    for i in range(62):
        filename = f"data/Y1_full/iron_subsample_{i:02d}.json"
        with open(filename, "r") as f:
            js = json.load(f)
    
        mu_all.extend(js["mu"])
        logV0_list.append(js["logV0"])
        
        # Collect additional fields
        V_0p4R26_all.extend(js["V_0p4R26"])
        R_MAG_SB26_all.extend(js["R_MAG_SB26"])
        R_MAG_SB26_ERR_all.extend(js["R_MAG_SB26_ERR"])
        V_0p4R26_err_all.extend(js["V_0p4R26_err"])
        # mu_per_subsample.append(js["mu"])  # if you want to keep track by file


    
    # Bundle into unified 'infile'
    infile = {
        "mu": mu_all,
        "logV0": numpy.mean(logV0_list),
    }

    data = {
        "V_0p4R26": V_0p4R26_all,
        "R_MAG_SB26": R_MAG_SB26_all,
        "R_MAG_SB26_ERR": R_MAG_SB26_ERR_all,
        "V_0p4R26_err": V_0p4R26_err_all,
        "mu_all": mu_all
    }
    ncluster=1
    logV0= infile['logV0']
    data['N_per_cluster'] = [len(data["V_0p4R26"])]

    # infile = 
    
    print(ncluster)

    chains=[]
    c = ChainConsumer()
    
    c2 = ChainConsumer()
    fig_b, ax_b = plt.subplots()
    fig_b2, ax_b2 = plt.subplots()

    name='Perpendicular'

    dum = []
    for subsample_idx in range(20):
        for chain_idx in range(1, 5):
            chain_path = f"{subdir}/cluster_411_{subsample_idx:02d}_{chain_idx}.csv"
            try:
                df = pandas.read_csv(chain_path, comment='#')
                dum.append(df)
            except Exception as e:
                print(f"Failed to read {chain_path}: {e}")


    for df_ in dum:
        df_["bR_use"] = df_["bR.1"].mean() - df_["xi_dist"]*df_["aR"]
        df_["omega_dist_use"] = df_["omega_dist"] * numpy.cos(df_["atanAR"])
        df_['sigR_proj'] = 1/numpy.cos(df_["atanAR"])*df_["sigR"]
        df_['theta_2'] = df_["atanAR"]+numpy.pi/2
            
    dum = pandas.concat(dum, ignore_index=True)
    # print("Available columns in dum:", dum.columns.tolist())
    chains.append(dum)
    # Diagnostics for aR
    # print("\n--- Diagnostics for aR ---")
    # print("aR dtype:", dum["aR"].dtype)
    # print("aR nulls:", dum["aR"].isnull().sum())
    # print("aR unique values:", dum["aR"].nunique())
    # print(dum["aR"].describe())
    
    # # Save histogram instead of showing it
    # plt.hist(dum["aR"], bins=30)
    # plt.title("Histogram of aR")
    # plt.tight_layout()
    # plt.savefig("output/{}/aR_hist_debug.png".format(subdir))
    # plt.clf()
    
    # dum=pandas.read_csv("output/temp_{}.csv".format(1),comment='#')
    lrmn=[]
    lrmn2=[]
    for cin in range(1,ncluster+1):
        use = dum["bR.{}".format(cin)] - dum["xi_dist"]*dum["aR"]
        lrmn.append(numpy.percentile(use, (32,50,68)))
        use2 = dum["bR.{}".format(cin)] - dum["bR.{}".format(1)] 
        lrmn2.append(numpy.percentile(use2, (32,50,68)))
        
    # for cin in range(1,ncluster+1):
    #     use = dum["bR"] - dum["xi_dist"]*dum["aR"]
    #     lrmn.append(numpy.percentile(use, (32,50,68)))
    #     use2 = dum["bR"] - dum["bR"] 
    #     lrmn2.append(numpy.percentile(use2, (32,50,68)))
    
    lrmn = numpy.array(lrmn).transpose()
    lrmn[0]=lrmn[1]-lrmn[0]
    lrmn[2]=lrmn[2]-lrmn[1]
    yerr=numpy.array((lrmn[0],lrmn[2]))
    lrmn2 = numpy.array(lrmn2).transpose()
    lrmn2[0]=lrmn2[1]-lrmn2[0]
    lrmn2[2]=lrmn2[2]-lrmn2[1]
    yerr2=numpy.array((lrmn2[0],lrmn2[2]))        

    off = 0.025
    if nocluster:
        ax_b2.errorbar(numpy.mean(infile["mu"]) + off,[lrmn2[1][0]],fmt="+",yerr=yerr2,label=name)
    else:
        ax_b2.errorbar(numpy.array(infile["mu"]) + off, lrmn2[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)
    # ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)

    # c = ChainConsumer()
    c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2"]], name=name))
    c2.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))
    cred = numpy.percentile(dum["aR"], [16, 50, 84])
    aR_summary = [cred[1], cred[2] - cred[1], cred[1] - cred[0]]
    
    # Ensure the key exists
    if name not in c.analysis._summaries:
        c.analysis._summaries[name] = {}
    
    # Inject summary
    # c.analysis._summaries[name]["aR"] = aR_summary
    
    # c.add_chain(Chain(
    # samples=dum[["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2"]],
    # name=name,
    # parameters=["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2"],
    # plot=True  # ensures ChainConsumer doesn’t suppress it
    # ))
    # c2.add_chain(Chain(
    # samples=dum[["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2","sigR_proj"]],
    # name=name,
    # parameters=["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2","sigR_proj"],
    # plot=True  # ensures ChainConsumer doesn’t suppress it
    # ))
    # summary = c.analysis.get_summary()
    # print("Summary for 'aR':", summary.get("aR"))


    _v = numpy.percentile(numpy.sin(dum["atanAR"])*dum["sigR"],(32,50,100-32))
    print("${:5.3f}_{:5.3f}^+{:5.3f}$".format(_v[1], -_v[1]+_v[0],_v[2]-_v[1]))      
    fig = c.plotter.plot()
    plt.savefig("{}/output/corner_cluster_4.png".format(subdir))
    # plt.show()
    plt.clf()
    print(c.analysis.get_latex_table())

    
    # ax_b.set_xlabel(r"$\mu$") ### this only matters for the inverse case
    # ax_b.set_ylabel(r"$b$")
    # ax_b.legend()
    # fig_b.tight_layout()
    # fig_b.savefig("output/{}/b_cluster.png".format(subdir))

    # ax_b2.set_xlabel(r"$\mu$")
    # ax_b2.set_ylabel(r"$b-b_0$")
    # ax_b2.legend(loc=3)
    # fig_b2.tight_layout()
    # fig_b2.savefig("output/{}/b_cluster2.png".format(subdir))   

    # plt.clf()

    c.set_plot_config(
        PlotConfig(
            summary_font_size=20, label_font_size=20, constrain=False, legend_kwargs={'prop':{'size':20}}, labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$", "theta_2": r"$\theta_2$"}, 
        )
    )

    fig = c.plotter.plot()
    allaxes = fig.get_axes()

    allaxes[35].set_ylim((0,65))
    plt.savefig("{}/output/corner_cluster.png".format(subdir))
    # plt.show()
    plt.clf()
    print(c.analysis.get_latex_table())
    print(c2.analysis.get_latex_table())



    # fn="data/"+file+".json"
    # fn_all="output/jura_cluster_all.json"


    # with open(fn, 'r') as f:
    #     data = json.load(f)
    #     print(len(data))

    # with open(fn_all, 'r') as f:
    #     data_all = json.load(f)

    MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
    print(f'---------------- The length of the dataset being plotted is: {len(MR)}')
    # MR_all  = numpy.array(data_all["R_MAG_SB26"]) - numpy.array(data_all["mu_all"])

    # plt.errorbar(data_all["V_0p4R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p4R26_err"], fmt="+", label="cut",color='black',alpha=0.5)
    plt.errorbar(data["V_0p4R26"], MR ,yerr=data["R_MAG_SB26_ERR"],xerr=data["V_0p4R26_err"], fmt="+", label="jura galaxies",color='blue', alpha=0.05)

    index = 0
    # for i in range(0,ncluster): #range(data["N_cluster"]):
    #     if True:
    #         plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    #     index = index+data["N_per_cluster"][i]
    
    mn = chains[0][["aR","bR_use"]].mean()
    cov = chains[0][["aR","bR_use"]].cov()

    dum = numpy.array(plt.xlim())
    if dum[0] <=0:
        dum[0]=10
    for i in range(1000):
        aR, bR = numpy.random.multivariate_normal(mn, cov)
        if V0:
            bR -= aR*logV0
            mean_b = chains[0]["bR_use"].mean() - chains[0]["aR"].mean()*logV0
        else: 
            mean_b = chains[0]["bR_use"].mean()
        plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.02,color='black')    
    # plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    plt.plot(dum, mean_b + chains[0]["aR"].mean()*numpy.log10(dum),label="Bayesian", color='pink')
    plt.axhline(-18, linestyle='dotted') 
    plt.axvline(70, linestyle='dotted')    
    plt.xscale('log',base=10)
    plt.xlabel(r"$\hat{V}$")
    plt.ylabel(r"$\hat{m}$-$\mu$")
    plt.ylim((MR.max()+.5,MR.min()-.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/output/tf_cluster_all.png".format(subdir)) 
    plt.clf()

    # index = 0
    # for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
    #     if True:
    #         plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    #     index = index+data["N_per_cluster"][i]
    # if len(chains) > 2:
    #     mn = chains[2][["aR","bR_use"]].mean()
    #     cov = chains[2][["aR","bR_use"]].cov()
        
    #     dum = numpy.array(plt.xlim())
    #     if dum[0] <=0:
    #         dum[0]=10
    #     for i in range(1000):
    #         aR, bR = numpy.random.multivariate_normal(mn, cov)
    #         plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')    
    
    #     plt.plot(dum, chains[2]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Free")
    #     plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    #     plt.plot(dum, chains[0]["bR_use"].mean() + chains[0]["aR"].mean()*numpy.log10(dum),label="Inverse TF")  
    #     plt.xscale('log',base=10)
    #     plt.xlabel(r"$\hat{V}$")
    #     plt.ylabel(r"$\hat{m}$-$\mu$")
    #     plt.ylim((MR.max()+.5,MR.min()-.5))
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("output/{}/tf_cluster.png".format(subdir))
    #     plt.clf()

    plt.hist(numpy.log10(data["V_0p4R26"]),density=True)
    x=numpy.linspace(1.8,2.5,100)
    # plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[2]["omega_dist_use"].mean()) ,label="Free")
    # plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[1]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[0]["xi_dist"].mean(),chains[0]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.xlabel(r"$\log{(\hat{V})}$")
    plt.legend()
    plt.savefig("{}/output/hist_cluster.png".format(subdir))
    plt.clf()

    ###### Make a pull distribution
    MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
    
    # Predicted m = a * logV + b, where a and b are means from chain
    logV = numpy.log10(data["V_0p4R26"])
    a = chains[0]["aR"].mean()
    b = chains[0]["bR_use"].mean()
    if V0:
        m_pred = a * (logV- logV0) + b
    else:
        m_pred = a * logV + b
    
    # Pulls: (observed - predicted) / uncertainty
    pulls = (MR - m_pred) / numpy.array(data["R_MAG_SB26_ERR"])
    pull_mean, std = norm.fit(pulls)
    
    # Plot
    plt.hist(pulls, bins=30, alpha=0.7, label=f"Y3_full")
    x = numpy.linspace(-5, 5, 200)
    plt.xlabel("Pull")
    plt.ylabel("Log Density")
    plt.title("Pull Distribution")
    plt.yscale('log') 
    plt.axvline(x=0)
    plt.axvline(x=0, color='k', label = f"mean= {pull_mean:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{subdir}/output/pull_distribution.png")
    plt.clf()

    ######### Save data as a pickle file
    chain_df = chains[0]
    
    # Extract samples
    aR_samples   = chain_df["aR"].to_numpy()
    bR_samples   = chain_df["bR_use"].to_numpy()
    sigR_samples = chain_df["sigR"].to_numpy()
    
    # Combine into list of 1D arrays (to match Hyperfit pickles)
    tfr_mcmc_samples = [aR_samples, bR_samples, sigR_samples]
    
    sample_matrix = numpy.vstack(tfr_mcmc_samples)
    cov_ab = numpy.cov(sample_matrix)
    
    # Save to pickle
    output_path = f"{subdir}/output/cluster_result.pickle"
    with open(output_path, "wb") as f:
        pickle.dump((cov_ab, tfr_mcmc_samples, logV0), f)


    ######## Extra diagnostic when running with subsamples to investigate each individually:
    # Loop through each subsample to create a corner plot
    for i in range(62):  # 20 subsamples
        subset_chains = []
        for j in range(1, 5):  # 4 chains per subsample
            chain_path = f"{subdir}/cluster_411_{i:02d}_{j}.csv"
            try:
                df = pandas.read_csv(chain_path, comment='#')
                df["bR_use"] = df["bR.1"].mean() - df["xi_dist"] * df["aR"]
                df["omega_dist_use"] = df["omega_dist"] * numpy.cos(df["atanAR"])
                df['sigR_proj'] = 1 / numpy.cos(df["atanAR"]) * df["sigR"]
                df['theta_2'] = df["atanAR"] + numpy.pi / 2
                subset_chains.append(df)
            except Exception as e:
                print(f"Failed to read {chain_path}: {e}")
                continue
        
        if len(subset_chains) == 0:
            print(f"No valid chains for subsample {i}")
            continue
    
        combined_df = pandas.concat(subset_chains, ignore_index=True)
    
        # Create ChainConsumer object
        subset_c = ChainConsumer()
        subset_c.add_chain(Chain(samples=combined_df[["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2"]], name=f"Subset {i}"))
        
        # Set consistent plot configuration
        subset_c.set_plot_config(
            PlotConfig(
                summary_font_size=20, 
                label_font_size=20, 
                constrain=False, 
                legend_kwargs={'prop': {'size': 20}}, 
                labels={
                    "aR": r"$a$", 
                    "bR_use": r"$b$", 
                    "sigR": r"$\sigma_R$",  
                    "xi_dist": r"$\log{V}_{TF}$", 
                    "omega_dist_use": r"$\sigma_{\log{V}_{TF}}$", 
                    "theta_2": r"$\theta_2$"
                }
            )
        )
        
        fig = subset_c.plotter.plot()
        output_path = f"{subdir}/output/corner_cluster_subset_{i:02d}.png"
        fig.savefig(output_path)
        plt.clf()



# cluster_perp(file='Y3_full_v3/jura_Y3_subsample_00',subdir='/pscratch/sd/s/sgmoore1/Y3_full_v3', nocluster=True, V0=True)
cluster_perp('iron_cluster_Tully_nocluster_V0','iron', nocluster=True, V0=True)
wfe


























def cluster_all():
    desi_sga_dir = "/global/homes/s/sgmoore1/DESI_SGA/"
    
    # ncluster  = len(glob.glob(desi_sga_dir+"/TF/Y3/output_*.txt"))
    # print(ncluster)
    # wef
    infile = json.load(open("data/jura_cluster.json",))
    ncluster = infile["N_cluster"]

    chains=[]
    c = ChainConsumer()
    c2 = ChainConsumer()
    fig_b, ax_b = plt.subplots()
    fig_b2, ax_b2 = plt.subplots()

    for _ in [3,4,5]:
        if _ == 3:
            name = 'Inverse TF'
        elif _==4:
            name = 'Perpendicular'
        elif _==5:
            name = "Free"   
        dum=[pandas.read_csv("output/{}/cluster_{}11_{}.csv".format(subdir,_,i),comment='#') for i in range(1,5)]

        bRcols=["bR.{}".format(cin) for cin in range(1,ncluster+1)]
        for df_ in dum:
            df_["bR_use"] = df_[bRcols].mean(axis=1) - df_["xi_dist"]*df_["aR"]
            df_["omega_dist_use"] = df_["omega_dist"] * numpy.cos(df_["atanAR"])
            if _==3:
                df_['sigR_proj'] = -df_['aR']* df_['sigR']
                df_['theta_2'] = numpy.random.normal(0,0.00001, len(df_['aR']))
            elif _==4:
                df_['sigR_proj'] = 1/numpy.cos(df_["atanAR"])*df_["sigR"]
                df_['theta_2'] = df_["atanAR"]+numpy.pi/2
            elif _==5:
                _y = numpy.sin(df_["theta_2"])*df_["sigR"]
                _x = numpy.cos(df_["theta_2"])*df_["sigR"]
                df_['sigR_proj'] = _y - _x* df_['aR']
        dum=pandas.concat(dum)
        chains.append(dum)
        # dum=pandas.read_csv("output/temp_{}.csv".format(1),comment='#')

        lrmn=[]
        lrmn2=[]
        for cin in range(1,ncluster+1):
            use = dum["bR.{}".format(cin)] - dum["xi_dist"]*dum["aR"]
            lrmn.append(numpy.percentile(use, (32,50,68)))
            use2 = dum["bR.{}".format(cin)] - dum["bR.{}".format(1)] 
            lrmn2.append(numpy.percentile(use2, (32,50,68)))

        lrmn = numpy.array(lrmn).transpose()
        lrmn[0]=lrmn[1]-lrmn[0]
        lrmn[2]=lrmn[2]-lrmn[1]
        yerr=numpy.array((lrmn[0],lrmn[2]))

        lrmn2 = numpy.array(lrmn2).transpose()
        lrmn2[0]=lrmn2[1]-lrmn2[0]
        lrmn2[2]=lrmn2[2]-lrmn2[1]
        yerr2=numpy.array((lrmn2[0],lrmn2[2]))        

        if _ == 3:
            off = 0
            ax_b.errorbar(numpy.array(infile["mu"])+off,lrmn[1],fmt="+",yerr=yerr,label=name)
            ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)
        elif _==4:
            off = 0.025
            ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)
        elif _==5:
            off = 0.05
            ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)

        # c = ChainConsumer()
        c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2"]], name=name))
        c2.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))

        if _==3:
            _v = numpy.percentile(dum["aR"]*dum["sigR"],(32,50,100-32))
        elif _==4:
            _v = numpy.percentile(numpy.sin(dum["atanAR"])*dum["sigR"],(32,50,100-32))
        print("${:5.3f}_{:5.3f}^+{:5.3f}$".format(_v[1], -_v[1]+_v[0],_v[2]-_v[1]))      

        fig = c.plotter.plot()
        plt.savefig("output/{}/corner_cluster_{}.png".format(suubdir,_))
        # plt.show()
        plt.clf()

        print(c.analysis.get_latex_table())

    
    ax_b.set_xlabel(r"$\mu$")
    ax_b.set_ylabel(r"$b$")
    ax_b.legend()
    fig_b.tight_layout()
    fig_b.savefig("output/{}/b_cluster.png".format(subdir))

    ax_b2.set_xlabel(r"$\mu$")
    ax_b2.set_ylabel(r"$b-b_0$")
    ax_b2.legend(loc=3)
    fig_b2.tight_layout()
    fig_b2.savefig("output/{}/b_cluster2.png".format(subdir))   

    plt.clf()

    c.set_plot_config(
        PlotConfig(
            summary_font_size=20, label_font_size=20, legend_kwargs={'prop':{'size':20}}, labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$", "theta_2": r"$\theta_2$"}, 
        )
    )

    fig = c.plotter.plot()
    allaxes = fig.get_axes()

    allaxes[35].set_ylim((0,65))
    plt.savefig("output/{}/corner_cluster.png".format(subdir))
    # plt.show()
    plt.clf()
    print(c.analysis.get_latex_table())
    print(c2.analysis.get_latex_table())



    fn="data/"+file+".json"
    # fn_all="output/jura_cluster_all.json"


    with open(fn, 'r') as f:
        data = json.load(f)

    # with open(fn_all, 'r') as f:
    #     data_all = json.load(f)

    MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
    # MR_all  = numpy.array(data_all["R_MAG_SB26"]) - numpy.array(data_all["mu_all"])

    # plt.errorbar(data_all["V_0p4R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p4R26_err"], fmt="+", label="cut",color='black',alpha=0.5)
    plt.errorbar(data["V_0p4R26"], MR ,yerr=data["R_MAG_SB26_ERR"],xerr=data["V_0p4R26_err"], fmt="+", label="sample",color='black')

    index = 0
    for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
        if True:
            plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
        index = index+data["N_per_cluster"][i]
    
    print("Length of chains:", len(chains))
    if len(chains) > 1:
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
        plt.axhline(-18, linestyle='dotted') 
        plt.axvline(70, linestyle='dotted')    
        plt.xscale('log',base=10)
        plt.xlabel(r"$\hat{V}$")
        plt.ylabel(r"$\hat{m}$-$\mu$")
        plt.ylim((MR.max()+.5,MR.min()-.5))
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/{}/tf_cluster_all.png".format(subdir)) 
        plt.clf()

    # index = 0
    # for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
    #     if True:
    #         plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=data["R_MAG_SB26_ERR"][index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
    #     index = index+data["N_per_cluster"][i]
    # if len(chains) > 2:
    #     mn = chains[2][["aR","bR_use"]].mean()
    #     cov = chains[2][["aR","bR_use"]].cov()
        
    #     dum = numpy.array(plt.xlim())
    #     if dum[0] <=0:
    #         dum[0]=10
    #     for i in range(1000):
    #         aR, bR = numpy.random.multivariate_normal(mn, cov)
    #         plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')    
    
    #     plt.plot(dum, chains[2]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Free")
    #     plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    #     plt.plot(dum, chains[0]["bR_use"].mean() + chains[0]["aR"].mean()*numpy.log10(dum),label="Inverse TF")  
    #     plt.xscale('log',base=10)
    #     plt.xlabel(r"$\hat{V}$")
    #     plt.ylabel(r"$\hat{m}$-$\mu$")
    #     plt.ylim((MR.max()+.5,MR.min()-.5))
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("output/{}/tf_cluster.png".format(subdir))
    #     plt.clf()

    plt.hist(numpy.log10(data["V_0p4R26"]),density=True)
    x=numpy.linspace(1.8,2.5,100)
    # plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[2]["omega_dist_use"].mean()) ,label="Free")
    # plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[1]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[0]["xi_dist"].mean(),chains[0]["omega_dist_use"].mean()) ,label="Inverse TF")
    plt.xlabel(r"$\log{(\hat{V})}$")
    plt.legend()
    plt.savefig("output/{}/hist_cluster.png".format(subdir))
    plt.clf()



