'''
This script is the primary diagnostic tool that reads in the output chains from any stan fit. It generates the following plots:
- Histogram of the best fit slopes
- Corner plots of best fit parameters
- Histogram of log(v) distribution
- Scatterplot with best-fit TFR 
- Pull distribution
- Optional: For fits in bins/clusters, plot the best fit TFR in each bin

It also stores the best fit parameters in a pickle file, which is useful for downstream analyses.
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
import math
import numpy as np
from matplotlib.ticker import LogFormatter

matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["lines.linewidth"] = 2


def cluster_perp(file, subdir, nocluster=False, V0=True, individual_plots=False, num_chains=4, rcorr=False):
    desi_sga_dir = "/global/homes/s/sgmoore1/DESI_SGA/"
    
    # ncluster  = len(glob.glob(desi_sga_dir+"/TF/Y3/output_*.txt"))
    # print(ncluster)
    # wef
    infile = json.load(open("data/{}.json".format(file)))
    logV0 = infile["logV0"]
    
    if nocluster:
        ncluster=1
    else:
        ncluster = infile["N_cluster"]
    
    print(ncluster)

    chains=[]
    c = ChainConsumer()
    
    c2 = ChainConsumer()
    fig_b, ax_b = plt.subplots()
    fig_b2, ax_b2 = plt.subplots()

    for _ in [4]:#[3,4,5]:
        if _ == 3:
            name = 'Inverse TF'
        elif _==4:
            name = 'Perpendicular'
        elif _==5:
            name = "Free"   
        dum=[pandas.read_csv("{}/cluster_{}11_{}.csv".format(subdir,_,i),comment='#') for i in range(1,num_chains+1)]

        bRcols=["bR.{}".format(cin) for cin in range(1,ncluster+1)]
        for df_ in dum:
            df_["bR_use"] = df_[bRcols].mean(axis=1) - df_["xi_dist"]*df_["aR"]
        # for df_ in dum:
        #     df_["bR_use"] = df_["bR"] - df_["xi_dist"]*df_["aR"]
            
            df_["omega_dist_use"] = df_["omega_dist"] * numpy.cos(df_["atanAR"])
            # df_['sigR'] = 0.4
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

        # Diagnostics for aR
        print("\n--- Diagnostics for aR ---")
        print("aR dtype:", dum["aR"].dtype)
        print("aR nulls:", dum["aR"].isnull().sum())
        print("aR unique values:", dum["aR"].nunique())
        print(dum["aR"].describe())
        
        # Save histogram instead of showing it
        plt.hist(dum["aR"], bins=30)
        plt.title("Histogram of aR")
        plt.tight_layout()
        plt.savefig("{}/aR_hist_debug.png".format(subdir))
        plt.clf()


        
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

        if _ == 3:
            off = 0
            ax_b.errorbar(numpy.array(infile["mu"])+off,lrmn[1],fmt="+",yerr=yerr,label=name)
            ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)
        elif _==4:
            off = 0.025
            if nocluster:
                ax_b2.errorbar(numpy.mean(infile["mu"]) + off,[lrmn2[1][0]],fmt="+",yerr=yerr2,label=name)
            else:
                ax_b2.errorbar(numpy.array(infile["mu"]) + off, lrmn2[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)

            # ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)
        elif _==5:
            off = 0.05
            ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)

        # c = ChainConsumer()
        c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2"]], name=name))
        c2.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))
        # c.add_chain(Chain(samples=dum[["aR","bR_use","xi_dist","omega_dist_use","theta_2"]], name=name))
        # c2.add_chain(Chain(samples=dum[["aR","bR_use","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))


        cred = numpy.percentile(dum["aR"], [16, 50, 84])
        aR_summary = [cred[1], cred[2] - cred[1], cred[1] - cred[0]]
        
        # Ensure the key exists
        if name not in c.analysis._summaries:
            c.analysis._summaries[name] = {}
        
        # Inject summary
        c.analysis._summaries[name]["aR"] = aR_summary
        
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
        summary = c.analysis.get_summary()
        print("Summary for 'aR':", summary.get("aR"))
   

        if _==3:
            _v = numpy.percentile(dum["aR"]*dum["sigR"],(32,50,100-32))
        elif _==4:
            _v = numpy.percentile(numpy.sin(dum["atanAR"])*dum["sigR"],(32,50,100-32))
        print("${:5.3f}_{:5.3f}^+{:5.3f}$".format(_v[1], -_v[1]+_v[0],_v[2]-_v[1]))      

        fig = c.plotter.plot()
        plt.savefig("{}/corner_cluster_{}.png".format(subdir,_))
        # plt.show()
        plt.clf()

        print(c.analysis.get_latex_table())

    
    # ax_b.set_xlabel(r"$\mu$") ### this only matters for the inverse case
    # ax_b.set_ylabel(r"$b$")
    # ax_b.legend()
    # fig_b.tight_layout()
    # fig_b.savefig("{}/b_cluster.png".format(subdir))

    ax_b2.set_xlabel(r"$\mu$")
    ax_b2.set_ylabel(r"$b-b_0$")
    ax_b2.legend(loc=3)
    fig_b2.tight_layout()
    fig_b2.savefig("{}/b_cluster2.png".format(subdir))   

    plt.clf()

    c.set_plot_config(
        PlotConfig(
            summary_font_size=20, label_font_size=20, constrain=False, legend_kwargs={'prop':{'size':20}}, labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$", "theta_2": r"$\theta_2$"}, 
        )
    )

    fig = c.plotter.plot()
    allaxes = fig.get_axes()

    allaxes[35].set_ylim((0,65))
    plt.savefig("{}/corner_cluster.png".format(subdir))
    # plt.show()
    plt.clf()
    print(c.analysis.get_latex_table())
    print(c2.analysis.get_latex_table())



    fn="data/"+file+".json"
    # fn_all="output/jura_cluster_all.json"


    with open(fn, 'r') as f:
        data = json.load(f)
        print(len(data))

    # with open(fn_all, 'r') as f:
    #     data_all = json.load(f)
    if rcorr:
        MR = numpy.array(data["R_MAG_SB26"]) + numpy.array(data['R_correction']) - numpy.array(data["mu_all"])
        # MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
        MR_err = numpy.sqrt(numpy.array(data["R_MAG_SB26_ERR"])**2 + numpy.array(data["R_correction_err"])**2)
    else:
        MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
        MR_err = data["R_MAG_SB26_ERR"]
    # MR_all  = numpy.array(data_all["R_MAG_SB26"]) - numpy.array(data_all["mu_all"])

    # plt.errorbar(data_all["V_0p4R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p4R26_err"], fmt="+", label="cut",color='black',alpha=0.5)
    if nocluster:
        plt.errorbar(data["V_0p4R26"], MR ,yerr=MR_err,xerr=data["V_0p4R26_err"], fmt=".", label="sample", ecolor='lightgray')
    else:
        index = 0
        for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
            if True:
                plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=MR_err[index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
            
            index = index+data["N_per_cluster"][i]
    
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
        plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')    
    # plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    plt.plot(dum, mean_b + chains[0]["aR"].mean()*numpy.log10(dum),label="Bayesian")
    plt.axhline(-18, linestyle='dotted') 
    plt.axvline(70, linestyle='dotted')    
    plt.xscale('log',base=10)
    plt.xlabel(r"$\hat{V}$")
    plt.ylabel(r"$\hat{m}$-$\mu$")
    plt.ylim((MR.max()+.5,MR.min()-.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/tf_cluster_all.png".format(subdir)) 
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
    #     plt.savefig("{}/tf_cluster.png".format(subdir))
    #     plt.clf()

    plt.hist(numpy.log10(data["V_0p4R26"]),density=True, bins=50)
    x=numpy.linspace(1.8,2.5,100)
    # plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[2]["omega_dist_use"].mean()) ,label="Free")
    # plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[1]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[0]["xi_dist"].mean(),chains[0]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.xlabel(r"$\log{(\hat{V})}$")
    plt.legend()
    plt.savefig("{}/hist_cluster.png".format(subdir))
    plt.clf()

    ###### Make a pull distribution
    # MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
    
    # Predicted m = a * logV + b, where a and b are means from chain
    logV = numpy.log10(data["V_0p4R26"])
    a = chains[0]["aR"].mean()
    b = chains[0]["bR_use"].mean()
    sig = chains[0]["sigR"].mean()* np.cos(np.arctan(a))

    MR_err = np.sqrt(MR_err**2 + sig**2)
    if V0:
        m_pred = a * (logV- logV0) + b
    else:
        m_pred = a * logV + b
    
    # Pulls: (observed - predicted) / uncertainty
    pulls = (MR - m_pred) / MR_err
    pull_mean, std = norm.fit(pulls)
    
    # Plot
    plt.hist(pulls, bins=30, alpha=0.7, label=f"{file}")
    x = numpy.linspace(-5, 5, 200)
    plt.xlabel("Pull")
    plt.ylabel("Log Density")
    plt.title("Pull Distribution")
    plt.yscale('log') 
    plt.axvline(x=0)
    plt.axvline(x=pull_mean, color='k', label = f"mean= {pull_mean:.4f}")
    plt.axvline(x=pull_mean + std, color='k', alpha=0.5, label = f"std= {std:.4f}")
    plt.axvline(x=pull_mean - std, color='k', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{subdir}/pull_distribution.png")
    plt.clf()

    ######### Save data as a pickle file
    chain_df = chains[0]
    
    aR_samples   = chain_df["aR"].to_numpy()
    bR_samples   = chain_df["bR_use"].to_numpy()
    sigR_samples = chain_df["sigR"].to_numpy()
    
    # Collect all samples into a list
    tfr_mcmc_samples = [aR_samples, bR_samples]
    
    # Add per-cluster intercepts
    for i in range(0, ncluster + 1):  # assuming cluster indices go from 0 to ncluster
        key = f"bR.{i}"
        if key in chain_df:
            tfr_mcmc_samples.append(chain_df[key].to_numpy())
        else:
            print(f"Warning: {key} not found in MCMC chain.")

    tfr_mcmc_samples.append(sigR_samples)
    
    # # Combine into list of 1D arrays (to match Hyperfit pickles)
    # tfr_mcmc_samples = [aR_samples, bR_samples,  sigR_samples]
    
    sample_matrix = numpy.vstack(tfr_mcmc_samples)
    cov_ab = numpy.cov(sample_matrix)
    
    # Save to pickle
    output_path = f"{subdir}/cluster_result.pickle"
    with open(output_path, "wb") as f:
        pickle.dump((cov_ab, tfr_mcmc_samples, logV0), f)

    ####### Build a full pickle file 
    chain_df = chains[0]

    param_names = ["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2"]
    tfr_mcmc_samples = [chain_df[param].to_numpy() for param in param_names]

    sample_matrix = numpy.vstack(tfr_mcmc_samples)
    cov_matrix = numpy.cov(sample_matrix)

    output_path = f"{subdir}/cluster_result_all.pickle"
    with open(output_path, "wb") as f:
        pickle.dump((cov_matrix, tfr_mcmc_samples, logV0), f)


    ######## Plot individual cluster fits
    if individual_plots:
        n_clusters = data["N_cluster"]
        ncols = 4
        nrows = math.ceil(n_clusters / ncols)
        
        # Create figure and axes grid
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 6 * nrows), sharex=False, sharey=False)
        axs = axs.flatten()  # Make it easy to index
        
        index = 0
        for i in range(n_clusters):
            n = data["N_per_cluster"][i]
            
            v = np.array(data["V_0p4R26"][index:index+n])
            v_err = np.array(data["V_0p4R26_err"][index:index+n])
            mr = MR[index:index+n]
            mr_err = MR_err[index:index+n]
            
            ax = axs[i]
            ax.errorbar(v, mr, xerr=v_err, yerr=mr_err, fmt='.', alpha=0.8)
            
            # Line range
            dum = np.array([v.min()*0.9, v.max()*1.1])
            if dum[0] <= 0:
                dum[0] = 10
        
            bR_key = f'bR.{i+1}'  # Capital 'R'
            bR_samples = chains[0][bR_key].to_numpy()
            aR_samples = chains[0]['aR'].to_numpy()
            
            for j in range(1000):
                aR = aR_samples[j]
                bR = bR_samples[j]
                if V0:
                    bR -= aR * logV0
                ax.plot(dum, bR + aR * np.log10(dum), alpha=0.01, color='gray')
            
            # Mean line for that cluster
            aR_mean = aR_samples.mean()
            bR_mean = bR_samples.mean()
            if V0:
                bR_mean -= aR_mean * logV0
            ax.plot(dum, bR_mean + aR_mean * np.log10(dum), color='red', lw=1)
        
            ax.set_title(f'Cluster {i}')
            ax.set_xscale('log', base=10)
            ax.axhline(-18, linestyle='dotted', color='gray', lw=0.8)
            ax.axvline(70, linestyle='dotted', color='gray', lw=0.8)
            ax.tick_params(labelsize=8)
            # ax.ticklabel_format(style='plain', axis='x')
            ax.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
            ax.set_ylim([-18.5, -23])
        
            index += n
        
        # Remove unused axes if N_cluster is not a multiple of 4
        for i in range(n_clusters, len(axs)):
            fig.delaxes(axs[i])
        
        fig.supxlabel(r"$\hat{V}$")
        fig.supylabel(r"$\hat{m} - \mu$")
        fig.tight_layout()
        plt.savefig(f"{subdir}/tf_cluster_grid.png")
        plt.close()

### Call the function 
cluster_perp('iron_cluster_zbins_nocluster_v15_rcorr','/pscratch/sd/s/sgmoore1/stan_outputs/iron/zbins/unclustered/v15/new_pa/rcorr/no_jacobian/',nocluster=True, V0=True, num_chains=4, individual_plots=False, rcorr=True)


# cluster_perp('iron_cluster_Tully_zbins_subsample','iron/zbins', nocluster=False, V0=True, num_chains=8)
# cluster_perp('jura_cluster_full_V0','tully_full/V0', nocluster=False, V0=True)






















def cluster_all(file, subdir, nocluster=False, V0=True, individual_plots=False, num_chains=4, rcorr=False):
    desi_sga_dir = "/global/homes/s/sgmoore1/DESI_SGA/"
    
    # ncluster  = len(glob.glob(desi_sga_dir+"/TF/Y3/output_*.txt"))
    # print(ncluster)
    # wef
    infile = json.load(open("data/{}.json".format(file)))
    logV0 = infile["logV0"]
    
    if nocluster:
        ncluster=1
    else:
        ncluster = infile["N_cluster"]
    
    print(ncluster)

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
        dum=[pandas.read_csv("{}/cluster_{}11_{}.csv".format(subdir,_,i),comment='#') for i in range(1,num_chains+1)]

        bRcols=["bR.{}".format(cin) for cin in range(1,ncluster+1)]
        for df_ in dum:
            df_["bR_use"] = df_[bRcols].mean(axis=1) - df_["xi_dist"]*df_["aR"]
        # for df_ in dum:
        #     df_["bR_use"] = df_["bR"] - df_["xi_dist"]*df_["aR"]
            
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

        # # Diagnostics for aR
        # print("\n--- Diagnostics for aR ---")
        # print("aR dtype:", dum["aR"].dtype)
        # print("aR nulls:", dum["aR"].isnull().sum())
        # print("aR unique values:", dum["aR"].nunique())
        # print(dum["aR"].describe())
        
        # # Save histogram instead of showing it
        # plt.hist(dum["aR"], bins=30)
        # plt.title("Histogram of aR")
        # plt.tight_layout()
        # plt.savefig("{}/aR_hist_debug.png".format(subdir))
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

        if _ == 3:
            off = 0
            if nocluster:
                ax_b.errorbar(numpy.mean(infile["mu"]) + off,[lrmn[1][0]],fmt="+",yerr=yerr2,label=name)
                ax_b2.errorbar(numpy.mean(infile["mu"]) + off,[lrmn2[1][0]],fmt="+",yerr=yerr2,label=name)
            else:
                ax_b.errorbar(numpy.array(infile["mu"]) + off, lrmn[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)
                ax_b2.errorbar(numpy.array(infile["mu"]) + off, lrmn2[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)
        elif _==4:
            off = 0.025
            if nocluster:
                ax_b2.errorbar(numpy.mean(infile["mu"]) + off,[lrmn2[1][0]],fmt="+",yerr=yerr2,label=name)
            else:
                ax_b2.errorbar(numpy.array(infile["mu"]) + off, lrmn2[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)

        elif _==5:
            off = 0.05
            if nocluster:
                ax_b2.errorbar(numpy.mean(infile["mu"]) + off,[lrmn2[1][0]],fmt="+",yerr=yerr2,label=name)
            else:
                ax_b2.errorbar(numpy.array(infile["mu"]) + off, lrmn2[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)

        # c = ChainConsumer()
        c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2"]], name=name))
        c2.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))


        # if _ == 3:
        #     off = 0
        #     ax_b.errorbar(numpy.array(infile["mu"])+off,lrmn[1],fmt="+",yerr=yerr,label=name)
        #     ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)
        # elif _==4:
        #     off = 0.025
        #     if nocluster:
        #         ax_b2.errorbar(numpy.mean(infile["mu"]) + off,[lrmn2[1][0]],fmt="+",yerr=yerr2,label=name)
        #     else:
        #         ax_b2.errorbar(numpy.array(infile["mu"]) + off, lrmn2[1],fmt="+",yerr=numpy.array([[yerr2[0][0]], [yerr2[1][0]]]), label=name)

        #     # ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)
        # elif _==5:
        #     off = 0.05
        #     ax_b2.errorbar(numpy.array(infile["mu"])+off,lrmn2[1],fmt="+",yerr=yerr2,label=name)

        # # c = ChainConsumer()
        # c.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2"]], name=name))
        # c2.add_chain(Chain(samples=dum[["aR","bR_use","sigR","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))
        # # c.add_chain(Chain(samples=dum[["aR","bR_use","xi_dist","omega_dist_use","theta_2"]], name=name))
        # # c2.add_chain(Chain(samples=dum[["aR","bR_use","xi_dist","omega_dist_use","theta_2","sigR_proj"]], name=name))




        cred = numpy.percentile(dum["aR"], [16, 50, 84])
        aR_summary = [cred[1], cred[2] - cred[1], cred[1] - cred[0]]
        
        # Ensure the key exists
        if name not in c.analysis._summaries:
            c.analysis._summaries[name] = {}
        
        # Inject summary
        c.analysis._summaries[name]["aR"] = aR_summary
        
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
        summary = c.analysis.get_summary()
        print("Summary for 'aR':", summary.get("aR"))


        if _==3:
            _v = numpy.percentile(dum["aR"]*dum["sigR"],(32,50,100-32))
        elif _==4:
            _v = numpy.percentile(numpy.sin(dum["atanAR"])*dum["sigR"],(32,50,100-32))
        print("${:5.3f}_{:5.3f}^+{:5.3f}$".format(_v[1], -_v[1]+_v[0],_v[2]-_v[1]))      

        fig = c.plotter.plot()
        plt.savefig("{}/corner_cluster_{}.png".format(subdir,_))
        # plt.show()
        plt.clf()

        print(c.analysis.get_latex_table())

    
    # ax_b.set_xlabel(r"$\mu$") ### this only matters for the inverse case
    # ax_b.set_ylabel(r"$b$")
    # ax_b.legend()
    # fig_b.tight_layout()
    # fig_b.savefig("{}/b_cluster.png".format(subdir))

    # ax_b2.set_xlabel(r"$\mu$")
    # ax_b2.set_ylabel(r"$b-b_0$")
    # ax_b2.legend(loc=3)
    # fig_b2.tight_layout()
    # fig_b2.savefig("{}/b_cluster2.png".format(subdir))   

    plt.clf()

    c.set_plot_config(
        PlotConfig(
            summary_font_size=20, label_font_size=20, constrain=False, legend_kwargs={'prop':{'size':20}}, labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$", "theta_2": r"$\theta_2$"}, 
        )
    )

    fig = c.plotter.plot()
    allaxes = fig.get_axes()

    allaxes[35].set_ylim((0,65))
    plt.savefig("{}/corner_cluster.png".format(subdir))
    # plt.show()
    plt.clf()
    print(c.analysis.get_latex_table())
    print(c2.analysis.get_latex_table())



    fn="data/"+file+".json"
    # fn_all="output/jura_cluster_all.json"
    
    with open(fn, 'r') as f:
        data = json.load(f)
        print(len(data))

    # with open(fn_all, 'r') as f:
    #     data_all = json.load(f)
    if rcorr:
        MR = numpy.array(data["R_MAG_SB26"]) + numpy.array(data['R_correction']) - numpy.array(data["mu_all"])
        # MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
        MR_err = numpy.sqrt(numpy.array(data["R_MAG_SB26_ERR"])**2 + numpy.array(data["R_correction_err"])**2)
    else:
        MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
        MR_err = data["R_MAG_SB26_ERR"]
    # MR_all  = numpy.array(data_all["R_MAG_SB26"]) - numpy.array(data_all["mu_all"])

    # plt.errorbar(data_all["V_0p4R26"], MR_all ,yerr=data_all["R_MAG_SB26_ERR"],xerr=data_all["V_0p4R26_err"], fmt="+", label="cut",color='black',alpha=0.5)
    if nocluster:
        plt.errorbar(data["V_0p4R26"], MR ,yerr=MR_err,xerr=data["V_0p4R26_err"], fmt=".", label="sample", ecolor='lightgray')
    else:
        index = 0
        for i in range(0,data["N_cluster"]): #range(data["N_cluster"]):
            if True:
                plt.errorbar(data["V_0p4R26"][index:index+data["N_per_cluster"][i]], MR[index:index+data["N_per_cluster"][i]] ,yerr=MR_err[index:index+data["N_per_cluster"][i]],xerr=data["V_0p4R26_err"][index:index+data["N_per_cluster"][i]], fmt=".")
            
            index = index+data["N_per_cluster"][i]
    
    mn = chains[1][["aR","bR_use"]].mean()
    cov = chains[1][["aR","bR_use"]].cov()
    mean_b = [0,0,0]
    dum = numpy.array(plt.xlim())
    if dum[0] <=0:
        dum[0]=10
    for i in range(1000):
        aR, bR = numpy.random.multivariate_normal(mn, cov)
        if V0:
            bR -= aR*logV0
            mean_b[0] = chains[0]["bR_use"].mean() - chains[0]["aR"].mean()*logV0
            mean_b[1] = chains[1]["bR_use"].mean() - chains[0]["aR"].mean()*logV0
            mean_b[2] = chains[2]["bR_use"].mean() - chains[0]["aR"].mean()*logV0
        else: 
            mean_b[0] = chains[0]["bR_use"].mean()
            mean_b[1] = chains[1]["bR_use"].mean()
            mean_b[2] = chains[2]["bR_use"].mean()
        plt.plot(dum, bR + aR*numpy.log10(dum),alpha=0.01,color='black')   

    # plt.plot(dum, chains[1]["bR_use"].mean() + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    plt.plot(dum, mean_b[2] + chains[2]["aR"].mean()*numpy.log10(dum),label="Free")
    plt.plot(dum, mean_b[1] + chains[1]["aR"].mean()*numpy.log10(dum),label="Perpendicular")
    plt.plot(dum, mean_b[0] + chains[0]["aR"].mean()*numpy.log10(dum),label="Inverse TF")  
    
    plt.axhline(-18, linestyle='dotted') 
    plt.axvline(70, linestyle='dotted')    
    plt.xscale('log',base=10)
    plt.xlabel(r"$\hat{V}$")
    plt.ylabel(r"$\hat{m}$-$\mu$")
    plt.ylim((MR.max()+.5,MR.min()-.5))
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}/tf_cluster_all.png".format(subdir)) 
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
    #     plt.savefig("{}/tf_cluster.png".format(subdir))
    #     plt.clf()


    
    plt.hist(numpy.log10(data["V_0p4R26"]),density=True, bins=50)
    x=numpy.linspace(1.8,2.5,100)
    plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[2]["omega_dist_use"].mean()) ,label="Free")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[1]["xi_dist"].mean(),chains[1]["omega_dist_use"].mean()) ,label="Perpendicular")
    plt.plot(x, scipy.stats.norm.pdf(x, chains[0]["xi_dist"].mean(),chains[0]["omega_dist_use"].mean()) ,label="Inverse TF")
    plt.xlabel(r"$\log{(\hat{V})}$")
    plt.legend()
    plt.savefig("{}/hist_cluster.png".format(subdir))
    plt.clf()

    ###### Make a pull distribution (perpendicular fit)
    # MR = numpy.array(data["R_MAG_SB26"]) - numpy.array(data["mu_all"])
    
    # Predicted m = a * logV + b, where a and b are means from chain
    logV = numpy.log10(data["V_0p4R26"])
    a = chains[1]["aR"].mean()
    b = chains[1]["bR_use"].mean()
    if V0:
        m_pred = a * (logV- logV0) + b
    else:
        m_pred = a * logV + b
    
    # Pulls: (observed - predicted) / uncertainty
    pulls = (MR - m_pred) / MR_err
    pull_mean, std = norm.fit(pulls)
    
    # Plot
    plt.hist(pulls, bins=30, alpha=0.7, label=f"{file}")
    x = numpy.linspace(-5, 5, 200)
    plt.xlabel("Pull")
    plt.ylabel("Log Density")
    plt.title("Pull Distribution")
    plt.yscale('log') 
    plt.axvline(x=0)
    plt.axvline(x=pull_mean, color='k', label = f"mean= {pull_mean:.4f}")
    plt.axvline(x=pull_mean + std, color='k', alpha=0.5, label = f"std= {std:.4f}")
    plt.axvline(x=pull_mean - std, color='k', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{subdir}/pull_distribution.png")
    plt.clf()

    ######### Save data as a pickle file (for perpendicular case)
    chain_df = chains[0]
    
    aR_samples   = chain_df["aR"].to_numpy()
    bR_samples   = chain_df["bR_use"].to_numpy()
    sigR_samples = chain_df["sigR"].to_numpy()
    
    # Collect all samples into a list
    tfr_mcmc_samples = [aR_samples, bR_samples]
    
    # Add per-cluster intercepts
    for i in range(0, ncluster + 1):  # assuming cluster indices go from 0 to ncluster
        key = f"bR.{i}"
        if key in chain_df:
            tfr_mcmc_samples.append(chain_df[key].to_numpy())
        else:
            print(f"Warning: {key} not found in MCMC chain.")

    tfr_mcmc_samples.append(sigR_samples)
    
    # # Combine into list of 1D arrays (to match Hyperfit pickles)
    # tfr_mcmc_samples = [aR_samples, bR_samples,  sigR_samples]
    
    sample_matrix = numpy.vstack(tfr_mcmc_samples)
    cov_ab = numpy.cov(sample_matrix)
    
    # Save to pickle
    output_path = f"{subdir}/cluster_result.pickle"
    with open(output_path, "wb") as f:
        pickle.dump((cov_ab, tfr_mcmc_samples, logV0), f)

    ####### Build a full pickle file for the perpendicular case
    chain_df = chains[1]

    param_names = ["aR", "bR_use", "sigR", "xi_dist", "omega_dist_use", "theta_2"]
    tfr_mcmc_samples = [chain_df[param].to_numpy() for param in param_names]

    sample_matrix = numpy.vstack(tfr_mcmc_samples)
    cov_matrix = numpy.cov(sample_matrix)

    output_path = f"{subdir}/cluster_result_all.pickle"
    with open(output_path, "wb") as f:
        pickle.dump((cov_matrix, tfr_mcmc_samples, logV0), f)


    # ######## Plot individual cluster fits
    # if individual_plots:
    #     n_clusters = data["N_cluster"]
    #     ncols = 4
    #     nrows = math.ceil(n_clusters / ncols)
        
    #     # Create figure and axes grid
    #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 6 * nrows), sharex=False, sharey=False)
    #     axs = axs.flatten()  # Make it easy to index
        
    #     index = 0
    #     for i in range(n_clusters):
    #         n = data["N_per_cluster"][i]
            
    #         v = np.array(data["V_0p4R26"][index:index+n])
    #         v_err = np.array(data["V_0p4R26_err"][index:index+n])
    #         mr = MR[index:index+n]
    #         mr_err = MR_err[index:index+n]
            
    #         ax = axs[i]
    #         ax.errorbar(v, mr, xerr=v_err, yerr=mr_err, fmt='.', alpha=0.8)
            
    #         # Line range
    #         dum = np.array([v.min()*0.9, v.max()*1.1])
    #         if dum[0] <= 0:
    #             dum[0] = 10
        
    #         bR_key = f'bR.{i+1}'  # Capital 'R'
    #         bR_samples = chains[0][bR_key].to_numpy()
    #         aR_samples = chains[0]['aR'].to_numpy()
            
    #         for j in range(1000):
    #             aR = aR_samples[j]
    #             bR = bR_samples[j]
    #             if V0:
    #                 bR -= aR * logV0
    #             ax.plot(dum, bR + aR * np.log10(dum), alpha=0.01, color='gray')
            
    #         # Mean line for that cluster
    #         aR_mean = aR_samples.mean()
    #         bR_mean = bR_samples.mean()
    #         if V0:
    #             bR_mean -= aR_mean * logV0
    #         ax.plot(dum, bR_mean + aR_mean * np.log10(dum), color='red', lw=1)
        
    #         ax.set_title(f'Cluster {i}')
    #         ax.set_xscale('log', base=10)
    #         ax.axhline(-18, linestyle='dotted', color='gray', lw=0.8)
    #         ax.axvline(70, linestyle='dotted', color='gray', lw=0.8)
    #         ax.tick_params(labelsize=8)
    #         # ax.ticklabel_format(style='plain', axis='x')
    #         ax.xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    #         ax.set_ylim([-18.5, -23])
        
    #         index += n
        
    #     # Remove unused axes if N_cluster is not a multiple of 4
    #     for i in range(n_clusters, len(axs)):
    #         fig.delaxes(axs[i])
        
    #     fig.supxlabel(r"$\hat{V}$")
    #     fig.supylabel(r"$\hat{m} - \mu$")
    #     fig.tight_layout()
    #     plt.savefig(f"{subdir}/tf_cluster_grid.png")
    #     plt.close()

# cluster_all('iron_cluster_zbins_nocluster_v15_rcorr','/pscratch/sd/s/sgmoore1/stan_outputs/iron/zbins/unclustered/v15/new_pa/rcorr2',nocluster=True, V0=True, num_chains=4, individual_plots=False, rcorr=True)


