'''
This file generates bowtie plots based on the best fit TFR parameters, and is used to compare the quality of differing fits. Fits from "chains" are outputs of the Bayesian fit whereas the pickle files read in are from the Hyperfit outputs.
'''


import fitsio
import numpy as np
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
import os
import re
import pandas as pd

matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["lines.linewidth"] = 2


def plot_bowties_from_chain_dirs(file_json, chain_dirs, labels=None, colors=None,
                                 output_path="bowtie_chainconsumer_compare.png", nocluster=False, V0=True):
    """
    Plot bowtie confidence bands from multiple ChainConsumer chain directories,
    each containing 4 chain csv files.

    Parameters:
    - file_json: path to JSON data file with datapoints
    - chain_dirs: list of directories containing 4 chain CSV files each (cluster_411_1.csv ... cluster_411_4.csv)
    - labels: list of labels for each model (one per directory)
    - colors: list of colors for each model
    - output_path: where to save the figure
    - nocluster: if True, use ncluster=1 for plotting datapoints (optional)
    - V0: apply V0 offset to intercept if True
    """

    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(chain_dirs))]
    if colors is None:
        colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']

    # # Load JSON data for points
    with open(file_json, 'r') as f:
        data = json.load(f)

    # MR = np.array(data["R_MAG_SB26"]) - np.array(data["mu_all"])

    if nocluster:
        ncluster = 1
    else:
        ncluster = data.get("N_cluster", 1)

    fig, ax = plt.subplots(figsize=(8, 10))

    # # Plot datapoints grouped by cluster
    # index = 0
    # for i in range(ncluster):
    #     count = data["N_per_cluster"][i]
    #     ax.errorbar(data["V_0p4R26"][index:index+count],
    #                 MR[index:index+count],
    #                 xerr=data["V_0p4R26_err"][index:index+count],
    #                 yerr=data["R_MAG_SB26_ERR"][index:index+count],
    #                 fmt='.', alpha=0.7)
    #     index += count

    # X values for plotting lines/bands
    x_vals = np.linspace(10, 1000, 200)
    logx = np.log10(x_vals)

    # For each directory (model), load & concatenate all 4 chains, plot band
    for chain_dir, label, color in zip(chain_dirs, labels, colors):
        # Find all 4 cluster csv files matching pattern
        
        all_files = glob.glob(os.path.join(chain_dir, "cluster_411_*.csv"))
        chain_files = sorted([f for f in all_files if re.search(r'cluster_411_[1-4]\.csv$', f)])

        if len(chain_files) != 4:
            print(f"Warning: Expected 4 chain files in {chain_dir} but found {len(chain_files)}")

        # Read and concatenate chains
        dfs = []
        for f in chain_files:
            df = pandas.read_csv(f, comment="#")
            
            # Recompute bR_use = mean of bR.{i} - xi_dist * aR
            bR_cols = [col for col in df.columns if col.startswith("bR.")]
            if len(bR_cols) == 0:
                raise ValueError(f"No bR.* columns found in {f}")
            
            df["bR_use"] = df[bR_cols].mean(axis=1) - df["xi_dist"] * df["aR"]
            dfs.append(df)

            combined_chain = pandas.concat(dfs, ignore_index=True)

        a_samples = combined_chain['aR'].values
        b_samples = combined_chain['bR_use'].values

        # Apply V0 correction if needed
        if V0 and 'logV0' in data:
            logV0 = data['logV0']
            b_samples = b_samples - a_samples * logV0

        # Compute y predictions
        Y = np.array([b + a * logx for a, b in zip(a_samples, b_samples)])

        # Calculate 16th, 50th, 84th percentiles
        lower = np.percentile(Y, 16, axis=0)
        median = np.percentile(Y, 50, axis=0)
        upper = np.percentile(Y, 84, axis=0)

        # Plot median line and confidence band
        ax.plot(x_vals, median, color=color, label=label, linewidth=2)
        ax.fill_between(x_vals, lower, upper, color=color, alpha=0.3)

    # Formatting
    ax.set_xscale('log', base=10)
    ax.set_xlabel(r"$\hat{V}$", fontsize=18)
    ax.set_ylabel(r"$\hat{m} - \mu$", fontsize=18)
    ax.set_ylim(-23,-18)
    ax.set_xlim(50,1000)
    ax.invert_yaxis()

    # Reference lines
    ax.axvline(70, linestyle='dotted', color='gray')
    ax.axhline(-18, linestyle='dotted', color='gray')

    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved bowtie comparison plot to {output_path}")




def plot_bowties_with_chains_and_pickles(file_json, chain_dirs, pickle_files,
                                         labels_chains=None, labels_pickles=None,
                                         colors_chains=None, colors_pickles=None,
                                         output_path="bowtie_combined_plot.png",
                                         nocluster=False, V0=True):
    """
    Plot bowtie confidence bands from ChainConsumer chains and pickle MCMC samples on the same figure.

    Parameters:
    - file_json: path to JSON data file (needed for logV0)
    - chain_dirs: list of directories each containing 4 chain CSVs
    - pickle_files: list of paths to pickle files (cov, samples, logV0_alt)
    - labels_chains: labels for chain models
    - labels_pickles: labels for pickle models
    - colors_chains: colors for chain models
    - colors_pickles: colors for pickle models
    - output_path: save path for final figure
    - nocluster: ignore cluster-specific columns if True
    - V0: apply V0 correction to intercept
    """

    import re

    if labels_chains is None:
        labels_chains = [f"Chain Model {i+1}" for i in range(len(chain_dirs))]
    if labels_pickles is None:
        labels_pickles = [f"Pickle Model {i+1}" for i in range(len(pickle_files))]
    if colors_chains is None:
        colors_chains = ['tab:orange', 'tab:blue', 'tab:green']
    if colors_pickles is None:
        colors_pickles = ['tab:red', 'tab:purple', 'tab:brown']

    with open(file_json, 'r') as f:
        data = json.load(f)

    if nocluster:
        ncluster = 1
    else:
        ncluster = data.get("N_cluster", 1)

    fig, ax = plt.subplots(figsize=(8, 10))
    
    MR = np.array(data["R_MAG_SB26"]) - np.array(data["mu_all"])
    # Plot datapoints grouped by cluster
    # index = 0
    # for i in range(ncluster):
    #     count = data["N_per_cluster"][i]
    #     ax.errorbar(data["V_0p4R26"][index:index+count],
    #                 MR[index:index+count],
    #                 xerr=data["V_0p4R26_err"][index:index+count],
    #                 yerr=data["R_MAG_SB26_ERR"][index:index+count],
    #                 fmt='.', alpha=0.7)
    #     index += count

    
    logV0_data = data.get("logV0", np.log10(70))  # default fallback


    # X values
    x_vals = np.linspace(10, 1000, 200)
    logx = np.log10(x_vals)


    # ---------- Plot ChainConsumer-based models ----------
    for chain_dir, label, color in zip(chain_dirs, labels_chains, colors_chains):
        all_files = glob.glob(os.path.join(chain_dir, "cluster_411_*.csv"))
        chain_files = sorted([f for f in all_files if re.search(r'cluster_411_[1-4]\.csv$', f)])

        if len(chain_files) != 4:
            print(f"Warning: Expected 4 chain files in {chain_dir} but found {len(chain_files)}")

        dfs = []
        for f in chain_files:
            df = pandas.read_csv(f, comment="#")
            bR_cols = [col for col in df.columns if col.startswith("bR.")]
            if len(bR_cols) == 0:
                raise ValueError(f"No bR.* columns found in {f}")
            df["bR_use"] = df[bR_cols].mean(axis=1) - df["xi_dist"] * df["aR"]
            dfs.append(df)

        combined = pandas.concat(dfs, ignore_index=True)
        a_samples = combined['aR'].values
        b_samples = combined['bR_use'].values


        if V0:
            b_samples -= a_samples * logV0_data

        sigma = np.median(combined['sigR'])
        
        Y = np.array([b + a * logx for a, b in zip(a_samples, b_samples)])
        lower, median, upper = np.percentile(Y, [16, 50, 84], axis=0)

        ax.plot(x_vals, median, color=color, label=label, linewidth=2)
        ax.plot(x_vals, median + sigma, color=color, linestyle="dotted", linewidth=2)
        ax.plot(x_vals, median - sigma, color=color, linestyle="dotted", linewidth=2)
        ax.fill_between(x_vals, lower, upper, color=color, alpha=0.3)

 
    # ---------- Plot Pickle-based models ----------
    for pickle_file, label, color in zip(pickle_files, labels_pickles, colors_pickles):
        with open(pickle_file, "rb") as f:
            cov, samples, logV0_alt = pickle.load(f)
            
        a_samples = samples.T[:, 0]

        b0_samples = samples.T[:, 1]
        
        b_samples = b0_samples - a_samples * logV0_alt if V0 else b0_samples

        sigma = np.median(samples.T[:,-1])

        Y = np.array([b + a * logx for a, b in zip(a_samples, b_samples)])
        lower, median, upper = np.percentile(Y, [16, 50, 84], axis=0)

        ax.plot(x_vals, median, color=color, label=label, linestyle="--", linewidth=2)
        ax.plot(x_vals, median + sigma, color=color, linestyle="dotted", linewidth=2)
        ax.plot(x_vals, median - sigma, color=color, linestyle="dotted", linewidth=2)
        ax.fill_between(x_vals, lower, upper, color=color, alpha=0.2)

    # ---------- Plot formatting ----------
    ax.set_xscale('log', base=10)
    ax.set_xlabel(r"$V_{rot}$", fontsize=18)
    ax.set_ylabel(r"$M_{abs}$", fontsize=18)
    ax.set_ylim(-23, -18)
    ax.set_xlim(30, 500)
    ax.invert_yaxis()
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined bowtie plot to {output_path}")


# plot_bowties_with_chains_and_pickles(
#     # file_json="data/jura_cluster_williams_V0.json",
#     file_json='data/jura_cluster_full_V0.json',
#     chain_dirs=[
#         "output/williams/V0",
#         "output/tully_full/V0"
#                ],
#     pickle_files=[
#         '/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/cov_ab_jura_jointTFR_varyV0_binaryML_weightsVmax-1_williams_v2.pickle',
#         "/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/cov_ab_jura_jointTFR_varyV0_binaryML_weightsVmax-1_Tully_full_lessdwarves.pickle"
#     ],
#     labels_chains=[
#         "Williams (Bayesian)",
#         "Tully Full (Bayesian)"
#                   ],
#     labels_pickles=[
#         "Williams (Hyperfit)", 
#         "Tully Full (Hyperfit)"
#                    ],
#     colors_chains=[
#         "tab:blue",
#         "tab:orange"
#                   ],
#     colors_pickles=[
#         "tab:purple", 
#         "tab:red"
#                    ],
#     output_path="/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/Figures/combined_bowtie_plot_sigma.png"
# )

def plot_bowties(file_json, chain_dirs, pickle_files,
                                         labels_chains=None, labels_pickles=None,
                                         colors_chains=None, colors_pickles=None,
                                         output_path="bowtie_combined_plot.png",
                                         nocluster=False, V0=True):

    if labels_chains is None:
        labels_chains = [f"Chain Model {i+1}" for i in range(len(chain_dirs))]
    if labels_pickles is None:
        labels_pickles = [f"Pickle Model {i+1}" for i in range(len(pickle_files))]
    if colors_chains is None:
        colors_chains = ['tab:orange', 'tab:blue', 'tab:green']
    if colors_pickles is None:
        colors_pickles = ['tab:red', 'tab:purple', 'tab:brown']

    with open(file_json, 'r') as f:
        data = json.load(f)

    ncluster = 1 if nocluster else data.get("N_cluster", 1)

    fig, ax = plt.subplots(figsize=(8, 10))
    
    MR = np.array(data["R_MAG_SB26"]) - np.array(data["mu_all"])
    index = 0
    for i in range(ncluster):
        count = data["N_per_cluster"][i]
        ax.errorbar(data["V_0p4R26"][index:index+count],
                    MR[index:index+count],
                    xerr=data["V_0p4R26_err"][index:index+count],
                    yerr=data["R_MAG_SB26_ERR"][index:index+count],
                    fmt='.', alpha=0.3, color='grey', markersize=0.3)
        index += count

    logV0_data = data.get("logV0")
    x_vals = np.linspace(10, 1000, 200)
    logx = np.log10(x_vals)

    # ----------- Chains -------------
    for chain_dir, label, color in zip(chain_dirs, labels_chains, colors_chains):
        all_files = glob.glob(os.path.join(chain_dir, "cluster_411_*.csv"))
        chain_files = sorted([f for f in all_files if re.search(r'cluster_411_[1-4]\.csv$', f)])
        if len(chain_files) != 4:
            print(f"Warning: Expected 4 chain files in {chain_dir} but found {len(chain_files)}")

        dfs = []
        for f in chain_files:
            df = pd.read_csv(f, comment="#")
            bR_cols = [col for col in df.columns if col.startswith("bR.")]
            if not bR_cols:
                raise ValueError(f"No bR.* columns found in {f}")
            df["bR_use"] = df[bR_cols].mean(axis=1) - df["xi_dist"] * df["aR"]
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        a_samples = combined["aR"].values
        b_samples = combined["bR_use"].values
        if V0:
            b_samples -= a_samples * logV0_data

        a_mean = np.mean(a_samples)
        b_mean = np.mean(b_samples)
        cov_ab = np.cov(a_samples, b_samples)
        print(a_mean, b_mean, logV0_data)

        sigma = np.median(combined["sigR"].values)

        ab_samples = np.random.multivariate_normal([a_mean, b_mean], cov_ab, size=1000)
        Y = np.array([b + a * logx for a, b in ab_samples])
        lower, median, upper = np.percentile(Y, [16, 50, 84], axis=0)

        ax.plot(x_vals, a_mean * logx + b_mean, color=color, label=label, linewidth=2)
        ax.plot(x_vals, a_mean * logx + b_mean - sigma, color=color, linestyle='dotted', linewidth=1)
        ax.plot(x_vals, a_mean * logx + b_mean + sigma, color=color, linestyle='dotted', linewidth=1)
        ax.fill_between(x_vals, lower, upper, color=color, alpha=0.3)

    # ----------- Pickles -------------
    for pickle_file, label, color in zip(pickle_files, labels_pickles, colors_pickles):
        with open(pickle_file, "rb") as f:
            cov, samples, logV0_alt = pickle.load(f)

        # Use same structure as working script
        a_samples_2 = samples.T[:, 0]


        a_mean_2 = np.mean(a_samples_2)
        # b_mean_2 = -19.941 - a_mean_2*logV0_data #the directly comparable value we computed before
        # ###### Force the y-intercept to be shifted to the same point as the other one (to compare slopes)
        b_mean_2 = ((a_mean - a_mean_2)*logV0_data + b_mean)

        ##### Force the y-intercept to be the value we computed manually to be comparable to Bayesian
        # b_mean_2 = (-20.033 - a_mean_2*logV0_data)

        sigma = np.median(samples.T[:, -1])
        # cov_ab = np.cov(a_samples, b_samples)
        print(a_mean_2, b_mean_2)

        # ab_samples = np.random.multivariate_normal([a_mean, b_mean], cov_ab, size=1000)
        # Y = np.array([b + a * logx for a, b in ab_samples])
        # lower, median, upper = np.percentile(Y, [16, 50, 84], axis=0)

        ax.plot(x_vals, a_mean_2 * logx + b_mean_2, color=color, label=label, linestyle="--", linewidth=2)
        ax.plot(x_vals, a_mean_2 * logx + b_mean_2 - sigma, color=color, linestyle="dotted", linewidth=1)
        ax.plot(x_vals, a_mean_2 * logx + b_mean_2 + sigma, color=color, linestyle="dotted", linewidth=1)
        
        # ax.fill_between(x_vals, lower, upper, color=color, alpha=0.2)

    # ----------- Final Formatting -------------
    ax.set_xscale('log', base=10)
    ax.set_xlabel(r"$V_{rot}$", fontsize=18)
    ax.set_ylabel(r"$M_{abs}$", fontsize=18)
    ax.set_ylim(-23, -18)
    ax.set_xlim(30, 500)
    ax.invert_yaxis()
    ax.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined bowtie plot to {output_path}")



plot_bowties(
    file_json="data/iron_cluster_Tully_V0.json",
    # file_json='data/jura_cluster_full_V0.json',
    chain_dirs=[
        "output/iron/clustered",
        # "output/tully_full/V0"
               ],
    pickle_files=[
        '/global/homes/s/sgmoore1/DESI_SGA/TF/Y1/cov_ab_iron_jointTFR_varyV0-perpdwarfs0_z0p1_binaryMLupdated_Anthony2_weightsVmax-1_dVsys.pickle',
        # "/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/cov_ab_jura_jointTFR_varyV0_binaryML_weightsVmax-1_Tully_full_lessdwarves.pickle"
    ],
    labels_chains=[
        "Iron (Bayesian)",
        # "Tully Full (Bayesian)"
                  ],
    labels_pickles=[
        "Iron (Hyperfit)", 
        # "Tully Full (Hyperfit)"
                   ],
    colors_chains=[
        # "tab:blue",
        "tab:orange"
                  ],
    colors_pickles=[
        # "tab:blue", 
        "tab:blue"
                   ],
    output_path="/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/Figures/bowtie_comp_iron.png"
)
