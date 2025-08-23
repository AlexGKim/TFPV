import fitsio
import numpy
import json
import astropy
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt
import matplotlib
import glob
import pandas
from astropy.table import QTable
import re
import os
from astropy.io import fits

DATA_DIR = os.environ.get('DATA_DIR', '/Users/akim/Projects/TFPV/data_fit')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/Users/akim/Projects/TFPV/output_fit')
RELEASE_DIR = os.environ.get('RELEASE_DIR', 'unclustered')

# def look():

def main():
    nrows=8309
    nrows=111

    mu=[]
    mu_std=[]
    mu_32=[]
    mu_50=[]
    mu_68=[]
    # V_TF = []
    # V_TF_std = []
    # V_TF_32=[]
    # V_TF_50=[]
    # V_TF_68=[]    

    # for i in range(8716):
    for i in range(nrows):
        dfs = []
        for j in range(4):
            infile = os.path.join(OUTPUT_DIR, RELEASE_DIR, 'fit_{}_{}.csv'.format(i,j+1))
            df = pandas.read_csv(infile, comment='#')
            dfs.append(df)
        combined_df = pandas.concat(dfs, ignore_index=True)
        mu.append(combined_df['mu.1'].mean())
        mu_std.append(combined_df['mu.1'].std())
        _ = numpy.percentile(combined_df['mu.1'],[32,50,68])
        mu_32.append(_[0])
        mu_50.append(_[1])
        mu_68.append(_[2])

        # V_TF.append(combined_df['V_TF.1'].mean())
        # V_TF_std.append(combined_df['V_TF.1'].std())        
        # _ = numpy.percentile(combined_df['V_TF.1'],[32,50,68])
        # V_TF_32.append(_[0])
        # V_TF_50.append(_[1])
        # V_TF_68.append(_[2])
        # print(i, mu_32[-1],mu_50[-1], mu_68[-1])


    master_file = os.path.join(DATA_DIR, RELEASE_DIR, "DESI-DR1_TF_pv_cat_v13_cut.csv")
    df_master = pandas.read_csv(master_file)
    df_master = df_master.head(nrows)
    df_master['MU_ALEX'] = mu
    df_master['MU_ALEX_ERR'] = mu_std
    df_master['MU_ALEX_32'] = mu_32
    df_master['MU_ALEX_50'] = mu_50
    df_master['MU_ALEX_68'] = mu_68
    # df_master['V_TF'] = V_TF
    # df_master['V_TF_ERR'] = V_TF_std
    # df_master['V_TF_32'] = V_TF_32
    # df_master['V_TF_50'] = V_TF_50
    # df_master['V_TF_68'] = V_TF_68
    out_file = os.path.join(DATA_DIR, RELEASE_DIR, "DESI-DR1_TF_pv_cat_v13_cut.pkl")
    df_master.to_pickle(out_file)
    # dat = QTable.from_pandas(df_master)
    # dat.write('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10_cut.fits', format='fits', overwrite=True)

if __name__ == '__main__':
    main()
