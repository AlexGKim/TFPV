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


# def look():

def main():
    nrows=50

    mu=[]
    mu_std=[]
    mu_32=[]
    mu_50=[]
    mu_68=[]
    # for i in range(8716):
    for i in range(nrows):
        dfs = []
        for j in range(4):
            df = pandas.read_csv('/Users/akim/Projects/TFPV/output_fit/Y1/fit_{}_{}.csv'.format(i,j+1), comment='#')
            dfs.append(df)
        combined_df = pandas.concat(dfs, ignore_index=True)
        mu.append(combined_df['mu.1'].mean())
        mu_std.append(combined_df['mu.1'].std())
        _ = numpy.percentile(combined_df['mu.1'],[32,50,68])
        mu_32.append(_[0])
        mu_50.append(_[1])
        mu_68.append(_[2])
        print(i, mu_32[-1],mu_50[-1], mu_68[-1])

    df_master = pandas.read_csv('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10_cut.csv')
    df_master = df_master.head(nrows)
    df_master['MU_ALEX'] = mu
    df_master['MU_ALEX_ERR'] = mu_std
    df_master['MU_ALEX_32'] = mu_32
    df_master['MU_ALEX_50'] = mu_50
    df_master['MU_ALEX_68'] = mu_68
    df_master.to_pickle("/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10_cut.pkl")
    # dat = QTable.from_pandas(df_master)
    # dat.write('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10_cut.fits', format='fits', overwrite=True)

if __name__ == '__main__':
    main()
