import fitsio
import numpy
import json
import astropy
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt
import matplotlib
import glob
import pandas
from astropy.table import Table
import re
import os
from astropy.io import fits


# def look():

def main():
    # df_master = pandas.read_csv('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10_cut.csv')

    mu=[]
    mu_std=[]
    # for i in range(8716):
    for i in range(73):
        dfs = []
        for j in range(4):
            df = pandas.read_csv('/Users/akim/Projects/TFPV/output_fit/Y1/fit_{}_{}.csv'.format(i,j+1), comment='#')
            dfs.append(df)
        combined_df = pandas.concat(dfs, ignore_index=True)
        mu.append(combined_df['mu.1'].mean())
        mu_std.append(combined_df['mu.1'].std())
        print(mu[-1], mu_std[-1])

    # dat = Table.from_pandas(df)
    # dat.write('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10_cut.fits', format='fits')

if __name__ == '__main__':
    main()
