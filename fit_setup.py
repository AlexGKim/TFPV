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


DATA_DIR = os.environ.get('DATA_DIR', '/Users/akim/Projects/TFPV/data_fit')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/Users/akim/Projects/TFPV/output_fit')
RELEASE_DIR = os.environ.get('RELEASE_DIR', 'Y1')

rng = numpy.random.default_rng(seed=42)


def main():


    dat = Table.read('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10.fits', format='fits')
    df = dat.to_pandas()
    Rlim = 17.75
    Mlim = -17.
    Vmin = 70
    Vmax = 350. # nothing this bright
    Rcut = numpy.minimum(Rlim, df['MU_ZCMB']+Mlim)
    w= (df['R_MAG_SB26'] < Rcut) & (df['V_0p4R26'] > Vmin) &  (df['V_0p4R26'] < Vmax)
    df = df[w]
    Rcut = Rcut[w]
    df = df[["V_0p4R26","V_0p4R26_ERR","R_MAG_SB26","R_MAG_SB26_ERR","MU_ZCMB"]]

    cov_ab, tfr_samples, logV0  = pandas.read_pickle('/Users/akim/Projects/TFPV/data/cluster_result_all.pickle')
    df_samples = pandas.DataFrame(data=numpy.array(tfr_samples).T,columns=["atanAR", "bR", "sigR", "xi_dist", "omega_dist",  "theta_2"])
    df_samples['theta_1'] = numpy.atan(df_samples['atanAR'])
    df_samples['logL0'] = df_samples['xi_dist']/numpy.cos(df_samples['theta_1'])
    df_prune=df_samples[["theta_1", "theta_2", "bR", "sigR", "logL0", "omega_dist"]]
    pop_mn = df_prune.mean()
    cov = df_prune.cov()
    pop_cov_L = numpy.linalg.cholesky(cov)

    data_dic=dict()
    for series_name, series in df.items():
        data_dic[series_name]=series.tolist()
    data_dic['N'] = len(df)
    data_dic['Rcut'] = Rcut.tolist()
    data_dic['Vmin'] = Vmin
    data_dic['Vmax'] = Vmax       
    data_dic['pop_mn'] = pop_mn.tolist()
    data_dic['pop_cov_L'] = pop_cov_L.tolist()
    data_dic['V0'] = 10**logV0 

    outname = os.path.join(DATA_DIR, RELEASE_DIR, "fit.json")

    json_object = json.dumps(data_dic)
    with open(outname, 'w+') as f:
        f.write(json_object)

    init_dic=dict()
    init_dic["mu"] = df["MU_ZCMB"].tolist()
    init_dic["logL"] = (numpy.log10(df["V_0p4R26"])-logV0).tolist()

    init_dic["theta_1"] = (numpy.zeros(len(df)) + pop_mn["theta_1"]).tolist()
    init_dic["theta_2"] = (numpy.zeros(len(df)) + pop_mn["theta_2"]).tolist()
    init_dic["b"] = (numpy.zeros(len(df)) + pop_mn["bR"]).tolist()
    init_dic["sigR"] = (numpy.zeros(len(df)) + pop_mn["sigR"]).tolist()
    init_dic["logL0"] = (numpy.zeros(len(df)) + pop_mn["logL0"]).tolist()
    init_dic["sigma_logL0"]= (numpy.zeros(len(df)) + pop_mn["omega_dist"]).tolist()

    outname = os.path.join(DATA_DIR, RELEASE_DIR, "fit_init.json")
    json_object = json.dumps(init_dic)
    with open(outname, 'w+') as f:
        f.write(json_object)


if __name__ == '__main__':
    main()
