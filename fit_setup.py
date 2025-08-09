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

    pvcat = os.path.join(DATA_DIR, RELEASE_DIR, 'DESI-DR1_TF_pv_cat_v10.fits')
    dat = Table.read(pvcat, format='fits')
    df = dat.to_pandas()

    Rlim = 17.75           # close to real magnitude limit
    Mlim = -17. + 5*numpy.log10(cosmo.h)           # absolute magnitude limit
    Vmin = 70.            # galaxies with lower velocities have different TF relation
    Vmax = 300.            # nothing faster in training set
        
    logVM_slope = .3      # there are missing high-velocity galaxies at low redshift
    logVM_zero =  34 + 5*numpy.log10(cosmo.h)

    Vlim_eff = numpy.minimum(Vmax, 10**(logVM_slope*numpy.array(df["MU_ZCMB"]-logVM_zero) + 2))
    Rlim_eff = numpy.minimum(Rlim, df['MU_ZCMB']+Mlim)
    w= (df['R_MAG_SB26'] < Rlim_eff) & (df['V_0p4R26'] > Vmin)  & (df["V_0p4R26"] <  Vlim_eff)
    df = df[w]
    Rlim_eff = Rlim_eff[w]
    Vlim_eff = Vlim_eff[w]
    outcat = os.path.join(DATA_DIR, RELEASE_DIR, 'DESI-DR1_TF_pv_cat_v10_cut.csv')
    df.to_csv(outcat) 
    df = df[["V_0p4R26","V_0p4R26_ERR","R_MAG_SB26","R_MAG_SB26_ERR","MU_ZCMB"]]


    fitres = os.path.join(DATA_DIR, RELEASE_DIR, 'cluster_result_all.pickle')
    cov_ab, tfr_samples, logV0  = pandas.read_pickle(fitres)
    df_prune = pandas.DataFrame(data=numpy.array(tfr_samples).T,columns=["aR", "bR", "sigR", "xi_dist", "omega_dist",  "theta_2"])
    df_prune["atanAR"] = numpy.arctan(df_prune["aR"])
    bR0 = df_prune["bR"].mean()

    # df_prune = df_prune[["atanAR","sigR", "xi_dist", "omega_dist",  "theta_2"]]
    df_prune = df_prune[["atanAR","sigR", "theta_2"]]

    pop_mn = df_prune.mean()


    cov = df_prune.cov()
    pop_cov_L = numpy.linalg.cholesky(cov)

    for i in range(len(df)):
        # df = df.iloc[[0]]
        # Rlim_eff = Rlim_eff[[0]]



        data_dic=dict()
        for series_name, series in df.iloc[[i]].items():
            data_dic[series_name]=series.tolist()
        data_dic['N'] = 1
        data_dic['Rlim_eff'] = Rlim_eff.iloc[[i]].tolist()
        data_dic['Vlim_eff'] = Vlim_eff[[i]].tolist()
        data_dic['Vmin'] = Vmin
        data_dic['Vmax'] = Vmax       
        data_dic['pop_mn'] = pop_mn.tolist()
        data_dic['pop_cov_L'] = pop_cov_L.tolist()
        data_dic['V0'] = 10**logV0 
        data_dic['bR0'] = bR0

        outname = os.path.join(DATA_DIR, RELEASE_DIR, "fit_{}.json".format(i))

        json_object = json.dumps(data_dic)
        with open(outname, 'w+') as f:
            f.write(json_object)

    init_dic=dict()

    init_dic["atanAR"] = (numpy.random.normal(pop_mn["atanAR"], 0.001, 1)).tolist()
    # init_dic["bR"] = (numpy.random.normal(pop_mn["bR"], 0.1, 1)).tolist()
    init_dic["sigR"] = pop_mn["sigR"].tolist()
    # init_dic["sigR"] = (numpy.random.normal(pop_mn["sigR"], 0.001, 1)).tolist()

    # init_dic["xi_dist"] = (numpy.random.normal(pop_mn["xi_dist"], 0.001, 1)).tolist()
    # init_dic["omega_dist"]= (numpy.random.normal(pop_mn["omega_dist"], 0.001, 1)).tolist()
    init_dic["theta_2"]= (numpy.random.normal(pop_mn["theta_2"], 0.001, 1)).tolist()

    outname = os.path.join(DATA_DIR, RELEASE_DIR, "fit_init.json")
    json_object = json.dumps(init_dic)
    with open(outname, 'w+') as f:
        f.write(json_object)


if __name__ == '__main__':
    main()
