'''
This file is Sara's modification of the fitstojson.py script. It primarily has the following changes:
- Accounts for Alex's outlier cuts determined in Fall 2025
- Stores data to use magnitudes after photometric corretions
- Adds option to fit intercept at x=mean(logV) instead of x=0
'''

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
from astropy.io import fits

# fn = "SGA-2020_iron_Vrot_cuts"
# fn = "DESI-DR1_TF_pv_cat_v15"

desi_sga_dir = "/global/homes/s/sgmoore1/DESI_SGA"

rng = numpy.random.default_rng(seed=42)


def jura_cluster_json(sn=False, V0=False, precorr=False, i=0):
    # fn = "SGA-2020_jura_Vrot_VI"
    # fn = "SGA-2020_iron_Vrot_VI"
    
    #### this is a subset from the most up-to date DR1 catalog as of January 2026
    fn = 'SGA-2020_iron_v15_cat_calib_subset'
    #####################

    ##### Parameter's for Alex's outlier cuts
    Rlim = 17.75           # close to real magnitude limit
    Mlim = -17.5           # absolute magnitude limit
    Vmin = 70            # galaxies with lower velocities have different TF relation
    Vmax = 300            # nothing faster in training set
        
    logVM_slope = 0.3      # there are missing high-velocity galaxies at low redshift
    # logVM_zero =  -9.9 + numpy.log10(70)

    # logVM_zero =  -7.9 + numpy.log10(cosmo.h)
    logVM_zero =  34 + 5*numpy.log10(cosmo.h)
    #####

    cosi = 1/numpy.sqrt(2)
    q0=0.2
    balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)


    # read in the cluster files
    N_per_cluster = []
    mu = []
    R2t = []
    Rlim_eff = []
    Vlim_eff = []

    alldf=[]

    ### instead of pulling in cluster by cluster, just read in all of the galaxies 
    # file = 'SGA-2020_iron_Vrot_VI_ML_photocorr'

    #### this is a subset from the most up-to date DR1 catalog as of January 2026
    file = 'SGA-2020_iron_v15_cat_calib_subset'
    #####################
    

    data = Table.read("pscratch/sd/s/sgmoore1/stan_inputs/" + file + ".fits")
    # data = Table.read("/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/Y3_subset_v3/" + file + ".fits")
    df = data.to_pandas()


    zval = df['Z_DESI'].astype(float).to_numpy()
    df['mu'] = cosmo.distmod(zval).value

    ##### Switch over to corrected magnitudes: (for now don't change names for downstream tasks)
    # df['R_MAG_SB26'] = df['R_MAG_SB26_CORR']
    # df['R_MAG_SB26_ERR'] = df['R_MAG_SB26_ERR_CORR']
    ######
    
    # if sn:
    #     table = Table.read("/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/SGA-2020_jura_Vrot_VI_0pt_calib_z0p1.fits")
    #     sn_df = table.to_pandas()
    #     sn_df['SGA_ID']=sn_df['SGA_ID'].astype(int)
    #     sn_df['is_sn'] = True
    #     mask = numpy.ma.getmaskarray(sn_df['MU_PRIMARY'])
    #     sn_df['mu'] = numpy.where(~mask, sn_df['MU_PRIMARY'], sn_df['MU_SECONDARY'])
    #     df = pandas.concat([df, sn_df], ignore_index=True)

    # Apply selection cuts
    Rcut = numpy.minimum(Rlim, df['mu'] + Mlim)
    Vcut = numpy.minimum(Vmax, 10**(logVM_slope*(df["mu"] - logVM_zero) + 2))
    select = (df['R_MAG_SB26'] < Rcut) & (df['V_0p4R26'] > Vmin) & (df['V_0p4R26'] < Vcut) & (df['BA'] < balim)
    df = df[select]

    nsn=0
    if sn: #update number of supernovae to be only however many made it past the selection cuts
        nsn = df['is_sn'].sum()
        
    Rlim_eff = numpy.minimum(Rlim, df["mu"] + Mlim).tolist()
    Vlim_eff = numpy.minimum(Vmax, 10**(logVM_slope*df["mu"] + logVM_zero)).tolist()

    N_per_cluster = [len(df)]
    N_cluster= 1
    mu = [cosmo.distmod(numpy.median(df['Z_DESI'])).value]
    df['R_ABSMAG_SB26'] = df['R_MAG_SB26'] - df['mu']
    df['V_0p4R26_err'] = df['V_0p4R26_ERR']

        
    # table = Table.read("/global/homes/s/sgmoore1/DESI_SGA/TF/Y3/SGA-2020_jura_Vrot_VI_0pt_calib_z0p1.fits")
    # df = table.to_pandas()
    # df['SGA_ID']=df['SGA_ID'].astype(int)
    # df.to_csv('temp.txt',columns=['SGA_ID'],index=False )
    # mu_sn=37.

    # for index, _ in df.iterrows():
    #     row=df.iloc[[index]]
    #     combo_df = row.merge(pv_df, on=['SGA_ID'],suffixes=["","y"]) #,'R_MAG_SB26', 'V_0p4R26','BA'])

    #     Rcut = numpy.minimum(Rlim, combo_df['MU_SECONDARY'].tolist()[0]+Mlim)
    #     # print((combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) ,(combo_df['R_MAG_SB26'] < Rcut))
    #     select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
    #     combo_df = combo_df[select]
    #     if combo_df.shape[0] > 0:
    #         nsn=nsn+1
    #         combo_df = combo_df[select]
    #         combo_df['R_MAG_SB26'] = combo_df['R_MAG_SB26']  - combo_df['MU_SECONDARY'] + mu_sn
    #         Rcut = Rcut  - combo_df['MU_SECONDARY'].tolist()[0] + mu_sn
    #         combo_df['R_MAG_SB26_ERR'] = numpy.sqrt(combo_df['R_MAG_SB26_ERR'] + combo_df['MU_ERR']**2) 
    #         Nest = df["SGA_ID"]
    #         _first = "{} & ".format(Nest)
    #         _second = "{} & ".format(mu_sn)
    #         # glue these together into a comma string
    #         dum = combo_df['SGA_ID'].tolist()
    #         # for i in range(len(dum)):
    #         #     dum[i] = str(dum[i])
    #         # my_string = ', '.join(dum)
    #         # print(_first + _second + my_string + ' \\\\')
    #         N_per_cluster.append(combo_df.shape[0])
    #         alldf.append(combo_df)

    #         mu.append(mu_sn)
    #         # R2t.append(0)
    #         Rlim_eff.append(Rcut);          


    # alldf = pandas.concat(alldf,ignore_index=True)
    # alldf = df[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR", "R_ABSMAG_SB26"]]
    alldf = df[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR", "R_ABSMAG_SB26", 'R_correction', 'R_correction_err']]

    # z = astropy.cosmology.z_at_value(cosmo.distmod, numpy.array(mu)*astropy.units.mag)
    # d = cosmo.luminosity_distance(z)
    logV0 = numpy.median(numpy.log10(alldf['V_0p4R26']))
    
    data_dic=dict()

    for series_name, series in alldf.items():
        data_dic[series_name]=series.tolist()

    N = len(df)
    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Rlim'] = Rlim
    data_dic['Mlim'] = Mlim
    data_dic['Vmin'] = Vmin
    data_dic['Vmax'] = Vmax

    data_dic["N_cluster"] = 1
    data_dic["N_per_cluster"] = [N]
    data_dic["N_sn"] = nsn
    data_dic["mu"] = df['mu'].tolist()
    data_dic["Rlim_eff"] = numpy.minimum(Rlim, df["mu"] + Mlim).tolist()

    data_dic["aR_init"]= -6.26
    data_dic["alpha_dist_init"]=1.25
    data_dic["xi_dist_init"]= 13.3 * numpy.cos(numpy.arctan(data_dic["aR_init"]))
    data_dic["omega_dist_init"]= .844   

    data_dic["mu_all"] = df['mu'].tolist()
    data_dic["logV0"] = logV0
    data_dic['Vlim_eff'] = Vlim_eff

    ###### This shifts the origin of the fit to be at the median logV to remove the correlation between a and b
    if precorr:
        data_dic['V0'] = numpy.median(alldf['V_0p4R26'])
        data_dic['Vmin'] = (Vmin/data_dic['V0'])
        data_dic['Vmax'] = (Vmax/data_dic['V0'])
        data_dic["xi_dist_init"] = 0.1
        data_dic['V_0p4R26_lognorm'] = (alldf['V_0p4R26']/data_dic['V0']).tolist()
        data_dic['V_0p4R26_lognorm_err'] = (alldf['V_0p4R26_err'] / data_dic['V0']).tolist()
        data_dic['Vlim_eff'] = (Vlim_eff/data_dic['V0']).tolist()

    json_object = json.dumps(data_dic)

    # outname = "iron_Y1_V0.json"
    # outname2 = "iron_init_Y1_V0.json"
    if precorr:
        # outname = f"Y1_full/iron_subsample_{i:02d}.json"
        # outname2 = f"Y1_full/iron_init_subsample_{i:02d}.json"
        outname = "iron_cluster_zbins_nocluster_v15_rcorr.json"
        outname2 = "iron_cluster_init_zbins_nocluster_v15_rcorr.json"
    # elif sn:
    #     outname = "jura_cluster_full_nocluster_SN.json"
    #     outname2 = "jura_cluster_init_full_nocluster_SN.json"
    # else:
    #     outname = "jura_cluster_full_nocluster.json"
    #     outname2 = "jura_cluster_init_full_nocluster.json"    
    else:
        outname = 'test_name.json'
        outname2 = 'test_name_init.json'

    with open("data/"+outname, 'w') as f:
        f.write(json_object)

#  vector[N] v = 373.137*v_raw + 222.371;
    init = dict()

    init["alpha_dist"]=data_dic["alpha_dist_init"]
    init["xi_dist"]= data_dic["xi_dist_init"]
    init["omega_dist"]=data_dic["omega_dist_init"]

    init["atanAR"] = numpy.arctan(data_dic["aR_init"])
    # init['bR'] = (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster)).tolist()
    init['sigR'] = 0.1 ## this value is arbitrary and can be changed, does not seem to change the fit value at all
    
    if precorr:
        logL = (numpy.log10(data_dic["V_0p4R26"]) - logV0)/numpy.cos(init["atanAR"])
    else:
        logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])

    init["logL_raw"]  = ((logL-init["xi_dist"]*numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()

    
    init["random_realization_raw"] = [0.0] * data_dic['N']

    bR_val = -6 - df['mu'].mean()
    init["bR"] = [bR_val] # we want as a 1D array for now
    if V0:
        init['bR'] = [(bR_val - data_dic["aR_init"] * data_dic['logV0'])]
    init["bR_offset"] =  0.0
    init["logV0"] = data_dic['logV0']
    
    # init["random_realization_raw"] = (numpy.zeros(data_dic['N'])).tolist()
    # init["bR_offset"]= (numpy.zeros(data_dic['N_cluster'])).tolist()
    with open("data/"+outname2, 'w') as f:
        f.write(json.dumps(init))




if __name__ == '__main__':
    jura_cluster_json(sn=False, V0=True, precorr=True)

    # for j in range(0,5):
    #     jura_cluster_json(sn=False, V0=True, precorr=True, i=j)
    # all_table()
    # iron_mag_plot()
    # for i in range(1,11):
    #     segev_json("data/SGA_TFR_simtest_{}".format(str(i).zfill(3)))
    # # segev_plot()
