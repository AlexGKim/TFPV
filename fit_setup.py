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


# fn = "SGA-2020_iron_Vrot_cuts"
# fn = "DESI-DR1_TF_pv_cat_v3"
# fn_sga = "data/SGA-2020_fuji_Vrot"
# fn_segev2 = "SGA_TFR_simtest_20240307"

# desi_sga_dir = "/Users/akim/Projects/DESI_SGA/"

# desi_sga_dir = os.path.join(DATA_DIR, "DESI_SGA/")

rng = numpy.random.default_rng(seed=42)




def main():
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    DESI_SGA_DIR = os.environ.get('DESI_SGA_DIR', 'data/DESI_SGA')
    RELEASE_DIR = os.environ.get('RELEASE_DIR', 'Y1')
    OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')


    # The data

    dat = Table.read('/Users/akim/Projects/TFPV/data/DESI-DR1_TF_pv_cat_v10.fits', format='fits')
    df = dat.to_pandas()
    Rlim = 17.75
    Mlim = -17.
    Vmin = 70
    Vmax = 350. # nothing this bright

    w= (df['R_MAG_SB26'] < Rcut) & (df['V_0p4R26'] > Vmin) &  (df['V_0p4R26'] < Vmax)
    df = df[w]
    Rcut = numpy.minimum(Rlim, df['MU_ZCMB']+Mlim)
    df = df[["V_0p4R26","V_0p4R26_ERR","R_MAG_SB26","R_MAG_SB26_ERR"]]

    cov_ab, tfr_samples, logV0  = pandas.read_pickle('/Users/akim/Projects/TFPV/data/cluster_result_all.pickle')
    df_samples = pandas.DataFrame(data=numpy.array(tfr_samples).T,columns=["atanAR", "bR", "sigR", "xi_dist", "omega_dist",  "theta_2"])
    df_samples['theta_1'] = numpy.atan(df_samples['atanAR'])
    df_samples['logL0'] = df_samples['xi_dist']/numpy.cos(df_samples['theta_1'])
    df_prune=df_samples[["theta_1", "theta_2", "bR", "sigR", "logL0", "xi_dist", "omega_dist"]]
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



    outname = os.path.join(OUTPUT_DIR, RELEASE_DIR, "fit.json")
    outname2 = os.path.join(OUTPUT_DIR, RELEASE_DIR, "fit_init.json")
    if not os.path.exists(os.path.join(OUTPUT_DIR, RELEASE_DIR)):
        os.makedirs(os.path.join(OUTPUT_DIR, RELEASE_DIR))






#     cosi = 1/numpy.sqrt(2)
#     q0=0.2
#     balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)

#     table = Table.read(fn)
#     pv_df = table.to_pandas()

#     table = Table.read(os.path.join(DESI_SGA_DIR,"TF","Tully15-Table3.fits"))
#     tully_df = table.to_pandas()

#     # # add extra noise degrading data to help fit
#     # dt = {'names':['Vhat','Vhat_noise','Rhat'], 'formats':[float, float,float]}
#     # extradata = numpy.zeros(len(data['Z_DESI']),dtype=dt)
#     # extradata['Vhat_noise'] = 0.02*data["V_0p4R26"]
#     # Rhat_noise = 0.1
#     # extradata['Vhat'] = numpy.random.normal(loc=data["V_0p4R26"], scale=extradata['Vhat_noise'])
#     # extradata['Rhat'] = numpy.random.normal(loc=data['R_MAG_SB26'], scale=Rhat_noise)

#     # read in the cluster files
#     N_per_cluster = []
#     mu = []
#     R2t = []
#     Rlim_eff = []

#     alldf=[]
#     file = open(os.path.join(OUTPUT_DIR, RELEASE_DIR, "cluster_tex.txt"), "w+")

#    # selection effects
#     for fn in glob.glob(os.path.join(DESI_SGA_DIR,"TF",RELEASE_DIR,"output_*.txt")):
#         if "output_sn.txt" in fn:
#             continue
#         Nest = re.search('output_(.+?).txt',fn).group(1)  # number of the galaxy
#         mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
#         R2t_=tully_df.loc[tully_df["Nest"]==int(Nest)]["R2t"].values[0]

#         df = pandas.read_csv(fn)
#         df = df.rename(columns={'# SGA_ID': 'SGA_ID'})
#         combo_df = df.merge(pv_df, on='SGA_ID')
#         Rcut = numpy.minimum(Rlim, mu_+Mlim)
#         select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
#         combo_df = combo_df[select]
#         if combo_df.shape[0] > 1:
#             _first = "{} & ".format(Nest)
#             _second = "{} & ".format(mu_)
#             # glue these together into a comma string
#             dum = combo_df['SGA_ID'].tolist()
#             for i in range(len(dum)):
#                 dum[i] = str(dum[i])
#             my_string = ', '.join(dum)
#             print(_first + _second + my_string + ' \\\\', file=file)
#             N_per_cluster.append(combo_df.shape[0])
#             alldf.append(combo_df)
#             Nest = re.search('output_(.+?).txt',fn).group(1)
#             mu.append(mu_)
#             # R2t.append(R2t_)
#             Rlim_eff.append(Rcut);

#     # if there are supernovae out them into data as well
#     nsn=0
#     include_sn = False
#     if include_sn:
#         table = Table.read(os.path.join(DATA_DIR, RELEASE_DIR, "SGA-2020_iron_Vrot_VI_0pt_calib_z0p1.fits"))
#         df = table.to_pandas()
#         df['SGA_ID']=df['SGA_ID'].astype(int)
#         df.to_csv('temp.txt',columns=['SGA_ID'],index=False )
#         mu_sn=37.

#         for index, _ in df.iterrows():
#             row=df.iloc[[index]]
#             combo_df = row.merge(pv_df, on=['SGA_ID'],suffixes=["","y"]) #,'R_MAG_SB26', 'V_0p4R26','BA'])

#             Rcut = numpy.minimum(Rlim, combo_df['MU_SECONDARY'].tolist()[0]+Mlim)
#             # print((combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) ,(combo_df['R_MAG_SB26'] < Rcut))
#             select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
#             combo_df = combo_df[select]
#             if combo_df.shape[0] > 0:
#                 nsn=nsn+1
#                 combo_df = combo_df[select]
#                 combo_df['R_MAG_SB26'] = combo_df['R_MAG_SB26']  - combo_df['MU_SECONDARY'] + mu_sn
#                 Rcut = Rcut  - combo_df['MU_SECONDARY'].tolist()[0] + mu_sn
#                 combo_df['R_MAG_SB26_ERR'] = numpy.sqrt(combo_df['R_MAG_SB26_ERR'] + combo_df['MU_ERR']**2) 
#                 Nest = df["SGA_ID"]
#                 _first = "{} & ".format(Nest)
#                 _second = "{} & ".format(mu_sn)
#                 # glue these together into a comma string
#                 dum = combo_df['SGA_ID'].tolist()
#                 # for i in range(len(dum)):
#                 #     dum[i] = str(dum[i])
#                 # my_string = ', '.join(dum)
#                 # print(_first + _second + my_string + ' \\\\')
#                 N_per_cluster.append(combo_df.shape[0])
#                 alldf.append(combo_df)

#                 mu.append(mu_sn)
#                 # R2t.append(0)
#                 Rlim_eff.append(Rcut);      

#     N_cluster=len(alldf)


#     alldf = pandas.concat(alldf,ignore_index=True)
#     alldf = alldf[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]

#     z = astropy.cosmology.z_at_value(cosmo.distmod, numpy.array(mu)*astropy.units.mag)
#     d = cosmo.luminosity_distance(z)


#     data_dic=dict()

#     for series_name, series in alldf.items():
#         data_dic[series_name]=series.tolist()

#     data_dic['N'] = len(data_dic['SGA_ID'])
#     data_dic['Rlim'] = Rlim
#     data_dic['Mlim'] = Mlim
#     data_dic['Vmin'] = Vmin
#     data_dic['Vmax'] = Vmax

#     data_dic["N_cluster"] = N_cluster
#     data_dic["N_per_cluster"] = N_per_cluster
#     data_dic["N_sn"] = nsn
#     data_dic["mu"] = mu
#     data_dic["Rlim_eff"] = Rlim_eff

#     data_dic["aR_init"]= -6.26
#     data_dic["alpha_dist_init"]=1.25
#     data_dic["xi_dist_init"]= 13.3 * numpy.cos(numpy.arctan(data_dic["aR_init"]))
#     data_dic["omega_dist_init"]= .844   

#     dum=[]
#     for npc,m in zip(N_per_cluster,mu):
#         for j in range(npc):
#             dum.append(m)
#     data_dic["mu_all"]=dum

#     json_object = json.dumps(data_dic)


#     with open(outname, 'w+') as f:
#         f.write(json_object)

# #  vector[N] v = 373.137*v_raw + 222.371;
#     init = dict()

#     init["alpha_dist"]=data_dic["alpha_dist_init"]
#     init["xi_dist"]= data_dic["xi_dist_init"]
#     init["omega_dist"]=data_dic["omega_dist_init"]

#     init["atanAR"] = numpy.arctan(data_dic["aR_init"])
#     init['bR'] = (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster)).tolist()
#     init['sigR'] = 0.1
#     logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])


#     init["logL_raw"]  = ((logL-init["xi_dist"]*numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()

#     init["random_realization_raw"] = (numpy.zeros(data_dic['N'])).tolist()
#     init["bR_offset"]= (numpy.zeros(data_dic['N_cluster'])).tolist()
#     with open(outname2, 'w+') as f:
#         f.write(json.dumps(init))





if __name__ == '__main__':
    main()
