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

cluster_cuts ={
    'Rlim': 17.75,
    'Vmin': 70,
    'Vmax': 300.,
    'Mlim': -18,
    'Vmax': 1e4,
    'cosi': 1/numpy.sqrt(2),
    'q0' : 0.2,
    'balim': numpy.sqrt((1/numpy.sqrt(2))**2 * (1-0.2**2) + 0.2**2)
    }

# fn = "SGA-2020_iron_Vrot_cuts"
fn = "DESI-DR1_TF_pv_cat_v3"

fn_sga = "data/SGA-2020_fuji_Vrot"
fn_segev2 = "SGA_TFR_simtest_20240307"

desi_sga_dir = "/Users/akim/Projects/DESI_SGA/"

rng = numpy.random.default_rng(seed=42)

def coma_json(cuts=False):
    fits=fitsio.FITS(fn_sga+".fits")
    data=fits[1].read()

    cstr=""
    if cuts:
        cstr="_cuts"
    outname = fn_sga+cstr+".json"
    outname2 = fn_sga+cstr+"_init.json"

    Vmin = 50


    # comalist = [8032,20886,25532,98934,100987,122260,127141,128944,139660,171794,191275,191496,192582,196592,202666,221178,238344,289665,291879,301194,302524,309306,330166,337817,343570,364410,364929,365429,366393,378180,378842,381769,390630,455486,465951,477610,479267,486394,540744,556334,566771,573264,629860,637552,645151,652931,665961,729931,733069,735080,747077,748600,753474,796671,811359,819754,824392,826543,827339,834049,837120,841705,900049,905270,908303,917608,918100,928810,972260,993595,995924,1009928,1014365,1020852,1050173,1089288,1115705,1122082,1144453,1167691,1195008,1198552,1201916,1203610,1203786,1204237,1206707,1209774,1269260,1272144,1274171,1274189,1274409,1281982,1284002,1293940,1294562,1323268,1349168,1356626,1364394,1379275,1387126,1387991]
    comalist = [25532,30149,98934,122260,191275,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]
    # comalist = [25532,30149,98934,122260,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]

    if cuts:
        select = numpy.logical_and.reduce((numpy.isin(data['SGA_ID'],comalist) , data['V_0p33R26'] > Vmin))
    else:
        select = numpy.isin(data['SGA_ID'],comalist)

    print(len(comalist), select.sum())
    qqwd
    data_dic=dict()
    for k in data.dtype.names:
        if k not in ['SGA_GALAXY','GALAXY','MORPHTYPE','BYHAND','REF','GROUP_NAME','GROUP_PRIMARY','BRICKNAME','D26_REF']:
            if k in ['G_MAG_SB26_ERR','R_MAG_SB26_ERR','Z_MAG_SB26_ERR']:
                w=numpy.where(data[k]<0)
                data[k][w[0]]=1e8

            data_dic[k]=data[k][select].tolist()


    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Vmin'] = Vmin

    json_object = json.dumps(data_dic)

    with open(outname, 'w') as f:
        f.write(json_object)

    init = dict()

    init["atanAR"] = numpy.arctan(-6.1)
    logL = numpy.log10(data_dic["V_0p33R26"])/numpy.cos(init["atanAR"])

    init["alpha_dist"]=-3.661245022462153
    init["xi_dist"]= 14.913405242237685  * numpy.cos(init["atanAR"])
    init["omega_dist"]=2.2831016215521247
    init['bR'] =  init["xi_dist"] * numpy.tan(init["atanAR"]) 
    init["logL_raw"]  = ((logL-init["xi_dist"]/ numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()
    init['epsilon_unif'] =  numpy.zeros(data_dic['N']).tolist()
    with open(outname2, 'w') as f:
        f.write(json.dumps(init))



def iron_cluster_json(cepheid=False):

    fn_str = ""
    if cepheid:
        fn_str="_clustercepheid"

    fn = "SGA-2020_iron_Vrot"

    Rlim = 17.75
    Vmin = 70
    # Vmax = 300. # nothing this bright

    Mlim = -18
    Vmax = 1e4

    cosi = 1/numpy.sqrt(2)
    q0=0.2
    balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)

    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()

    table = Table.read(desi_sga_dir+"/TF/Tully15-Table3.fits")
    tully_df = table.to_pandas()

    # # add extra noise degrading data to help fit
    # dt = {'names':['Vhat','Vhat_noise','Rhat'], 'formats':[float, float,float]}
    # extradata = numpy.zeros(len(data['Z_DESI']),dtype=dt)
    # extradata['Vhat_noise'] = 0.02*data["V_0p4R26"]
    # Rhat_noise = 0.1
    # extradata['Vhat'] = numpy.random.normal(loc=data["V_0p4R26"], scale=extradata['Vhat_noise'])
    # extradata['Rhat'] = numpy.random.normal(loc=data['R_MAG_SB26'], scale=Rhat_noise)

    # read in the cluster files
    N_per_cluster = []
    mu = []
    R2t = []
    Rlim_eff = []

    alldf=[]

    # remove the file that will be created later
    os.remove(desi_sga_dir+'/TF/Y1/output_sn.txt')

   # selection effects
    for fn in glob.glob(desi_sga_dir+"/TF/Y1/output_*.txt"):
        Nest = re.search('output_(.+?).txt',fn).group(1)  # number of the galaxy
        mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
        R2t_=tully_df.loc[tully_df["Nest"]==int(Nest)]["R2t"].values[0]

        df = pandas.read_csv(fn)
        df = df.rename(columns={'# SGA_ID': 'SGA_ID'})
        combo_df = df.merge(pv_df, on='SGA_ID')
        Rcut = numpy.minimum(Rlim, mu_+Mlim)
        select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
        combo_df = combo_df[select]
        if combo_df.shape[0] > 1:
            _first = "{} & ".format(Nest)
            _second = "{} & ".format(mu_)
            # glue these together into a comma string
            dum = combo_df['SGA_ID'].tolist()
            for i in range(len(dum)):
                dum[i] = str(dum[i])
            my_string = ', '.join(dum)
            # print(_first + _second + my_string + ' \\\\')
            N_per_cluster.append(combo_df.shape[0])
            alldf.append(combo_df)
            Nest = re.search('output_(.+?).txt',fn).group(1)
            mu.append(mu_)
            # R2t.append(R2t_)
            Rlim_eff.append(Rcut);

    # if there are supernovae out them into data as well
    nsn=0
    if cepheid:
        table = Table.read("data/SGA-2020_iron_Vrot_VI_0pt_calib_z0p1.fits")
        df = table.to_pandas()
        df['SGA_ID']=df['SGA_ID'].astype(int)
        dumdf = df.copy()
        dumdf.rename(columns={"SGA_ID": "# SGA_ID"},inplace=True)
        dumdf.to_csv(desi_sga_dir+'/TF/Y1/output_sn.txt',columns=['# SGA_ID'],index=False )
        mu_sn=37.

        for index, _ in df.iterrows():
            row=df.iloc[[index]]
            combo_df = row.merge(pv_df, on=['SGA_ID'],suffixes=["","y"]) #,'R_MAG_SB26', 'V_0p4R26','BA'])

            Rcut = numpy.minimum(Rlim, combo_df['MU_SECONDARY'].tolist()[0]+Mlim)
            # print((combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) ,(combo_df['R_MAG_SB26'] < Rcut))
            select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
            combo_df = combo_df[select]
            if combo_df.shape[0] > 0:
                nsn=nsn+1
                combo_df = combo_df[select]
                combo_df['R_MAG_SB26'] = combo_df['R_MAG_SB26']  - combo_df['MU_SECONDARY'] + mu_sn
                Rcut = Rcut  - combo_df['MU_SECONDARY'].tolist()[0] + mu_sn
                combo_df['R_MAG_SB26_ERR'] = numpy.sqrt(combo_df['R_MAG_SB26_ERR'] + combo_df['MU_ERR']**2) 
                Nest = df["SGA_ID"]
                _first = "{} & ".format(Nest)
                _second = "{} & ".format(mu_sn)
                # glue these together into a comma string
                dum = combo_df['SGA_ID'].tolist()
                # for i in range(len(dum)):
                #     dum[i] = str(dum[i])
                # my_string = ', '.join(dum)
                # print(_first + _second + my_string + ' \\\\')
                N_per_cluster.append(combo_df.shape[0])
                alldf.append(combo_df)

                mu.append(mu_sn)
                # R2t.append(0)
                Rlim_eff.append(Rcut);      

    N_cluster=len(alldf)


    alldf = pandas.concat(alldf,ignore_index=True)
    alldf = alldf[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]

    z = astropy.cosmology.z_at_value(cosmo.distmod, numpy.array(mu)*astropy.units.mag)
    d = cosmo.luminosity_distance(z)


    data_dic=dict()

    for series_name, series in alldf.items():
        data_dic[series_name]=series.tolist()

    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Rlim'] = Rlim
    data_dic['Mlim'] = Mlim
    data_dic['Vmin'] = Vmin
    data_dic['Vmax'] = Vmax

    data_dic["N_cluster"] = N_cluster
    data_dic["N_per_cluster"] = N_per_cluster
    data_dic["N_sn"] = nsn
    data_dic["mu"] = mu
    data_dic["Rlim_eff"] = Rlim_eff

    data_dic["aR_init"]= -6.26
    data_dic["alpha_dist_init"]=1.25
    data_dic["xi_dist_init"]= 13.3 * numpy.cos(numpy.arctan(data_dic["aR_init"]))
    data_dic["omega_dist_init"]= .844   

    dum=[]
    for npc,m in zip(N_per_cluster,mu):
        for j in range(npc):
            dum.append(m)
    data_dic["mu_all"]=dum

    json_object = json.dumps(data_dic)

    outname = "iron_cluster"+fn_str+".json"
    outname2 = "iron_cluster"+fn_str+"_init.json"

    with open("data/"+outname, 'w') as f:
        f.write(json_object)

#  vector[N] v = 373.137*v_raw + 222.371;
    init = dict()

    init["alpha_dist"]=data_dic["alpha_dist_init"]
    init["xi_dist"]= data_dic["xi_dist_init"]
    init["omega_dist"]=data_dic["omega_dist_init"]

    init["atanAR"] = numpy.arctan(data_dic["aR_init"])
    init['bR'] = (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster)).tolist()
    init['bR_sn'] = (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster-nsn+1)).tolist()

    init['sigR'] = 0.1
    logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])




    init["logL_raw"]  = ((logL-init["xi_dist"]*numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()

    init["random_realization_raw"] = (numpy.zeros(data_dic['N'])).tolist()
    init["bR_offset"]= (numpy.zeros(data_dic['N_cluster'])).tolist()
    with open("data/"+outname2, 'w') as f:
        f.write(json.dumps(init))

def iron_cepheid_json():
    fn = "SGA-2020_iron_Vrot"

    Rlim = 17.75
    Vmin = 70
    # Vmax = 300. # nothing this bright

    Mlim = -18
    Vmax = 1e4

    cosi = 1/numpy.sqrt(2)
    q0=0.2
    balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)

    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()

    table = Table.read(desi_sga_dir+"/TF/Tully15-Table3.fits")
    tully_df = table.to_pandas()

    # # add extra noise degrading data to help fit
    # dt = {'names':['Vhat','Vhat_noise','Rhat'], 'formats':[float, float,float]}
    # extradata = numpy.zeros(len(data['Z_DESI']),dtype=dt)
    # extradata['Vhat_noise'] = 0.02*data["V_0p4R26"]
    # Rhat_noise = 0.1
    # extradata['Vhat'] = numpy.random.normal(loc=data["V_0p4R26"], scale=extradata['Vhat_noise'])
    # extradata['Rhat'] = numpy.random.normal(loc=data['R_MAG_SB26'], scale=Rhat_noise)

    # read in the cluster files
    N_per_cluster = []
    mu = []
    R2t = []
    Rlim_eff = []

    alldf=[]

    # remove the file that will be created later
    os.remove(desi_sga_dir+'/TF/Y1/output_sn.txt')
   # selection effects
    # for fn in glob.glob(desi_sga_dir+"/TF/Y1/output_*.txt"):
    #     Nest = re.search('output_(.+?).txt',fn).group(1)  # number of the galaxy
    #     mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
    #     R2t_=tully_df.loc[tully_df["Nest"]==int(Nest)]["R2t"].values[0]

    #     df = pandas.read_csv(fn)
    #     df = df.rename(columns={'# SGA_ID': 'SGA_ID'})
    #     combo_df = df.merge(pv_df, on='SGA_ID')
    #     Rcut = numpy.minimum(Rlim, mu_+Mlim)
    #     select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
    #     combo_df = combo_df[select]
    #     if combo_df.shape[0] > 1:
    #         _first = "{} & ".format(Nest)
    #         _second = "{} & ".format(mu_)
    #         # glue these together into a comma string
    #         dum = combo_df['SGA_ID'].tolist()
    #         for i in range(len(dum)):
    #             dum[i] = str(dum[i])
    #         my_string = ', '.join(dum)
    #         print(_first + _second + my_string + ' \\\\')
    #         N_per_cluster.append(combo_df.shape[0])
    #         alldf.append(combo_df)
    #         Nest = re.search('output_(.+?).txt',fn).group(1)
    #         mu.append(mu_)
    #         # R2t.append(R2t_)
    #         Rlim_eff.append(Rcut);

    # if there are supernovae out them into data as well
    nsn=0
    table = Table.read("data/SGA-2020_iron_Vrot_VI_0pt_calib_z0p1.fits")
    df = table.to_pandas()
    df['SGA_ID']=df['SGA_ID'].astype(int)
    dumdf = df.copy()
    dumdf.rename(columns={"SGA_ID": "# SGA_ID"},inplace=True)
    dumdf.to_csv(desi_sga_dir+'/TF/Y1/output_sn.txt',columns=['# SGA_ID'],index=False )
    mu_sn=37.

    for index, _ in df.iterrows():
        row=df.iloc[[index]]
        combo_df = row.merge(pv_df, on=['SGA_ID'],suffixes=["","y"]) #,'R_MAG_SB26', 'V_0p4R26','BA'])

        Rcut = numpy.minimum(Rlim, combo_df['MU_SECONDARY'].tolist()[0]+Mlim)
        # print((combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) ,(combo_df['R_MAG_SB26'] < Rcut))
        select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
        combo_df = combo_df[select]
        if combo_df.shape[0] > 0:
            nsn=nsn+1
            combo_df = combo_df[select]
            combo_df['R_MAG_SB26'] = combo_df['R_MAG_SB26']  - combo_df['MU_SECONDARY'] + mu_sn
            Rcut = Rcut  - combo_df['MU_SECONDARY'].tolist()[0] + mu_sn
            combo_df['R_MAG_SB26_ERR'] = numpy.sqrt(combo_df['R_MAG_SB26_ERR'] + combo_df['MU_ERR']**2) 
            Nest = df["SGA_ID"]
            _first = "{} & ".format(Nest)
            _second = "{} & ".format(mu_sn)
            # glue these together into a comma string
            dum = combo_df['SGA_ID'].tolist()
            # for i in range(len(dum)):
            #     dum[i] = str(dum[i])
            # my_string = ', '.join(dum)
            # print(_first + _second + my_string + ' \\\\')
            N_per_cluster.append(combo_df.shape[0])
            alldf.append(combo_df)

            mu.append(mu_sn)
            # R2t.append(0)
            Rlim_eff.append(Rcut);      

    N_cluster=len(alldf)


    alldf = pandas.concat(alldf,ignore_index=True)
    alldf = alldf[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]

    z = astropy.cosmology.z_at_value(cosmo.distmod, numpy.array(mu)*astropy.units.mag)
    d = cosmo.luminosity_distance(z)


    data_dic=dict()

    for series_name, series in alldf.items():
        data_dic[series_name]=series.tolist()

    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Rlim'] = Rlim
    data_dic['Mlim'] = Mlim
    data_dic['Vmin'] = Vmin
    data_dic['Vmax'] = Vmax

    data_dic["N_cluster"] = N_cluster
    data_dic["N_per_cluster"] = N_per_cluster
    data_dic["N_sn"] = nsn
    data_dic["mu"] = mu
    data_dic["Rlim_eff"] = Rlim_eff

    data_dic["aR_init"]= -6.26
    data_dic["alpha_dist_init"]=1.25
    data_dic["xi_dist_init"]= 13.3 * numpy.cos(numpy.arctan(data_dic["aR_init"]))
    data_dic["omega_dist_init"]= .844   

    dum=[]
    for npc,m in zip(N_per_cluster,mu):
        for j in range(npc):
            dum.append(m)
    data_dic["mu_all"]=dum

    json_object = json.dumps(data_dic)

    outname = "_iron_cepheid.json"
    outname2 = "_iron_cepheid_init.json"

    with open("data/"+outname, 'w') as f:
        f.write(json_object)

#  vector[N] v = 373.137*v_raw + 222.371;
    init = dict()

    init["alpha_dist"]=data_dic["alpha_dist_init"]
    init["xi_dist"]= data_dic["xi_dist_init"]
    init["omega_dist"]=data_dic["omega_dist_init"]

    init["atanAR"] = numpy.arctan(data_dic["aR_init"])
    init['bR'] = (init["xi_dist"] * numpy.tan(init["atanAR"]))
    # init['bR_sn'] = (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster-nsn+1)).tolist()

    init['sigR'] = 0.1
    logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])




    init["logL_raw"]  = ((logL-init["xi_dist"]*numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()

    init["random_realization_raw"] = (numpy.zeros(data_dic['N'])).tolist()
    init["bR_offset"]= (numpy.zeros(data_dic['N_cluster'])).tolist()
    with open("data/"+outname2, 'w') as f:
        f.write(json.dumps(init))



def segev_json(fn='SGA_TFR_simtest_20240307'):



    fits=fitsio.FITS(fn+".fits")
    data=fits[1].read()


    data_dic=dict()
    for k in data.dtype.names:
        data_dic[k]=data[k].tolist()

    data_dic['N'] = len(data_dic['R_MAG_SB26'])


    json_object = json.dumps(data_dic)

    with open(fn+".json", 'w') as f:
        f.write(json_object)


    init = dict()
    # init["v_raw"]=(numpy.array(data_dic["V_0p33R26"])/139.35728557650154).tolist()

    # (-3.661245022462153, 14.913405242237685, 2.2831016215521247)
    init["atanAR"] = numpy.arctan(-6.1)
    logL = numpy.log10(data_dic["V_0p33R26"])/numpy.cos(init["atanAR"])
    init["logL"]  = logL.tolist()
    init["s_dist"]=0.5326792343583239
    init["scale_dist"]=139.35728557650154

    init["alpha_dist"]=-3.661245022462153
    init["xi_dist"]= 14.913405242237685
    init["omega_dist"]=2.2831016215521247

    init["mu_dist"]=13.133570672711606
    init["sigma_dist"]= 1.5160651053079683
    init["logL_raw"]  = ((logL-init["xi_dist"])/init["omega_dist"]).tolist()

    # init["r_raw"]=((numpy.array(data_dic["V_0p33R26"])+24.483252891972377)/3.8906505354308463).tolist()
    # init["r_s_dist"]=0.3203771381830672
    # init["r_offset_dist"]=-24.483252891972377
    # init["r_scale_dist"]=3.8906505354308463
    with open(fn+"_init.json", 'w') as f:
        f.write(json.dumps(init))

def plot():
    fits=fitsio.FITS(fn+".fits")
    data=fits[1].read()
    plt.plot(data['Z_DESI'],data['R_MAG_SB26'],'.')
    plt.xlabel('Z_DESI')
    plt.ylabel('R_MAG_SB26')
    plt.show()

def segev_plot(fn = fn_segev2):
    fits=fitsio.FITS(fn+".fits")
    data=fits[1].read()
    plt.errorbar(numpy.log10(data['V_0p33R26']),data['R_MAG_SB26'],yerr=data['R_MAG_SB26_ERR'], xerr=data['V_0p33R26_err']/data['V_0p33R26']/numpy.log(10),fmt='.')
    plt.xlabel('V_0p33R26')
    plt.ylabel('R_MAG_SB26')
    plt.ylim((19,12))
    plt.show()

def iron_mag_plot():

    # the cluster set
    pv_df, alldf, _, _, _ = cluster_set()
    # the coma set
    c_df = coma_set()

    plt.scatter(pv_df["Z_DESI"],pv_df["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Iron",color='black')
    plt.axhline(17.75,color='red')
    plt.xlabel('Z_DESI')
    plt.ylabel('R_MAG_SB26')

    plt.scatter(alldf["Z_DESI"],alldf["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Iron Cluster",color="orange")


    plt.scatter(c_df["Z_DESI"],c_df["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Coma",color="cyan")

    plt.legend()
    plt.savefig("zR.png")
    plt.show()


def iron_to_sn():

    inname = "iron_cluster.json"
    inname2 = "iron_cluster_init.json"

    outname = "sn.json"
    outname2 = "sn_init.json"
    with open('data/'+inname, 'r') as file:
        data = json.load(file)
    with open('data/'+inname2, 'r') as file:
        data_init = json.load(file)

    for k in data.keys():
        print(k,)
        print(numpy.array(data[k]).shape)

    outdata = dict(data)
    outdata_init = dict(data_init)


def all_table():

    fn = "SGA-2020_iron_Vrot"
    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()
    cols = ["SGA_ID", "Z_DESI","V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR",'BA']
    pv_df = pv_df[cols]

    for index, row in pv_df.iterrows():
        if index<20:
            print("{:10.0f} & {:5.3f} & ${:6.1f} \\pm {:6.1f}$ & ${:6.2f} \\pm {:6.2f}$ & {:6.3f} \\\\".format(*row.to_numpy()))

def coma_set():
    table2 = Table.read(fn_sga+".fits")
    c_df = table2.to_pandas()

    Vmin = 50

    comalist = [25532,30149,98934,122260,191275,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]
    select = numpy.logical_and.reduce((numpy.isin(c_df['SGA_ID'],comalist) , c_df['V_0p33R26'] > Vmin))
    return c_df[select]

def cluster_set():

    Rlim = cluster_cuts["Rlim"] 
    Vmin = cluster_cuts["Vmin"] 

    Mlim = cluster_cuts["Mlim"] 
    Vmax = cluster_cuts["Vmax"] 

    balim = cluster_cuts["balim"]

    fn = "DESI-DR1_TF_pv_cat_v3"
    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()

    table = Table.read("data/Tully15-Table3.fits")
    tully_df = table.to_pandas()
    # read in the cluster files
    N_per_cluster = []
    mu = []
    Rlim_eff = []

    alldf=[]
   # selection effects
    for fn in glob.glob(desi_sga_dir+"/TF/Y1/output_*.txt"):
        Nest = re.search('output_(.+?).txt',fn).group(1)
        if Nest != 'sn':
            mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
            df = pandas.read_csv(fn)
            df = df.rename(columns={'# SGA_ID': 'SGA_ID'})
            combo_df = df.merge(pv_df, on='SGA_ID')
            Rcut = numpy.minimum(Rlim, mu_+Mlim)
            select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
            combo_df = combo_df[select]
            if combo_df.shape[0] > 1:
                N_per_cluster.append(combo_df.shape[0])
                alldf.append(combo_df)
                Nest = re.search('output_(.+?).txt',fn).group(1)
                # mu.append(mu_)
                Rlim_eff.append(Rcut)

    alldf = pandas.concat(alldf,ignore_index=True)
            
    return pv_df, alldf, N_per_cluster, mu, Rlim_eff

def cepheid_set():

    Rlim = cluster_cuts["Rlim"] 
    Vmin = cluster_cuts["Vmin"] 

    Mlim = cluster_cuts["Mlim"] 
    Vmax = cluster_cuts["Vmax"] 

    # cosi = 1/numpy.sqrt(2)
    # q0=0.2
    balim = cluster_cuts["balim"]

    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()

    table = Table.read(desi_sga_dir+"/TF/Tully15-Table3.fits")
    tully_df = table.to_pandas()


    # read in the cluster files
    N_per_cluster = []
    mu = []
    R2t = []
    Rlim_eff = []

    alldf=[]

    # remove the file that will be created later
    if os.path.exists(desi_sga_dir+'/TF/Y1/output_sn.txt'):
        os.remove(desi_sga_dir+'/TF/Y1/output_sn.txt')

    # if there are supernovae out them into data as well
    nsn=0
    table = Table.read("data/SGA-2020_iron_Vrot_VI_0pt_calib_z0p1.fits")
    df = table.to_pandas()
    df['SGA_ID']=df['SGA_ID'].astype(int)
    dumdf = df.copy()
    dumdf.rename(columns={"SGA_ID": "# SGA_ID"},inplace=True)
    dumdf.to_csv(desi_sga_dir+'/TF/Y1/output_sn.txt',columns=['# SGA_ID'],index=False )
    mu_sn=37.

    for index, _ in df.iterrows():
        row=df.iloc[[index]]
        combo_df = row.merge(pv_df, on=['SGA_ID'],suffixes=["","y"]) #,'R_MAG_SB26', 'V_0p4R26','BA'])

        Rcut = numpy.minimum(Rlim, combo_df['MU_SECONDARY'].tolist()[0]+Mlim)
        # print((combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) ,(combo_df['R_MAG_SB26'] < Rcut))
        select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
        combo_df = combo_df[select]
        if combo_df.shape[0] > 0:
            nsn=nsn+1
            combo_df = combo_df[select]
            combo_df['R_MAG_SB26'] = combo_df['R_MAG_SB26']  - combo_df['MU_SECONDARY'] + mu_sn
            Rcut = Rcut  - combo_df['MU_SECONDARY'].tolist()[0] + mu_sn
            combo_df['R_MAG_SB26_ERR'] = numpy.sqrt(combo_df['R_MAG_SB26_ERR'] + combo_df['MU_ERR']**2) 
            Nest = df["SGA_ID"]
            _first = "{} & ".format(Nest)
            _second = "{} & ".format(mu_sn)
            # glue these together into a comma string
            dum = combo_df['SGA_ID'].tolist()
            # for i in range(len(dum)):
            #     dum[i] = str(dum[i])
            # my_string = ', '.join(dum)
            # print(_first + _second + my_string + ' \\\\')
            N_per_cluster.append(combo_df.shape[0])
            alldf.append(combo_df)

            mu.append(mu_sn)
            # R2t.append(0)
            Rlim_eff.append(Rcut);      

    alldf = pandas.concat(alldf,ignore_index=True)

    return pv_df, alldf, N_per_cluster, mu, Rlim_eff

if __name__ == '__main__':
    # iron_cluster_json()
    # iron_cluster_json(cepheid=True)
    iron_cepheid_json()

    # iron_mag_plot()

    # all_table()

    # to_json(frac=0.1,cuts=True)
    # coma_json(cuts=True)


    # iron_mag_plot()
    # for i in range(1,11):
    #     segev_json("data/SGA_TFR_simtest_{}".format(str(i).zfill(3)))
    # # segev_plot()
