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

fn = "SGA-2020_iron_Vrot_cuts"
fn_sga = "data/SGA-2020_fuji_Vrot"
fn_segev2 = "SGA_TFR_simtest_20240307"

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



def iron_cluster_json():
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

    table = Table.read("data/Tully15-Table3.fits")
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
   # selection effects
    for fn in glob.glob("data/output_*.txt"):
        Nest = re.search('output_(.+?).txt',fn).group(1)
        mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
        R2t_=tully_df.loc[tully_df["Nest"]==int(Nest)]["R2t"].values[0]

        df = pandas.read_csv(fn)
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
            print(_first + _second + my_string + ' \\\\')
            N_per_cluster.append(combo_df.shape[0])
            alldf.append(combo_df)
            Nest = re.search('output_(.+?).txt',fn).group(1)
            mu.append(mu_)
            R2t.append(R2t_)
            Rlim_eff.append(Rcut);

    N_cluster=len(alldf)


    alldf = pandas.concat(alldf,ignore_index=True)
    alldf = alldf[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]

    print(N_cluster)
    print(alldf.shape)

    z = astropy.cosmology.z_at_value(cosmo.distmod, numpy.array(mu)*astropy.units.mag)
    d = cosmo.luminosity_distance(z)
    print("Radial depth ", 5*numpy.log10(1+numpy.array(R2t)*0.1/d.value).max())
    wef

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
    data_dic["mu"] = mu
    data_dic["Rlim_eff"] = Rlim_eff

# (-1.3565289337241162, 14.193371687903761, 1.0984767423119663)
    # data_dic["aR_init"]=-6.1
    # data_dic["alpha_dist_init"]=-1.3565289337241162
    # data_dic["xi_dist_init"]= 14.193371687903761
    # data_dic["omega_dist_init"]=1.0984767423119663

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


    outname = "iron_cluster.json"
    outname2 = "iron_cluster_init.json"

    with open("data/"+outname, 'w') as f:
        f.write(json_object)

#  vector[N] v = 373.137*v_raw + 222.371;
    init = dict()

    init["alpha_dist"]=data_dic["alpha_dist_init"]
    init["xi_dist"]= data_dic["xi_dist_init"]
    init["omega_dist"]=data_dic["omega_dist_init"]

    init["atanAR"] = numpy.arctan(data_dic["aR_init"])
    init['bR'] = (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster)).tolist()
    init['sigR'] = 0.1
    logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])




    init["logL_raw"]  = ((logL-init["xi_dist"]*numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()

    init["random_realization_raw"] = (numpy.zeros(data_dic['N'])).tolist()
    init["bR_offset"]= (numpy.zeros(data_dic['N_cluster'])).tolist()
    with open("data/"+outname2, 'w') as f:
        f.write(json.dumps(init))


# def iron_cluster_json(all=False):
#     fn = "SGA-2020_iron_Vrot"

#     Rlim = 17.75
#     Mlim = -17.
#     Vmin = 70
#     Vmax = 300

#     Mlim = -18
#     Vmax = 1e4

#     cosi = 1/numpy.sqrt(2)
#     q0=0.2
#     balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)

#     table = Table.read("data/"+fn+".fits")
#     pv_df = table.to_pandas()

#     table = Table.read("data/Tully15-Table3.fits")
#     tully_df = table.to_pandas()

#     # read in the cluster files
#     N_per_cluster = []
#     mu = []
#     Rlim_eff = []

#     alldf=[]
#    # selection effects
#     for fn in glob.glob("data/output_*.txt"):
#         Nest = re.search('output_(.+?).txt',fn).group(1)
#         mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
#         df = pandas.read_csv(fn)
#         combo_df = df.merge(pv_df, on='SGA_ID')
#         Rcut = numpy.minimum(Rlim, mu_+Mlim)
#         if not all:
#             select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
#             combo_df = combo_df[select]
#         if combo_df.shape[0] > 1:
#             N_per_cluster.append(combo_df.shape[0])
#             alldf.append(combo_df)
#             Nest = re.search('output_(.+?).txt',fn).group(1)
#             mu.append(mu_)
#             Rlim_eff.append(Rcut);

#     N_cluster=len(alldf)

#     alldf = pandas.concat(alldf,ignore_index=True)
#     alldf = alldf[["SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]

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
#     data_dic["mu"] = mu
#     data_dic["Rlim_eff"] = Rlim_eff

#     data_dic["aR_init"]= -6.26
#     data_dic["atanAR"] = numpy.arctan(data_dic["aR_init"])
#     data_dic["alpha_dist_init"]=1.25
#     data_dic["xi_dist_init"]= 13.3 * numpy.cos(data_dic["atanAR"])
#     data_dic["omega_dist_init"]= .08


#     dum=[]
#     for npc,m in zip(N_per_cluster,mu):
#         for j in range(npc):
#             dum.append(m)
#     data_dic["mu_all"]=dum

#     json_object = json.dumps(data_dic)

#     outname = "iron_cluster.json"
#     outname2 = "iron_cluster_init.json"

#     if all:
#         outname = "iron_cluster_all.json"
#         outname2 = "iron_cluster_all_init.json"

#     with open("data/"+outname, 'w') as f:
#         f.write(json_object)

# #  vector[N] v = 373.137*v_raw + 222.371;
#     init = dict()

#     init["atanAR"] = data_dic["atanAR"]
#     # init['bR'] = (-5.3173*0+ numpy.zeros(N_cluster)).tolist()

#     init['sigR'] = 0.1
#     logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])


#     init["alpha_dist"]=data_dic["alpha_dist_init"]
#     init["xi_dist"]= data_dic["xi_dist_init"] 
#     init["omega_dist"]=data_dic["omega_dist_init"]
#     init['bR'] =  (init["xi_dist"] * numpy.tan(init["atanAR"]) + numpy.zeros(N_cluster)).tolist()

#     # init["logL_raw"]  = ((logL-init["xi_dist"]/numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()
#     # init["logL_raw"]  = ((logL-init["xi_dist"]/numpy.cos(init["atanAR"]))/init["omega_dist"]).tolist()
#     init["logL_raw"]  = ((numpy.log10(data_dic["V_0p4R26"]) - init["xi_dist"])/init["omega_dist"]).tolist()

#     init["random_realization_raw"] = (numpy.zeros(data_dic['N'])).tolist()
#     with open("data/"+outname2, 'w') as f:
#         f.write(json.dumps(init))
    

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

    Rlim = 17.75

    fn = "SGA-2020_iron_Vrot"
    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()

    plt.scatter(pv_df["Z_DESI"],pv_df["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Iron",color='black')
    plt.axhline(17.75,color='red')
    plt.xlabel('Z_DESI')
    plt.ylabel('R_MAG_SB26')



    Rlim = 17.75
    Mlim = -17.
    Vmin = 70
    Vmax = 300. # nothing this bright

    Mlim = -18
    Vmax = 1e4

    cosi = 1/numpy.sqrt(2)
    q0=0.2
    balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)

    table = Table.read("data/Tully15-Table3.fits")
    tully_df = table.to_pandas()
    # read in the cluster files
    N_per_cluster = []
    mu = []
    Rlim_eff = []

    alldf=[]
   # selection effects
    for fn in glob.glob("data/output_*.txt"):
        Nest = re.search('output_(.+?).txt',fn).group(1)
        mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0]
        df = pandas.read_csv(fn)
        combo_df = df.merge(pv_df, on='SGA_ID')
        Rcut = numpy.minimum(Rlim, mu_+Mlim)
        select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
        combo_df = combo_df[select]
        if combo_df.shape[0] > 1:
            N_per_cluster.append(combo_df.shape[0])
            alldf.append(combo_df)
            Nest = re.search('output_(.+?).txt',fn).group(1)
            # mu.append(mu_)
            Rlim_eff.append(Rcut);

    N_cluster=len(alldf)


    alldf = pandas.concat(alldf,ignore_index=True)
    alldf = alldf[["Z_DESI","SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]

    plt.scatter(alldf["Z_DESI"],alldf["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Iron Cluster",color="orange")


    table2 = Table.read(fn_sga+".fits")
    # fits=fitsio.FITS()
    # data=fits[1].read()
    c_df = table2.to_pandas()


    Vmin = 50

    comalist = [25532,30149,98934,122260,191275,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]
    select = numpy.logical_and.reduce((numpy.isin(c_df['SGA_ID'],comalist) , c_df['V_0p33R26'] > Vmin))

    plt.scatter(c_df["Z_DESI"][select],c_df["R_MAG_SB26"][select],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Coma",color="cyan")



    plt.legend()
    plt.savefig("zR.png")
    plt.show()  

def all_table():

    fn = "SGA-2020_iron_Vrot"
    table = Table.read("data/"+fn+".fits")
    pv_df = table.to_pandas()
    cols = ["SGA_ID", "Z_DESI","V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR",'BA']
    pv_df = pv_df[cols]

    for index, row in pv_df.iterrows():
        if index<20:
            print("{:10.0f} & {:5.3f} & ${:6.1f} \\pm {:6.1f}$ & ${:6.2f} \\pm {:6.2f}$ & {:6.3f} \\\\".format(*row.to_numpy()))
        

if __name__ == '__main__':
    # to_json(frac=0.1,cuts=True)
    # coma_json(cuts=True)
    iron_cluster_json()
    # all_table()
    # iron_mag_plot()
    # for i in range(1,11):
    #     segev_json("data/SGA_TFR_simtest_{}".format(str(i).zfill(3)))
    # # segev_plot()
