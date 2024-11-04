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

# fn = "SGA-2020_iron_Vrot_cuts"
# fn_sga = "data/SGA-2020_fuji_Vrot"
# fn_segev2 = "SGA_TFR_simtest_20240307"

fn = "DESI-DR1_TF_pv_cat_v2"
fn_sga = "SGA-2020_fuji_Vrot"
datadir = "../TFPV/data/"

cuts = {    "Rlim": 17.75,
    "Vmin": 70,
    # Vmax = 300. # nothing this bright
    "Mlim": -18,
    "Vmax": 1e4}

def mergedata():
    global fn, fn_sga, datadir, cuts

    # cuts
    Rlim = cuts["Rlim"]
    Vmin = cuts["Vmin"]
    Mlim = cuts["Mlim"]
    Vmax = cuts["Vmax"]

    cosi = 1/numpy.sqrt(2)
    q0=0.2
    balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)


    # DESI data
    table = Table.read(datadir+fn+".fits")
    pv_df = table.to_pandas()

    # Tully data
    table = Table.read(datadir+"/Tully15-Table3.fits")
    tully_df = table.to_pandas()

    # Cluster data files
    N_per_cluster = []
    mu = []
    Rlim_eff = []

    # analysis only is on cluster members.  Loop over all the cluster files.  Pick out TF data from those
    alldf=[]

    for fn in glob.glob(datadir+"/output_*.txt"):                       # loop over all cluster files
        Nest = re.search('output_(.+?).txt',fn).group(1)                # get the name of the cluster
        mu_ = tully_df.loc[tully_df["Nest"]==int(Nest)]["DM"].values[0] # distance modulus from Tully
        df = pandas.read_csv(fn)                                        # read the cluster file
        combo_df = df.merge(pv_df, on='SGA_ID')                         # combine with the DESI data

        # cuts
        Rcut = numpy.minimum(Rlim, mu_+Mlim)
        select = (combo_df['R_MAG_SB26'] < Rcut)  & (combo_df['V_0p4R26'] > Vmin) & (combo_df['V_0p4R26'] < Vmax) & (combo_df["BA"] < balim)
        combo_df = combo_df[select]
        if combo_df.shape[0] > 1:                                       # require at least 2 in a cluster
            N_per_cluster.append(combo_df.shape[0])
            alldf.append(combo_df)
            mu.append(mu_)
            Rlim_eff.append(Rcut);

    N_cluster=len(alldf)


    alldf = pandas.concat(alldf,ignore_index=True)
    alldf = alldf[["Z_DESI","SGA_ID", "V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR"]]
    return alldf, pv_df, N_cluster, N_per_cluster, mu, Rlim_eff

# make data and init files in stan format
def iron_cluster_json():

    global fn, fn_sga, datadir, cuts

    # cuts
    Rlim = cuts["Rlim"]
    Vmin = cuts["Vmin"]
    Mlim = cuts["Mlim"]
    Vmax = cuts["Vmax"]

    # DESI Cluster
    alldf, pv_df, N_cluster, N_per_cluster, mu, Rlim_eff = mergedata()

    # z = astropy.cosmology.z_at_value(cosmo.distmod, numpy.array(mu)*astropy.units.mag)
    # d = cosmo.luminosity_distance(z)

    # Make STAN data file
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

    with open("data/"+outname, 'w') as f:
        f.write(json_object)

    # make STAN init file
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

    outname2 = "iron_cluster_init.json"
    with open("data/"+outname2, 'w') as f:
        f.write(json.dumps(init))

def iron_mag_plot():
    global fn, fn_sga, datadir

    # DESI Cluster
    alldf, pv_df, N_cluster, N_per_cluster, mu, Rlim_eff = mergedata()


    plt.scatter(pv_df["Z_DESI"],pv_df["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Iron",color='black')
    plt.axhline(17.75,color='red')
    plt.xlabel('Z_DESI')
    plt.ylabel('R_MAG_SB26')


    plt.scatter(alldf["Z_DESI"],alldf["R_MAG_SB26"],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Iron Cluster",color="orange")

    # Coma

    Vmin_coma = 50

    table2 = Table.read(datadir+fn_sga+".fits")
    c_df = table2.to_pandas()

    comalist = [25532,30149,98934,122260,191275,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]
    select = numpy.logical_and.reduce((numpy.isin(c_df['SGA_ID'],comalist) , c_df['V_0p33R26'] > Vmin_coma))

    plt.scatter(c_df["Z_DESI"][select],c_df["R_MAG_SB26"][select],s=matplotlib.rcParams['lines.markersize'],linewidth=0,label="Coma",color="cyan")


    plt.legend()
    plt.savefig("zR.png")
    plt.show()  

# table with all the data in case this is first place where data is released.  Print only first 20
def all_table():
    global fn, fn_sga, datadir

    # DESI Cluster
    alldf, pv_df, N_cluster, N_per_cluster, mu, Rlim_eff = mergedata()

    cols = ["SGA_ID", "Z_DESI","V_0p4R26","V_0p4R26_err","R_MAG_SB26","R_MAG_SB26_ERR",'BA']
    pv_df = pv_df[cols]

    for index, row in pv_df.iterrows():
        if index<20:
            print("{:10.0f} & {:5.3f} & ${:6.1f} \\pm {:6.1f}$ & ${:6.2f} \\pm {:6.2f}$ & {:6.3f} \\\\".format(*row.to_numpy()))
        

if __name__ == '__main__':
    # iron_mag_plot()
    # all_table()
    iron_cluster_json()

