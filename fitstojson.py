import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt

fn = "SGA-2020_iron_Vrot_cuts"
fn_sga = "data/SGA-2020_fuji_Vrot"
fn_segev2 = "SGA_TFR_simtest_20240307"

rng = numpy.random.default_rng(seed=42)

def coma_json():
    fits=fitsio.FITS(fn_sga+".fits")
    data=fits[1].read()


    # comalist = [8032,20886,25532,98934,100987,122260,127141,128944,139660,171794,191275,191496,192582,196592,202666,221178,238344,289665,291879,301194,302524,309306,330166,337817,343570,364410,364929,365429,366393,378180,378842,381769,390630,455486,465951,477610,479267,486394,540744,556334,566771,573264,629860,637552,645151,652931,665961,729931,733069,735080,747077,748600,753474,796671,811359,819754,824392,826543,827339,834049,837120,841705,900049,905270,908303,917608,918100,928810,972260,993595,995924,1009928,1014365,1020852,1050173,1089288,1115705,1122082,1144453,1167691,1195008,1198552,1201916,1203610,1203786,1204237,1206707,1209774,1269260,1272144,1274171,1274189,1274409,1281982,1284002,1293940,1294562,1323268,1349168,1356626,1364394,1379275,1387126,1387991]
    comalist = [25532,30149,98934,122260,191275,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]
    # comalist = [25532,30149,98934,122260,196592,202666,221178,291879,309306,337817,364410,364929,365429,366393,378842,455486,465951,479267,486394,566771,645151,747077,748600,753474,759003,819754,826543,841705,917608,995924,1050173,1167691,1195008,1203610,1203786,1269260,1274409,1284002,1323268,1352019,1356626,1364394,1379275,1387991]
    select = numpy.isin(data['SGA_ID'],comalist)

    # plt.errorbar(numpy.log10(data['V_0p33R26'][select]), data['R_MAG_SB26'][select],yerr=data['R_MAG_SB26_ERR'][select],xerr=[numpy.log10(data['V_0p33R26'][select])-numpy.log10(data['V_0p33R26'][select]-data['V_0p33R26_err'][select]),numpy.log10(data['V_0p33R26'][select]+data['V_0p33R26_err'][select])-numpy.log10(data['V_0p33R26'][select])],fmt='.')
    # plt.plot(numpy.array([1.4,2.5]),23-3.5*numpy.array([1.4,2.5]))
    # plt.ylim((18,13))
    # plt.show()
    # wef

    data_dic=dict()
    for k in data.dtype.names:
        if k not in ['SGA_GALAXY','GALAXY','MORPHTYPE','BYHAND','REF','GROUP_NAME','GROUP_PRIMARY','BRICKNAME','D26_REF']:
            if k in ['G_MAG_SB26_ERR','R_MAG_SB26_ERR','Z_MAG_SB26_ERR']:
                w=numpy.where(data[k]<0)
                data[k][w[0]]=1e8

            data_dic[k]=data[k][select].tolist()

    data_dic['N'] = len(data_dic['SGA_ID'])

    json_object = json.dumps(data_dic)

    with open(fn_sga+".json", 'w') as f:
        f.write(json_object)

    init = dict()
    # init["v_raw"]=(numpy.array(data_dic["V_0p33R26"])/139.35728557650154).tolist()

    init["atanAR"] = numpy.arctan(-6.1)
    init['bR'] = -6.91
    logL = numpy.log10(data_dic["V_0p33R26"])/numpy.cos(init["atanAR"])

    init["alpha_dist"]=-3.661245022462153
    init["xi_dist"]= 14.913405242237685
    init["omega_dist"]=2.2831016215521247

    init["mu_dist"]=13.133570672711606
    init["sigma_dist"]= 1.5160651053079683
    init["logL_raw"]  = ((logL-init["xi_dist"])/init["omega_dist"]).tolist()
    with open(fn_sga+"_init.json", 'w') as f:
        f.write(json.dumps(init))

def to_json(frac=1, cuts=False):
    fn = "SGA-2020_iron_Vrot"

    Rlim = 17.75
    Mlim = -17.
    Vmin = 70
    Vmax = 500
    cosi = 1/numpy.sqrt(2)
    q0=0.2
    balim = numpy.sqrt(cosi**2 * (1-q0**2) + q0**2)

    fits=fitsio.FITS("data/"+fn+".fits")
    data=fits[1].read()

    # selection effects

    mu = cosmo.distmod(data['Z_DESI']).value
    Rlim_eff = numpy.minimum(Rlim, mu+Mlim)

    # add extra noise degrading data to help fit
    dt = {'names':['Vhat','Vhat_noise','Rhat'], 'formats':[float, float,float]}
    extradata = numpy.zeros(len(data['Z_DESI']),dtype=dt)
    extradata['Vhat_noise'] = 0.00*data["V_0p4R26"]
    Rhat_noise = 0.00
    extradata['Vhat'] = numpy.random.normal(loc=data["V_0p4R26"], scale=extradata['Vhat_noise'])
    extradata['Rhat'] = numpy.random.normal(loc=data['R_MAG_SB26'], scale=Rhat_noise)
    if cuts:
        select = numpy.logical_and.reduce((extradata['Rhat'] < Rlim_eff  , extradata['Vhat'] > Vmin, extradata['Vhat'] < Vmax, data["BA"] < balim))
    else:
        select = data['R_MAG_SB26'] < Rlim

    data_dic=dict()
    for k in data.dtype.names:
        if k not in ['SGA_GALAXY','GALAXY','MORPHTYPE','BYHAND','REF','GROUP_NAME','GROUP_PRIMARY','BRICKNAME','D26_REF']:
            if k in ['G_MAG_SB26_ERR','R_MAG_SB26_ERR','Z_MAG_SB26_ERR']:
                w=numpy.where(data[k]<0)
                data[k][w[0]]=1e8

            data_dic[k]=data[k][select].tolist()

    for k in extradata.dtype.names:
        data_dic[k]=extradata[k][select].tolist()

    data_dic['mu'] = cosmo.distmod(data_dic['Z_DESI']).value.tolist()
    data_dic['Rlim_eff'] = Rlim_eff.tolist()
    z = numpy.array(data_dic["Z_DESI"])
    dv = 300
    dm = (5/numpy.log(10)*(1+z)**2*dv/cosmo.H(z)/cosmo.luminosity_distance(z)).value
    data_dic['dm_v'] = dm.tolist()

    N_all = len(data_dic['Z_DESI'])
    if frac !=1 :
        ind = numpy.random.randint(0, high=N_all, size=int(N_all*frac))
        for key, value in data_dic.items():
            value=numpy.array(value)[ind]
            data_dic[key] = value.tolist()

    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Rlim'] = Rlim
    data_dic['Mlim'] = Mlim
    data_dic['Vmin'] = Vmin
    data_dic['Vmax'] = Vmax

    data_dic['Rhat_noise'] = Rhat_noise


    json_object = json.dumps(data_dic)

    cstr=""
    if cuts:
        cstr="_cuts"
    if frac==1:
        outname = fn+cstr+".json"
        outname2 = fn+cstr+"_init.json"
    else:
        outname =  fn+cstr+"_sub_{:4.2f}.json".format(frac)
        outname2 = fn+cstr+"_sub_{:4.2f}_init.json".format(frac)

    with open("data/"+outname, 'w') as f:
        f.write(json_object)

#  vector[N] v = 373.137*v_raw + 222.371;
    init = dict()

    init["atanAR"] = numpy.arctan(-6.1)
    init['bR'] = -6.8
    init['sigR'] = 0.1
    logL = numpy.log10(data_dic["V_0p4R26"])/numpy.cos(init["atanAR"])

    if cuts:
        # init["alpha_dist"]=-2.4813505391290436
        # init["xi_dist"]= 14.628796578863792
        # init["omega_dist"]=1.4880837674710605
        init["alpha_dist"]=-2.
        init["xi_dist"]= 14.5
        init["omega_dist"]=1.6    
    else:
        init["alpha_dist"]=-3.661245022462153
        init["xi_dist"]= 14.913405242237685
        init["omega_dist"]=2.2831016215521247

    # init["mu_dist"]=13.133570672711606
    # init["sigma_dist"]= 1.5160651053079683
    init["logL_raw"]  = ((logL-init["xi_dist"])/init["omega_dist"]).tolist()

    init["dv"] = (numpy.zeros(data_dic['N'])-.5).tolist()
    init["random_realization_raw"] = (numpy.zeros(data_dic['N'])-.5).tolist()
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


if __name__ == '__main__':
    to_json(frac=0.1,cuts=True)
    # coma_json()
    # for i in range(1,11):
    #     segev_json("data/SGA_TFR_simtest_{}".format(str(i).zfill(3)))
    # # segev_plot()