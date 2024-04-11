import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt

fn = "SGA-2020_iron_Vrot"
fn_sga = "SGA-2020_fuji_Vrot"
fn_segev2 = "SGA_TFR_simtest_20240307"

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


def to_json(frac=1):
    fn = "SGA-2020_iron_Vrot"
    Rlim = 17.75

    fits=fitsio.FITS("data/"+fn+".fits")
    data=fits[1].read()


    select = data['R_MAG_SB26'] < Rlim

    data_dic=dict()
    for k in data.dtype.names:
        if k not in ['SGA_GALAXY','GALAXY','MORPHTYPE','BYHAND','REF','GROUP_NAME','GROUP_PRIMARY','BRICKNAME','D26_REF']:
            if k in ['G_MAG_SB26_ERR','R_MAG_SB26_ERR','Z_MAG_SB26_ERR']:
                w=numpy.where(data[k]<0)
                data[k][w[0]]=1e8

            data_dic[k]=data[k][select].tolist()

    data_dic['mu'] = cosmo.distmod(data_dic['Z_DESI']).value.tolist()

    N_all = len(data_dic['Z_DESI'])
    if frac !=1 :
        ind = numpy.random.randint(0, high=N_all, size=int(N_all*frac))
        for key, value in data_dic.items():
            value=numpy.array(value)[ind]
            data_dic[key] = value.tolist()

    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Rlim'] = Rlim

    json_object = json.dumps(data_dic)

    if frac==1:
        outname = fn+".json"
    else:
        outname =  fn+"_sub.json"

    with open(outname, 'w') as f:
        f.write(json_object)

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
    to_json(0.1)
    # # coma_json()
    # #segev_json()
    # for i in range(1,11):
    #     segev_json("data/SGA_TFR_simtest_{}".format(str(i).zfill(3)))
    # # segev_plot()