import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt

fn = "SGA-2020_iron_Vrot"
fn_sga = "SGA-2020_fuji_Vrot"

def coma_json():
    fits=fitsio.FITS(fn_sga+".fits")
    data=fits[1].read()


    comalist = [8032,20886,25532,98934,100987,122260,127141,128944,139660,171794,191275,191496,192582,196592,202666,221178,238344,289665,291879,301194,302524,309306,330166,337817,343570,364410,364929,365429,366393,378180,378842,381769,390630,455486,465951,477610,479267,486394,540744,556334,566771,573264,629860,637552,645151,652931,665961,729931,733069,735080,747077,748600,753474,796671,811359,819754,824392,826543,827339,834049,837120,841705,900049,905270,908303,917608,918100,928810,972260,993595,995924,1009928,1014365,1020852,1050173,1089288,1115705,1122082,1144453,1167691,1195008,1198552,1201916,1203610,1203786,1204237,1206707,1209774,1269260,1272144,1274171,1274189,1274409,1281982,1284002,1293940,1294562,1323268,1349168,1356626,1364394,1379275,1387126,1387991]

    select = numpy.isin(data['SGA_ID'],comalist)

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


def to_json():

    Rlim = 17.75

    fits=fitsio.FITS(fn+".fits")
    data=fits[1].read()


    select = data['R_MAG_SB26'] < Rlim

    data_dic=dict()
    for k in data.dtype.names:
        if k not in ['SGA_GALAXY','GALAXY','MORPHTYPE','BYHAND','REF','GROUP_NAME','GROUP_PRIMARY','BRICKNAME','D26_REF']:
            if k in ['G_MAG_SB26_ERR','R_MAG_SB26_ERR','Z_MAG_SB26_ERR']:
                w=numpy.where(data[k]<0)
                data[k][w[0]]=1e8

            data_dic[k]=data[k][select].tolist()

    data_dic['N'] = len(data_dic['SGA_ID'])
    data_dic['Rlim'] = Rlim


    data_dic['mu'] = cosmo.distmod(data_dic['Z_DESI']).value.tolist()

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

if __name__ == '__main__':
    coma_json()