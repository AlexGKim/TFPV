import fitsio
import numpy
import json
from astropy.cosmology import Planck18 as cosmo
import  matplotlib.pyplot as plt

fn = "SGA-2020_iron_Vrot"

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
    data_dic['Rlim'] = Rlim

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
    to_json()