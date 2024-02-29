import fitsio
import numpy
import json

fn = "SGA-2020_iron_Vrot"

fits=fitsio.FITS(fn+".fits")
data=fits[1].read()

data_dic=dict()
for k in data.dtype.names:
        data_dic[k]=data[k].tolist()

data_dic['N'] = len(data_dic['SGA_ID'])

json_object = json.dumps(data_dic)

with open(fn+".json", 'w') as f:
    f.write(json_object)
