import numpy
import pandas

# def fuji():

pars = ['aR','bR_use','sigR','xi_dist','omega_dist_use']
labels={"aR": r"$a$", "bR_use": r"$b$", "sigR": r"$\sigma_R$",  "xi_dist": r"$\log{V}_{TF}$", "omega_dist_use" : r"$\sigma_{\log{V}_{TF}}$"}

for _ in [3,4]:
    dum=[pandas.read_csv("output/fuji_{}11_{}.csv".format(_,i),comment='#') for i in range(1,5)]
    for df_ in dum:
        df_["bR_use"] = df_["bR"] - df_["xi_dist"]*df_["aR"]
        df_["omega_dist_use"] = df_["omega_dist"] * numpy.cos(df_["atanAR"])
    dum=pandas.concat(dum)

    qs = [dum[s].quantile(q= (0.32,0.5,1-0.32)) for s in pars]

    for l, q in zip(labels,qs):
    	print(l,q)

fuji()