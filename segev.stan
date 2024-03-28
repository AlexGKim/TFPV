data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;
}

// transformed data{
//   vector[N] logV;
//   logV = log10(V_0p33R26);
// }

parameters {
  vector[N] logL;       // latent parameter

  // simplex[N] logL_simplex;
  // real logL_scale;  
  // real cR;

  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  real bR;

// parameters needed for second dispersion


  // Extra
  // vector<lower=-2, upper=2>[N] dlogL;       // latent parameter
  // simplex[N] rr_raw;
  // real<lower=0> rr_scale; // The bound needed to break degeneracy with TF relation
  vector[N] random_realization_1;
  real<lower=0> sigR;

}
transformed parameters {

  // real aR=tan(atanAR);
  // vector[N] logL = cR+ logL_scale*(logL_simplex-inv(N));
  // logL= logV+dlogL;
  // Extra
  // vector[N] random_realization_1;
  // random_realization_1=rr_scale*(rr_raw-inv(N));
}
model {
  real sinth=sin(atanAR);
  real costh=cos(atanAR);
  // R_MAG_SB26 ~ normal(bR + 27 + aR*logL , R_MAG_SB26_ERR);
  // V_0p33R26 ~ normal(pow(10,logL   ), V_0p33R26_err);
  R_MAG_SB26 ~ normal(bR + sinth*logL  - (random_realization_1)*costh, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(pow(10, costh*logL  + (random_realization_1)*sinth ), V_0p33R26_err);

  // Extra
  random_realization_1 ~ normal (0, sigR);
  sigR ~ cauchy(0.,1);
  // rr_scale ~ cauchy(0,1);

  // regularization helps to guide the fit along
  // log10(V_0p33R26) ~ normal(logL,2);
}
generated quantities {
   real aR=tan(atanAR);
}