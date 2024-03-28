data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;
}

parameters {
  vector[N] logL;       // latent parameter
  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  real bR;

  // Extra
  vector[N] random_realization_1;
  real<lower=0> sigR;

}
transformed parameters {
}
model {
  real sinth=sin(atanAR);
  real costh=cos(atanAR);
  // Just straight line
  // R_MAG_SB26 ~ normal(bR + 27 + aR*logL , R_MAG_SB26_ERR);
  // V_0p33R26 ~ normal(pow(10,logL   ), V_0p33R26_err);
  R_MAG_SB26 ~ normal(bR + sinth*logL  - (random_realization_1)*costh, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(pow(10, costh*logL  + (random_realization_1)*sinth ), V_0p33R26_err);

  random_realization_1 ~ normal (0, sigR);
  sigR ~ cauchy(0.,1);
}
generated quantities {
   real aR=tan(atanAR);
}