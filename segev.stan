
data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;
}

transformed data{
  vector[N] logV;
  logV = log10(V_0p33R26);
}

parameters {
  vector<lower=-1, upper=1>[N] dlogL;       // latent parameter
  // vector[N] logL;       // latent parameter
// parameters that describe each distribution
  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  // real<lower=atan(-pi()/2), upper=0> atanaR;
  real bR;
  real<lower=0,upper=0.1> sigR;
  vector[N] random_realization_1;
  // simplex[N] rr_raw;
  // real<lower=0,upper=N/2.> rr_scale; // The bound needed to break degeneracy with TF relation
}
transformed parameters {
  // vector[N] random_realization_1;
  // // random_realization_1=0.002*N*(rr_raw-inv(N));
  // random_realization_1=rr_scale*(rr_raw-inv(N));
  real aR;
  aR=tan(atanAR);
  vector[N] logL;
  logL= logV+dlogL;
}
model {
  random_realization_1 ~ normal(0, sigR);
  // real mn_random=mean(random_realization_1);
  // mn_random = 0.;
  // R_MAG_SB26 ~ normal(bR + aR*logL , R_MAG_SB26_ERR);
  // V_0p33R26 ~ normal(pow(10,logL   ), V_0p33R26_err);
  R_MAG_SB26 ~ normal(bR + aR*logL  - random_realization_1/aR, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(pow(10,logL  + random_realization_1 ), V_0p33R26_err);

  sigR ~ cauchy(0.,0.02);
}