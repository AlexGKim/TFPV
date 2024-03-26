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
  vector[N] logL;       // latent parameter
// parameters that describe each distribution
  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  real bR;

  // Extra
  // vector<lower=-2, upper=2>[N] dlogL;       // latent parameter
  real<lower=0> sigR;
  vector[N] random_realization_1;
  // simplex[N] rr_raw;
  // real<lower=0> rr_scale; // The bound needed to break degeneracy with TF relation
}
transformed parameters {

  real aR;
  aR=tan(atanAR);
  // vector[N] logL;
  // logL= logV+dlogL;
  // Extra
  // vector[N] random_realization_1;
  // random_realization_1=rr_scale*(rr_raw-inv(N));
}
model {

  // R_MAG_SB26 ~ normal(bR + aR*logL , R_MAG_SB26_ERR);
  // V_0p33R26 ~ normal(pow(10,logL   ), V_0p33R26_err);
  R_MAG_SB26 ~ normal(bR + aR*logL  - random_realization_1/aR, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(pow(10,logL  + random_realization_1 ), V_0p33R26_err);

  // Extra
  // real mn_random=mean(random_realization_1);
  // mn_random = 0.;
  random_realization_1 ~ normal (0, sigR);
  sigR ~ cauchy(0.,0.5);
  // rr_scale ~ cauchy(0,1);
}