
data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;
}

parameters {
  vector[N] logL;       // latent parameter

// parameters that describe each distribution
  real<upper=0> aR;
  real bR;
  real<lower=0> sigR;
  vector[N] random_realization_1;
}

model {
  vector[2] unit;

  unit[1]=1/sqrt(aR*aR+1);
  unit[2]=aR*unit[1];
  
  random_realization_1 ~ normal(0, 0.025);
  // R_MAG_SB26 ~ normal(bR + aR*logL , R_MAG_SB26_ERR);
  // V_0p33R26 ~ normal(pow(10,logL   ), V_0p33R26_err);
  R_MAG_SB26 ~ normal(bR + aR*logL + random_realization_1*unit[1], R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(pow(10,logL  - random_realization_1*unit[2] ), V_0p33R26_err);

  // sigR ~ cauchy(0.025,0.025);
}