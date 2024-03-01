functions {
  vector V_fiber(vector V, vector epsilon) {
    return V.*cos(epsilon);
  }
}
data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  // vector[N] G_MAG_SB26;
  // vector[N] G_MAG_SB26_ERR;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;
  // vector[N] Z_MAG_SB26;
  // vector[N] Z_MAG_SB26_ERR;
}
parameters {
  // real aG;
  // real bG;
  real aR;
  real bR;  
  // real aZ;
  // real bZ;
  vector<lower=0>[N] V;
  // vector[N] epsilon;
  // real<lower=0> sigG
  real<lower=0> sigR;
  // real<lower=0> sigZ
}
transformed parameters {
  vector[N] logV;
  logV = log10(V);
}
model {
  // V_0p33R26 ~ normal(V_fiber(V,epsilon), V_0p33R26_err);
  V_0p33R26 ~ normal(V, V_0p33R26_err);
  // G_MAG_SB26 ~ normal(mu + aG + bG*logV, G_MAG_SB26_ERR);
  R_MAG_SB26 ~ normal(bR + aR*logV, sqrt(square(R_MAG_SB26_ERR)+sigR*sigR));
  // Z_MAG_SB26 ~ normal(mu + aZ + bZ*logV, Z_MAG_SB26_ERR);
  // epsilon ~ normal(0,pi()/64.);
  sigR ~ cauchy(0,10);
}
