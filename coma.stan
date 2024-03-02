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
transformed data {
  real cor;
  cor=0.5;
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
  real<lower=0> sigV;
  // vector[N] dR;
  // vector[N] dV;
  // real<lower=-1, upper=1> cor;
  // real<lower=0> sigZ
}
model {
  vector[2] RV_data;
  vector[2] RV_model;
  // vector[2] zeros;
  matrix[2,2] cov;
  vector[N] logV;
  // zeros[1]=0;
  // zeros[2]=0;
  logV = log10(V);
  // V_0p33R26 ~ normal(V_fiber(V,epsilon), V_0p33R26_err);
  // V_0p33R26 ~ normal(V+dV, V_0p33R26_err);
  // V_0p33R26 ~ normal(V, sqrt(square(V_0p33R26_err)+sigV*sigV));
  // R_MAG_SB26 ~ normal(bR + aR*logV+dR , R_MAG_SB26_ERR);
  // epsilon ~ normal(0,pi()/64.);
  for (n in 1:N){
    // cov[1,1]=sigR*sigR;
    // cov[2,2]=sigV*sigV;
    cov[1,1]=sigR*sigR + R_MAG_SB26_ERR[n]*R_MAG_SB26_ERR[n];
    cov[2,2]=sigV*sigV + V_0p33R26_err[n]*V_0p33R26_err[n];
    cov[1,2]=cor*sigR*sigV;
    cov[2,1]=cov[1,2];
    // RV_data[1]=dR[n];
    // RV_data[2]=dV[n];
    RV_data[1]=R_MAG_SB26[n];
    RV_data[2]=V_0p33R26[n];
    RV_model[1]=bR + aR*logV[n];
    RV_model[2]=V[n];
    RV_data ~ multi_normal(RV_model, cov);
  }

  sigR ~ cauchy(0,10);
  sigV ~ cauchy(0,1000);
}
