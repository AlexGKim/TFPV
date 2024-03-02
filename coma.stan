functions {
  vector V_fiber(vector V, vector epsilon) {
    return V.*cos(epsilon);
  }
}
data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;
}
  transformed data {
    vector<lower=-1, upper=1>[2] cor;
    cor[1]=0; cor[2]=0;
  }
parameters {
  vector[N] epsilon;
  vector[N] logL;
  vector[N] logVr;
  vector[N] Rmag;
  real <lower=0, upper=0.5> pD;

// parameters that describe each distribution
  vector[2] aR;
  vector[2] bR;
  // vector<lower=-1, upper=1>[2] cor;
  vector<lower=0>[2] sigR;
  vector<lower=0>[2] siglogL;
}

model {
  vector[2] RV_data;
  vector[2] RV_model;
  matrix[2,2] cov;
  vector[2] logexp;

  vector[2] lnpDs;
  vector[N] Vr;
  vector[N] L;

  int angle_error = 0;
  int LisV = 1;

  lnpDs[1]=log(1-pD);
  lnpDs[2]=log(pD);
  Vr = pow(10, logVr);

  if (LisV == 1){
    // When there is the latent parameter logL is equal to logV
    L = pow(10,logL);
    if (angle_error == 0){
      V_0p33R26 ~ normal(L, V_0p33R26_err);
    } else {
      V_0p33R26 ~ normal(V_fiber(L,epsilon), V_0p33R26_err);
    }
    for (n in 1:N){
      for (m in 1:2){
        logexp[m] = lnpDs[m] + normal_lpdf(R_MAG_SB26[n] | bR[m] + aR[m]*logL[n], sqrt(sigR[m]*sigR[m]+R_MAG_SB26_ERR[n]*R_MAG_SB26_ERR[n]));
      }
      target += log_sum_exp(logexp);
    }
  } else
  {
    for (n in 1:N){
      RV_data[1]=Rmag[n];
      RV_data[2]=logVr[n];
      for (m in 1:2){
        cov[1,1]=sigR[m]*sigR[m];
        cov[2,2]=siglogL[m]*siglogL[m];
        cov[1,2]=cor[m]*sigR[m]*siglogL[m];
        cov[2,1]=cov[1,2];
        RV_model[1]=bR[m] + aR[m]*logL[n];
        RV_model[2]=logL[n];
        logexp[m] =  lnpDs[m] + multi_normal_lpdf(RV_data | RV_model, cov);
      }
      target += log_sum_exp(logexp);
    }

    if (angle_error == 0){
        V_0p33R26 ~ normal(Vr, V_0p33R26_err);
    } else {
        V_0p33R26 ~ normal(V_fiber(Vr,epsilon), V_0p33R26_err);
    }
    R_MAG_SB26 ~ normal(Rmag, R_MAG_SB26_ERR);
  }
  sigR ~ cauchy(0,100);
  siglogL ~ cauchy(0,.1);
  epsilon ~ normal(0,pi()/64.);
}
