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
  // 4 cases
  // 1 : line fit only
  // 2 : log-V dispersion
  // 3 : mag dispersion
  // 4 : perp dispersion
  int dispersion_case=4;
}

parameters {
  vector[N] epsilon;    // angle error
  real <lower=0, upper=0.5> pD;   // dwarf population fraction

  # population 1
  vector[N] logL;       // latent parameter
  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  real bR; //[2];

  vector[N] random_realization_1;
  real<lower=0> sigR;

  # population 1
  vector[N] logL2;       // latent parameter
  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR2;
  real bR2; //[2];

  vector[N] random_realization_2;
  real<lower=0> sigR2;

}
transformed parameters {
  real sinth; real costh;
  if (dispersion_case == 1)
  {
    sinth=0; costh=0;
  }
  else if (dispersion_case ==2)
  {
    sinth=1; costh=0;
  }
  else if (dispersion_case ==3)
  {
    sinth=0; costh=1;
  }
  else if (dispersion_case==4)
  {
    sinth=sin(atanAR); costh=cos(atanAR);
  }
}

model {

  R_MAG_SB26 ~ normal(bR + sinth*logL  - (random_realization_1)*costh, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(pow(10, costh*logL  + (random_realization_1)*sinth ), V_0p33R26_err);

  random_realization_1 ~ normal (0, sigR);
  sigR ~ cauchy(0.,1);
}
generated quantities {
   real aR=tan(atanAR);
}


// model {
//   vector[2] RV_data;
//   vector[2] RV_model;
//   matrix[2,2] cov;
//   vector[2] logexp;

//   vector[2] lnpDs;
//   vector[N] Vr;
//   vector[N] L;

//   int angle_error = 0;
//   int LisV = 0;

//   lnpDs[1]=log(1-pD);
//   lnpDs[2]=log(pD);
//   Vr = pow(10, logVr);

//   if (LisV == 1){
//     // When there is the latent parameter logL is equal to logV
//     L = pow(10,logL);
//     if (angle_error == 0){
//       V_0p33R26 ~ normal(L, V_0p33R26_err);
//     } else {
//       V_0p33R26 ~ normal(V_fiber(L,epsilon), V_0p33R26_err);
//     }
//     for (n in 1:N){
//       for (m in 1:2){
//         logexp[m] = lnpDs[m] + normal_lpdf(R_MAG_SB26[n] | bR[m] + aR[m]*logL[n], sqrt(sigR[m]*sigR[m]+R_MAG_SB26_ERR[n]*R_MAG_SB26_ERR[n]));
//       }
//       target += log_sum_exp(logexp);
//     }
//   } else if (LisV == 0)
//   {
//     for (n in 1:N){
//       RV_data[1]=Rmag[n];
//       RV_data[2]=logVr[n];
//       for (m in 1:2){
//         cov[1,1]=sigR[m]*sigR[m];
//         cov[2,2]=siglogL[m]*siglogL[m];
//         cov[1,2]=cor[m]*sigR[m]*siglogL[m];
//         cov[2,1]=cov[1,2];
//         RV_model[1]=bR[m] + aR[m]*logL[n];
//         RV_model[2]=logL[n];
//         // logexp[m] =  lnpDs[m] + multi_normal_lpdf(RV_data | RV_model, cov);
//         logexp[m] =  lnpDs[m] + multi_student_t_lpdf(RV_data |1., RV_model, cov);
//         // target += logexp[m];
//       }
//       target += log_sum_exp(logexp);
//     }

//     if (angle_error == 0){
//         V_0p33R26 ~ normal(Vr, V_0p33R26_err);
//     } else {
//         V_0p33R26 ~ normal(V_fiber(Vr,epsilon), V_0p33R26_err);
//     }
//     R_MAG_SB26 ~ normal(Rmag, R_MAG_SB26_ERR);
//   } 
//   // else if (LisV == 2)
//   // {
//   //   random_realization_1 ~ normal(0, sigR[1]);
//   //   random_realization_2 ~ normal(0, sigR[2]);
//   //   for (n in 1:N){
//   //     // for (m in 1:2){
//   //       // logexp[1] =  lnpDs[1] + normal_lpdf(R_MAG_SB26_ERR[n] |  bR[1] + aR[1]*logL[n] + random_realization_1[n]*1, R_MAG_SB26_ERR[n])
//   //       //                 + normal_lpdf(V_0p33R26[n]|  pow(10,logL[n]+ random_realization_1[n]*5), V_0p33R26_err[n]) ;

//   //       // logexp[2] =  lnpDs[2] + normal_lpdf(R_MAG_SB26_ERR[n] |  bR[2] + aR[2]*logL[n] + random_realization_2[n]*1, R_MAG_SB26_ERR[n])
//   //       //                 + normal_lpdf(V_0p33R26[n]|  pow(10,logL[n]+ random_realization_2[n]*5), V_0p33R26_err[n]) ;


//   //       // logexp[1] =  lnpDs[1] + normal_lpdf(R_MAG_SB26_ERR[n] |  bR[1] + aR[1]*logL[n] + random_realization_1[n]*cor_unit1[1], R_MAG_SB26_ERR[n])
//   //       //                 + normal_lpdf(V_0p33R26[n]|  pow(10,logL[n]+ random_realization_1[n]*cor_unit1[2]), V_0p33R26_err[n]) ;

//   //       // logexp[2] =  lnpDs[2] + normal_lpdf(R_MAG_SB26_ERR[n] |  bR[2] + aR[2]*logL[n] + random_realization_2[n]*cor_unit2[1], R_MAG_SB26_ERR[n])
//   //       //                 + normal_lpdf(V_0p33R26[n]|  pow(10,logL[n]+ random_realization_2[n]*cor_unit2[2]), V_0p33R26_err[n]) ;
//   //     // }
//   //     target += log_sum_exp(logexp);
//   //   }

//   //   // if (angle_error == 0){
//   //   //     V_0p33R26 ~ normal(Vr, V_0p33R26_err);
//   //   // } else {
//   //   //     V_0p33R26 ~ normal(V_fiber(Vr,epsilon), V_0p33R26_err);
//   //   // }

//   // }


//   sigR ~ cauchy(0,10);
//   siglogL ~ cauchy(0,10.);
//   epsilon ~ normal(0,pi()/64.);
// }
// generated quantities{
//   real cor1;
//   real cor2;
//   cor1=cor_unit1[1]/cor_unit1[2];
//   cor2=cor_unit2[1]/cor_unit2[2];
// }
