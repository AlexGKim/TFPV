// functions {
//   vector V_fiber(vector V, vector epsilon) {
//     return V./cos(epsilon);
//   }
// }

data {
  int<lower=0> N;
  // vector[N] V_0p33R26;
  // vector[N] V_0p33R26_err;
  vector[N] V_0p4R26;
  vector[N] V_0p4R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;

  //for iron
  vector[N] mu;
  vector[N] dm_v;
  real Rlim;
}

transformed data {

  int iron = 1;
  // 4 cases
  // 1 : line fit only
  // 2 : log-V dispersion
  // 3 : mag dispersion
  // 4 : perp dispersion
  int dispersion_case=2;

  int pure = 1;
  int angle_error = 0;

  real dwarf_mag=-17. + 34.7;

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 5.;
  real angle_dispersion = angle_dispersion_deg/180*pi();

  vector[N] dR = sqrt(R_MAG_SB26_ERR.*R_MAG_SB26_ERR+dm_v.*dm_v);

  vector[N] logVovercosth = log10(V_0p4R26)/cos(atan(-6.1));

}


// from eyeball look at data expect b ~ -7.1, a ~ -6.1
// average logV ~ 2.14
parameters {
  // vector<lower=0, upper=pi()/4>[N] epsilon;    // angle error. There is a 1/cos so avoid extreme


  // population 1
  // vector<lower=-.4/cos(atan(-6.1)), upper=.4/cos(atan(-6.1))>[N] logL_;
  vector<lower=pow(10,-.4/cos(atan(-6.1))), upper=pow(10,.4/cos(atan(-6.1)))>[N] L_;
  vector<lower=0>[N] V_; // V = V_^costh

  real<lower=-7.1-2, upper=-7.1+2> bR;
  // vector[N] logL;       // latent parameter
  // real bR;
  // real<lower=-pi()*(.5-1./32) , upper=-pi()*1./3> atanAR;
  real<lower=atan(-6.1)-.1 , upper=atan(-6.1)+.1> atanAR;

  vector[N] random_realization;
  real<lower=0> sigR;

  // population 2
  // vector <lower=0, upper=0.25>[N] pD;   // dwarf population fraction
  // vector[N] logL2;       // latent parameter
  // real<lower=pi()/4, upper=5*pi()/4> atanAR2;
  // real bR2;

  // vector[N] random_realization2;
  // real<lower=0> sigR2;

}
model {
  // slope of TF Relation
  real sinth = sin(atanAR);
  real costh = cos(atanAR);

  // vector[N] logL = logVovercosth + logL_;
  // vector[N] logL = logVovercosth + log(L_);
  vector[N] logL = log(L_);

  // real sinth2 = sin(atanAR2);
  // real costh2 = cos(atanAR2);

  // slope of redsidual dispersion
  real sinth_r; real costh_r; real sinth2_r; real costh2_r; 
  if (dispersion_case == 1)
  {
    sinth_r=0; costh_r=0; // sinth2_r=0; costh2_r=0;
  }
  else if (dispersion_case ==2)
  {
    sinth_r=1; costh_r=0; //sinth2_r=1; costh2_r=0;
  }
  else if (dispersion_case ==3)
  {
    sinth_r=0; costh_r=1; //sinth2_r=0; costh2_r=1;
  }
  else if (dispersion_case==4)
  {
    sinth_r=-costh; costh_r=sinth; //sinth2_r=-costh2; costh2_r=sinth2;
  }

  // vector[N] random_realization;
  // velocity model with or without axis error
  vector[N] VtoUse;
  if (dispersion_case ==1) {
    // VtoUse = pow(10, costh*logL );
    VtoUse = pow(V_,costh);
  }
  else {    
    VtoUse = pow(10, costh*logL  + random_realization*costh_r );
  }
  // if (angle_error == 1){
  //     VtoUse = V_fiber(VtoUse,epsilon);
  // } 

  // print(mean(log10(V_0p4R26))," ",mean(costh*logL));
  // print(mean(R_MAG_SB26)," ",mean(bR + mu + sinth*logL));
  if (pure == 1)
  {

    vector[N] mint;
    if (dispersion_case ==1){
        mint = bR + mu + sinth*logL;
    }
    else {
        mint = bR + mu + sinth*logL  + random_realization*sinth_r;
    }
    R_MAG_SB26 ~ normal(mint, dR);
    V_0p4R26 ~ cauchy(VtoUse, V_0p4R26_err);
    // print(mint-Rlim);
    // print(max(mint-Rlim));
    // print(log(erfc((mint-Rlim)./R_MAG_SB26_ERR/sqrt(2))));
    // print(-sum(log(erfc((mint-Rlim)./R_MAG_SB26_ERR/sqrt(2)))));
    // target += - sum(log(erfc((mint-Rlim)./R_MAG_SB26_ERR/sqrt(2))));
  }
  // else
  // {
  //   vector[N] lnpDs1 = log(1-pD);
  //   vector[N] lnpDs2 = log(pD);
  //   vector[N] VtoUse2 = pow(10, costh2*logL2  + (random_realization2)*costh2_r );
  //   if (angle_error == 1){
  //       VtoUse2 = V_fiber(VtoUse2,epsilon);
  //   } 
  //   vector[2] logexp;
  //   for (n in 1:N)
  //   {
  //     logexp[1] =  lnpDs1[n] + normal_lpdf(R_MAG_SB26[n] |  bR + sinth*logL[n]  + random_realization[n]*sinth_r, R_MAG_SB26_ERR[n])
  //                     + normal_lpdf(V_0p33R26[n]| VtoUse[n], V_0p33R26_err[n]) ;

  //     logexp[2] =  lnpDs2[n] + normal_lpdf(R_MAG_SB26[n] |  bR2 + sinth2*logL2[n]  + random_realization2[n]*sinth2_r, R_MAG_SB26_ERR[n])
  //                     + normal_lpdf(V_0p33R26[n]| VtoUse2[n], V_0p33R26_err[n]) ;
  //     target += log_sum_exp(logexp);
  //   }

  //   // real alpha = 40;
  //   // real omega = 10;
  //   // bR + sinth*logL ~ skew_normal(dwarf_mag, omega, -alpha);
  //   // bR2 + sinth2*logL2 ~ skew_normal(dwarf_mag, omega, alpha);

  //   random_realization2 ~ normal (0, sigR2);
  //   sigR2 ~ cauchy(0.,1);

  //   // sin(atanAR2-atanAR) ~ normal (0,0.5);

  // }

  random_realization ~ cauchy (0, sigR);
  sigR ~ cauchy(0.,1);
 
  // if (angle_error==1)
  //   epsilon ~ normal(0,angle_dispersion);
}
generated quantities {
   real aR=tan(atanAR);
   // real minLogL = min(logL_);
   // real maxLogL = max(logL_);
   // if (pure !=1) 
   //  real aR2=tan(atanAR2);
}