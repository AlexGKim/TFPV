// functions {
//   vector V_fiber(vector V, vector epsilon) {
//     return V./cos(epsilon);
//   }
// }

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

  int pure = 0;
  int angle_error = 0;

  real dwarf_mag=-17. + 34.7;

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 5.;
  real angle_dispersion = angle_dispersion_deg/180*pi();

  vector[N] logV = log10(V_0p33R26);

}

parameters {
  // vector<lower=0, upper=pi()/4>[N] epsilon;    // angle error. There is a 1/cos so avoid extreme


  // population 1
  vector[N] logL;       // latent parameter
  real<lower=-pi()/2, upper= 0 > atanAR;
  real bR;

  vector[N] random_realization;
  real<lower=0> sigR;

  // population 2
  // real<lower=0, upper=0.15> pD;   // dwarf population fraction
  // simplex[2] pD;
  vector[N] logL2;       // latent parameter
  // real<lower=pi()/4, upper=5*pi()/4> atanAR2;
  real<lower=-pi()/2, upper=0> atanAR2;
  real bR2;

  vector[N] random_realization2;
  real<lower=0> sigR2;

}
model {
  // slope of TF Relation
  real sinth = sin(atanAR);
  real costh = cos(atanAR);

  real sinth2 = sin(atanAR2);
  real costh2 = cos(atanAR2);

  vector[N] logL_ = logL+logV;
  vector[N] logL2_ = logL2+logV;

  // slope of redsidual dispersion
  real sinth_r; real costh_r; real sinth2_r; real costh2_r; 
  if (dispersion_case == 1)
  {
    sinth_r=0; costh_r=0;  sinth2_r=0; costh2_r=0;
  }
  else if (dispersion_case ==2)
  {
    sinth_r=1; costh_r=0; sinth2_r=1; costh2_r=0;
  }
  else if (dispersion_case ==3)
  {
    sinth_r=0; costh_r=1; sinth2_r=0; costh2_r=1;
  }
  else if (dispersion_case==4)
  {
    sinth_r=-costh; costh_r=sinth; sinth2_r=-costh2; costh2_r=sinth2;
  }

  // velocity model with or without axis error
  vector[N] VtoUse = pow(10, costh*logL_  + random_realization*costh_r );
  // if (angle_error == 1){
  //     VtoUse = V_fiber(VtoUse,epsilon);
  // } 

  if (pure == 1)
  {
    R_MAG_SB26 ~ normal(bR + sinth*logL_  + random_realization*sinth_r, R_MAG_SB26_ERR);
    V_0p33R26 ~ normal(VtoUse, V_0p33R26_err);
  }
  else
  {
    real pD=0.05;
    real lnpDs1 = log(1-pD);
    real lnpDs2 = log(pD);
    vector[N] VtoUse2 = pow(10, costh2*logL2_  + random_realization2*costh2_r );
    // if (angle_error == 1){
    //     VtoUse2 = V_fiber(VtoUse2,epsilon);
    // } 
    vector[2] logexp;
    for (n in 1:N)
    {
      logexp[1] =  lnpDs1 + normal_lpdf(R_MAG_SB26[n] |  bR + sinth*logL_[n]  + random_realization[n]*sinth_r, R_MAG_SB26_ERR[n])
                      + normal_lpdf(V_0p33R26[n]| VtoUse[n], V_0p33R26_err[n]) ;

      logexp[2] =  lnpDs2 + normal_lpdf(R_MAG_SB26[n] |  bR2 + sinth2*logL2_[n]  + random_realization2[n]*sinth2_r, R_MAG_SB26_ERR[n])
                      + normal_lpdf(V_0p33R26[n]| VtoUse2[n], V_0p33R26_err[n]) ;
      // print(pD," ",logexp);
      target += log_sum_exp(logexp);
    }

    // real alpha = 40;
    // real omega = 10;
    // bR + sinth*logL ~ skew_normal(dwarf_mag, omega, -alpha);
    // bR2 + sinth2*logL2 ~ skew_normal(dwarf_mag, omega, alpha);

    random_realization2 ~ normal (0, sigR2);
    sigR2 ~ cauchy(0.,1);
    // print((atanAR2-atanAR)*180/pi());
    // print(costh*mean(logL_)," ", bR+sinth*mean(logL_), " ", costh2*mean(logL2_)," ", bR2+sinth2*mean(logL2_));
    // sin(atanAR2-atanAR) ~ normal (0,0.5);

  }

  random_realization ~ normal (0, sigR);
  sigR ~ cauchy(0.,1);

  // this light constraint helps!
  logL ~ normal(0,10/costh);
  logL2 ~ normal(0,10/costh2);
 
  // if (angle_error==1)
  //   epsilon ~ normal(0,angle_dispersion);
}
generated quantities {
  real aR=tan(atanAR);
   // if (pure !=1) 
  real aR2=tan(atanAR2);
}