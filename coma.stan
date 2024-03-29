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

  int pure = 1;
  int angle_error = 0;

  real dwarf_mag=-17.;

}

parameters {
  vector[N] epsilon;    // angle error
  vector <lower=0, upper=1>[N] pD;   // dwarf population fraction

  // population 1
  vector[N] logL;       // latent parameter
  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  real bR;

  vector[N] random_realization;
  real<lower=0> sigR;

  // population 2
  vector[N] logL2;       // latent parameter
  real<lower=0, upper=pi()> atanAR2;
  real bR2;

  vector[N] random_realization2;
  real<lower=0> sigR2;

}

model {
  // dispersion axis
  real sinth; real costh; real sinth2; real costh2; 
  if (dispersion_case == 1)
  {
    sinth=0; costh=0; sinth2=0; costh2=0;
  }
  else if (dispersion_case ==2)
  {
    sinth=1; costh=0; sinth2=1; costh2=0;
  }
  else if (dispersion_case ==3)
  {
    sinth=0; costh=1; sinth2=0; costh2=1;
  }
  else if (dispersion_case==4)
  {
    sinth=sin(atanAR); costh=cos(atanAR); sinth2=sin(atanAR2); costh2=cos(atanAR2);
  }

  // velocity model with or without axis error
  vector[N] VtoUse = pow(10, costh*logL  + (random_realization)*sinth );
  if (angle_error == 1){
      VtoUse = V_fiber(VtoUse,epsilon);
  } 

  if (pure == 1)
  {
    R_MAG_SB26 ~ normal(bR + sinth*logL  - (random_realization)*costh, R_MAG_SB26_ERR);
    V_0p33R26 ~ normal(VtoUse, V_0p33R26_err);
  } else
  {
    real lnpDs1 = log(1-pD);
    real lnpDs2 = log(pD);
    vector[N] VtoUse2 = pow(10, costh2*logL2  + (random_realization2)*sinth2 );
    if (angle_error == 1){
        VtoUse2 = V_fiber(VtoUse2,epsilon);
    } 
    vector[2] logexp;
    for (n in 1:N)
    {
      logexp[1] =  lnpDs1[n] + normal_lpdf(R_MAG_SB26[n] |  bR + sinth*logL[n]  - random_realization[n]*costh, R_MAG_SB26_ERR[n])
                      + normal_lpdf(V_0p33R26[n]| VtoUse[n], V_0p33R26_err[n]) ;

      logexp[2] =  lnpDs2[n] + normal_lpdf(R_MAG_SB26[n] |  bR2 + sinth2*logL2[n]  - random_realization2[n]*costh2, R_MAG_SB26_ERR[n])
                      + normal_lpdf(V_0p33R26[n]| VtoUse2[n], V_0p33R26_err[n]) ;
      target += log_sum_exp(logexp);
    }
    bR + sinth*logL ~ uniform(dwarf_mag-10, dwarf_mag);
    bR2 + sinth2*logL2 ~ uniform(dwarf_mag, dwarf_mag+10);
  }

  random_realization ~ normal (0, sigR);
  sigR ~ cauchy(0.,1);
 
  random_realization2 ~ normal (0, sigR2);
  sigR2 ~ cauchy(0.,1);


  epsilon ~ normal(0,pi()/64.);
}
generated quantities {
   real aR=tan(atanAR);
   real aR2=tan(atanAR2);
}