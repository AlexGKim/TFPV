// ./coma sample algorithm=hmc engine=nuts max_depth=16 adapt delta=0.95 num_warmup=1000 num_samples=1000 num_chains=4 init=data/SGA-2020_fuji_Vrot_init.json data file=data/SGA-2020_fuji_Vrot.json output file=output/fuji_210_test.csv

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
  int dispersion_case=2;

  int pure = 1;
  int angle_error = 0;

  real mu_coma=34.7;

  real dwarf_mag=-17. + 34.7;

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 5.;
  real angle_dispersion = angle_dispersion_deg/180*pi();

}

parameters {
  // vector<lower=0, upper=pi()/4>[N] epsilon;    // angle error. There is a 1/cos so avoid extreme


  // population 1
  // vector[N] logL;       // latent parameter

  vector[N] v_raw;
  real<lower=0> s_dist;
  real<lower=0> scale_dist;

  real<lower=pi()*(1./2.+1./32.), upper=pi()*2./3.> atanAR;
  real bR;

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

  vector[N] v = scale_dist*v_raw;
  vector[N] logL = log10(v)/costh;

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

  // velocity model with or without axis error
  vector[N] VtoUse = pow(10, costh*logL  + (random_realization)*costh_r );
  // if (angle_error == 1){
  //     VtoUse = V_fiber(VtoUse,epsilon);
  // } 

  if (pure == 1)
  {
    R_MAG_SB26 ~ normal(bR + mu_coma+ sinth*logL  + (random_realization)*sinth_r, R_MAG_SB26_ERR);
    V_0p33R26 ~ normal(VtoUse, V_0p33R26_err);
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
  v_raw ~ lognormal(0, s_dist);
  random_realization ~ normal (0, sigR);
  sigR ~ cauchy(0.,1);
 
  // if (angle_error==1)
  //   epsilon ~ normal(0,angle_dispersion);
}
generated quantities {
   real aR=tan(atanAR);
   // if (pure !=1) 
   //  real aR2=tan(atanAR2);
}