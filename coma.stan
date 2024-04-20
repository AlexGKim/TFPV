// ./coma210f sample algorithm=hmc engine=nuts max_depth=18 adapt delta=0.99 num_warmup=1000 num_samples=1000 num_chains=4 init=data/SGA-2020_fuji_Vrot_init.json data file=data/SGA-2020_fuji_Vrot.json output file=output/fuji_210f.csv

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
  // 5 : free dispersion
  int dispersion_case=4;

  int pure = 1;
  int angle_error = 0;

  int flatDistribution = 0;

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
  vector[N] logL_raw;       // latent parameter

  // if (flatDistribution == 0)
  // {
  // parameters for SkewNormal
  real alpha_dist;
  real<lower=0> omega_dist;
  real xi_dist;
  // }

  real<lower=-pi()*(.5-1./32) , upper=-pi()*1./3> atanAR; // negative slope positive cosine
  real bR;

  vector[N] random_realization_raw;
  real<lower=0> sigR;
}
model {

  // vector[N] logL = sigma_dist*(logL_raw+mu_dist);
  vector[N] logL;
  if (flatDistribution==0) {
    logL=omega_dist*(logL_raw+xi_dist);
  } else {
    logL=logL_raw*1.5160651053079683 + 13.133570672711606;
  } 
  vector[N] random_realization=random_realization_raw*sigR;
  real sinth = sin(atanAR);
  real costh = cos(atanAR);

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
  else if (dispersion_case==5)
  {
    // sinth_r=sin(atanARr); costh_r=cos(atanARr); //sinth2_r=-costh2; costh2_r=sinth2;
  }

  // velocity model with or without axis error
  vector[N] VtoUse = pow(10, costh*logL  + (random_realization)*costh_r );
  if (angle_error == 1){
      // VtoUse = V_fiber(VtoUse,epsilon);
  } 

  R_MAG_SB26 ~ normal(bR + mu_coma+ sinth * logL  + (random_realization)*sinth_r, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(VtoUse, V_0p33R26_err);
  
  if (flatDistribution==0)
  {
      logL_raw ~ skew_normal(0, 1 ,alpha_dist);
  }

  random_realization_raw ~ normal (0, 1);
  sigR ~ cauchy(0.,1);
 
  // if (angle_error==1)
  //   epsilon ~ normal(0,angle_dispersion);
}
generated quantities {
   real aR=tan(atanAR);
}