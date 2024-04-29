// ./iron410 sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.99 num_warmup=1000 num_samples=1000 num_chains=4 init=data/SGA-2020_iron_Vrot_cuts_sub_0.10_init.json data file=data/SGA-2020_iron_Vrot_cuts_sub_0.10.json output file=output/iron_410_cuts_sub_0.10.csv


// functions {
//   vector V_fiber(vector V, vector epsilon) {
//     return V./cos(epsilon);
//   }
// }

data {
  int<lower=0> N;
  vector[N] V_0p4R26;
  vector[N] V_0p4R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;

  vector[N] Rhat;
  vector[N] Vhat;
  real Rhat_noise;
  vector[N] Vhat_noise;  

  //for iron
  vector[N] mu;
  vector[N] dm_v;
  real Rlim;
  vector[N] Rlim_eff;
  real Vmin;
  real Vmax;

  real omega_dist_init;
  real xi_dist_init;
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

  int flatDistribution = 0;

  vector[N] dR = sqrt(R_MAG_SB26_ERR.*R_MAG_SB26_ERR + Rhat_noise*Rhat_noise);
  vector[N] dV = sqrt(V_0p4R26_err.*V_0p4R26_err + Vhat_noise.*Vhat_noise);

  // real dwarf_mag=-17. + 34.7;

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  // real angle_dispersion_deg = 5.;
  // real angle_dispersion = angle_dispersion_deg/180*pi();

}


// from eyeball look at data expect b ~ -7.1, a ~ -6.1
// average logV ~ 2.14
parameters {
  // vector<lower=0, upper=pi()/4>[N] epsilon;    // angle error. There is a 1/cos so avoid extreme

  vector[N] logL_raw;       // latent parameter
  // if (flatDistribution == 0)
  // {
  // parameters for SkewNormal
  real<lower=-10, upper=0> alpha_dist;
  real<lower=0.5, upper=4> omega_dist;  
  real<lower=12, upper=18> xi_dist;
  // }

  real<lower=atan(-9) , upper=atan(-5.5)> atanAR; // negative slope positive cosine

  real bR;

  vector[N] random_realization_raw;
  real<lower=0> sigR;

  vector[N] dv;
}
model {


  vector[N] logL;
  if (flatDistribution==0) {
    logL=omega_dist*logL_raw+xi_dist;
  } else {
    logL=logL_raw*omega_dist_init + xi_dist_init;
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
  vector[N] m_realize = bR + mu+ sinth * logL  + (random_realization)*sinth_r + dm_v.*dv;

  Rhat ~ normal(m_realize, dR) T[,Rlim];
  Vhat ~ normal(VtoUse, dV) T[Vmin,Vmax];

  if (flatDistribution==0)
  {
      logL_raw ~ skew_normal(0, 1 ,alpha_dist);
  }

  random_realization_raw ~ normal (0, 1);
  sigR ~ cauchy(0.,1);

  dv ~ normal(0.,1.);

  // if (angle_error==1)
  //   epsilon ~ normal(0,angle_dispersion);
}
generated quantities {
   real aR=tan(atanAR);
}
