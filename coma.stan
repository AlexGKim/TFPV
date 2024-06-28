// ./coma311 sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.9999 num_warmup=1000 num_samples=1000 num_chains=4 init=data/SGA-2020_fuji_Vrot_cuts_init.json data file=data/SGA-2020_fuji_Vrot_cuts.json output file=output/fuji_311.csv
// ./coma411 sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.9999 num_warmup=1000 num_samples=1000 num_chains=4 init=data/SGA-2020_fuji_Vrot_cuts_init.json data file=data/SGA-2020_fuji_Vrot_cuts.json output file=output/fuji_411.csv


functions {
  vector V_fiber(vector V, vector epsilon) {
    return V./cos(epsilon);
  }
}

data {
  int<lower=0> N;
  vector[N] V_0p33R26;
  vector[N] V_0p33R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;

  // vector[N] Rhat;
  // vector[N] Vhat;
  // real Rhat_noise;
  // vector[N] Vhat_noise;  

  real Vmin;
}

transformed data {
  // 4 cases
  // 1 : line fit only
  // 2 : log-V dispersion
  // 3 : mag dispersion
  // 4 : perp dispersion
  // 5 : free dispersion
  int dispersion_case=3;

  int pure = 1;
  int angle_error = 1;

  int flatDistribution = 0;

  real mu_coma=34.7;

  real dwarf_mag=-17. + 34.7;

  // vector[N] dR = sqrt(R_MAG_SB26_ERR.*R_MAG_SB26_ERR + Rhat_noise*Rhat_noise);
  // vector[N] dV = sqrt(V_0p33R26_err.*V_0p33R26_err + Vhat_noise.*Vhat_noise);

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_distribution_nu = 0.8483460681555207;
  real angle_dispersion_deg = 4.217219770730775;
  real angle_dispersion = angle_dispersion_deg/180*pi();

}

parameters {
  // vector<lower=-pi()/2/angle_dispersion, upper=pi()/2/angle_dispersion>[N] epsilon_raw;    // angle error. There is a 1/cos so avoid extreme
  // vector<lower=-atan(pi()/2/angle_dispersion), upper=atan(pi()/2/angle_dispersion)>[N] epsilon_unif;   
  vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] epsilon_unif;   
  // population 1
  vector[N] logL_raw;       // latent parameter

  // if (flatDistribution == 0)
  // {
  // parameters for SkewNormal
  // real<lower=-10, upper=10> alpha_dist;
  real<lower=0.5, upper=5> omega_dist;  // this is in logL
  real<lower=1.5, upper=2.7> xi_dist;   // this is in logV
  // }

 real<lower=atan(-11.5) , upper=atan(-5.)> atanAR; // negative slope positive cosine
  real bR;

  vector[N] random_realization_raw;
  real<lower=0.0> sigR;
}
model {
  // vector[N] epsilon_raw = tan(epsilon_unif);
  vector[N] epsilon = angle_dispersion * tan(epsilon_unif);
  // vector[N] logL = sigma_dist*(logL_raw+mu_dist);

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

  vector[N] logL;
  if (flatDistribution==0) {
    logL=omega_dist*logL_raw+xi_dist/costh;
  } else {
    logL=logL_raw*2.2831016215521247 + 14.913405242237685;
  } 

  // velocity model with or without axis error
  vector[N] VtoUse = pow(10, costh*logL  + (random_realization)*costh_r );
  if (angle_error == 1){
      VtoUse = V_fiber(VtoUse,epsilon);
  } 

  // Rhat ~ normal(bR + mu_coma+ sinth * logL  + (random_realization)*sinth_r, dR);
  // Vhat ~ normal(VtoUse, dV) T[Vmin,];

  R_MAG_SB26 ~ normal(bR - xi_dist * tan(atanAR) + mu_coma+ sinth * logL  + (random_realization)*sinth_r, R_MAG_SB26_ERR);
  V_0p33R26 ~ normal(VtoUse, V_0p33R26_err) T[Vmin,];
  
  if (flatDistribution==0)
  {
      // logL_raw ~ skew_normal(0, 1 ,alpha_dist);
      logL_raw ~ normal(0,1.);
      target += -N*log(omega_dist);
  } else{
     logL_raw ~ normal(0, 10);
  }

  random_realization_raw ~ normal (0, 1);
  target += -N*log(sigR);
  // sigR ~ cauchy(0.,10);
 
  // if (angle_error==1){
  //   // epsilon_raw ~ cauchy(0, 1);
  //   // epsilon_unif ~ uniform(-pi() / 2, pi() / 2);)
  // }
}
generated quantities {
   real aR=tan(atanAR);
}