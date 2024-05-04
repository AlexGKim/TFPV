// ./cluster sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.999 num_warmup=2000 num_samples=1000 num_chains=4 init=data/iron_cluster_init.json data file=data/iron_cluster.json output file=output/cluster_410.csv

functions {
  vector V_fiber(vector V, vector epsilon) {
    return V./cos(epsilon);
  }
}

data {
  int<lower=0> N;

  int<lower=0> N_cluster;
  array[N_cluster] int N_per_cluster;

  vector[N] V_0p4R26;
  vector[N] V_0p4R26_err;
  vector[N] R_MAG_SB26;
  vector[N] R_MAG_SB26_ERR;

  // vector[N] Rhat;
  // vector[N] Vhat;
  // real Rhat_noise;
  // vector[N] Vhat_noise;  

  //for iron
  vector[N_cluster] mu;
  real Rlim;
  vector[N_cluster] Rlim_eff;
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
  int angle_error = 1;

  int flatDistribution = 0;


  // real dwarf_mag=-17. + 34.7;

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 10.;
  real angle_dispersion = angle_dispersion_deg/180*pi();

}


// from eyeball look at data expect b ~ -7.1, a ~ -6.1
// average logV ~ 2.14
parameters {
  vector<lower=-pi()/4, upper=pi()/4>[N] epsilon;    // angle error. There is a 1/cos so avoid extreme

  vector[N] logL_raw;       // latent parameter
  // if (flatDistribution == 0)
  // {
  // parameters for SkewNormal
  real<lower=-5, upper=0> alpha_dist;
  real<lower=0.2, upper=2> omega_dist;  
  real<lower=12, upper=18> xi_dist;
  // }

  real<lower=atan(-8) , upper=atan(-5.5)> atanAR; // negative slope positive cosine

  vector[N_cluster] bR;
  // vector[N_cluster] bR_offset;

  vector[N] random_realization_raw;
  real<lower=0> sigR;
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
  vector[N] VtoUse = pow(10, costh*logL  + random_realization*costh_r );
  if (angle_error == 1){
      VtoUse = V_fiber(VtoUse,epsilon);
  } 

  // vector[N] m_realize;
  int index=1;
  for (i in 1:N_cluster){
    vector[N_per_cluster[i]] R_;
    vector[N_per_cluster[i]] R_err;
    vector[N_per_cluster[i]] m_realize;    
    for (j in 1:N_per_cluster[i]){
      m_realize[j]= bR[i] +  mu[i]+ sinth * logL[index]  + random_realization[index]*sinth_r;
      R_[j]=R_MAG_SB26[index];
      R_err[j] = R_MAG_SB26_ERR[index];
      index=index+1;
    }
    R_ ~ normal(m_realize, R_err) T[,Rlim_eff[i]];
  }
  // m_realize = bR + m_realize;


  V_0p4R26 ~ normal(VtoUse, V_0p4R26_err) T[Vmin,];

  if (flatDistribution==0)
  {
      logL_raw ~ skew_normal(0, 1 ,alpha_dist);
  } else {
      logL_raw ~ normal(0, 10);
  }

  random_realization_raw ~ normal (0, 1);
  sigR ~ cauchy(0.,10);

  // bR_offset ~ normal(0,100);

  if (angle_error==1){
    epsilon ~ normal(0,angle_dispersion);
  }
}
generated quantities {
   real aR=tan(atanAR);
}
