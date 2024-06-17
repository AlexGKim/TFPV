// ./cluster311 sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.999 num_warmup=1000 num_samples=1000 num_chains=4 init=data/iron_cluster_init.json data file=data/iron_cluster.json output file=output/cluster_311.csv
// ./cluster411 sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.999 num_warmup=2000 num_samples=1000 num_chains=4 init=data/iron_cluster_init.json data file=data/iron_cluster.json output file=output/cluster_411.csv

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
  int dispersion_case=3;

  int pure = 1;
  int angle_error = 1;

  int flatDistribution = 0;


  // real dwarf_mag=-17. + 34.7;

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 4.217219770730775;
  real angle_dispersion = angle_dispersion_deg/180*pi();

  // shifted data to align to common magnitude cutoff.  Allows vectorization
  vector[N] R_;
  int index_=1;
  for (i in 1:N_cluster){ 
    for (j in 1:N_per_cluster[i]){
      R_[index_]=R_MAG_SB26[index_]-Rlim_eff[i];
      index_=index_+1;
    }
  }
}


// from eyeball look at data expect b ~ -7.1, a ~ -6.1
// average logV ~ 2.14
parameters {
  vector<lower=-pi()/2/angle_dispersion, upper=pi()/2/angle_dispersion>[N] epsilon_raw;    // angle error. There is a 1/cos so avoid extreme

  vector[N] logL_raw;       // latent parameter
  // if (flatDistribution == 0)
  // {
  // parameters for SkewNormal
  // real<lower=-3, upper=3> alpha_dist;
  real<lower=0.4, upper=2> omega_dist;  
  // real<lower=12, upper=18> xi_dist;
  real<lower=1.5, upper=2.7> xi_dist; 
  // }

  real<lower=atan(-8) , upper=atan(-5)> atanAR; // negative slope positive cosine

  vector[N_cluster] bR;
  // vector[N_cluster] bR_offset;

  vector[N] random_realization_raw;
  real<lower=0.01> sigR;
}
model {
  vector[N] epsilon=epsilon_raw*angle_dispersion;
  vector[N] logL;

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

  if (flatDistribution==0) {
    logL=omega_dist*logL_raw+xi_dist/costh;
  } else {
    logL=logL_raw*omega_dist_init + xi_dist_init;
  } 

  // velocity model with or without axis error
  vector[N] VtoUse = pow(10, costh*logL  + random_realization*costh_r );
  if (angle_error == 1){
      VtoUse = V_fiber(VtoUse,epsilon);
  } 

  vector[N] m_realize = sinth * logL  + random_realization*sinth_r - xi_dist * tan(atanAR);
  vector[N_cluster] a_term = bR + mu - Rlim_eff;
  int index=1;
  for (i in 1:N_cluster){  
    for (j in 1:N_per_cluster[i]){
      m_realize[index]= a_term[i] + m_realize[index];
      index=index+1;
    }
    // R_ ~ normal(m_realize, R_err) T[,Rlim_eff[i]];
  }
  // m_realize = bR + m_realize;

  R_ ~ normal(m_realize, R_MAG_SB26_ERR) T[,0];
  V_0p4R26 ~ normal(VtoUse, V_0p4R26_err) T[Vmin,];

  if (flatDistribution==0)
  {
      // logL_raw ~ skew_normal(0, 1 ,alpha_dist);
      logL_raw ~ normal(0,1);
      target+= -N * log(omega_dist);
  } else {
      logL_raw ~ normal(0, 10);
  }


  random_realization_raw ~ normal (0, 1);
  target += - N * log(sigR);

  if (angle_error==1){
    epsilon_raw ~ cauchy(0,1);
  }
}
generated quantities {
   real aR=tan(atanAR);
}
