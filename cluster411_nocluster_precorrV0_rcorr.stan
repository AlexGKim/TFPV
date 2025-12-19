// This file contains the modeled position angle update from October 2025
// Is designed for Alex's cuts using [Vmin, Vmax] truncations on the gaussian
// Fits in absolute magnitude space to fit all the data as a single cluster
// And adds a term for the magnitude corrections done in the Hyperfit notebooks

// ./cluster411_nocluster_precorrV0_rcorr sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.99 num_warmup=3000 num_samples=1000 num_chains=1 init=data/iron_cluster_init_zbins_nocluster_v15_rcorr.json data file=data/iron_cluster_zbins_nocluster_v15_rcorr.json output file=/pscratch/sd/s/sgmoore1/stan_outputs/iron/zbins/unclustered/v15/new_pa/rcorr2/cluster_411_.csv

// ./cluster411_nocluster_precorrV0_rcorr sample algorithm=hmc engine=nuts max_depth=17 adapt delta=0.99 num_warmup=3000 num_samples=1000 num_chains=1 init=data/iron_cluster_init_zbins_nocluster_v15_rcorr_intdust.json data file=data/iron_cluster_zbins_nocluster_v15_rcorr_intdust.json output file=/pscratch/sd/s/sgmoore1/stan_outputs/iron/zbins/unclustered/v15/new_pa/rcorr/intdust/cluster_411_.csv


functions {
  vector V_fiber(vector V, vector epsilon) {
    return V./cos(epsilon);
  }
}

data {
  int<lower=0> N;

  int<lower=0> N_cluster;
  array[N_cluster] int N_per_cluster;

  vector[N] V_0p4R26_lognorm;
  vector[N] V_0p4R26_lognorm_err;
  vector[N] R_MAG_SB26;
  vector[N] R_ABSMAG_SB26;
  vector[N] R_MAG_SB26_ERR;
  vector[N] R_correction;
  vector[N] R_correction_err;


  //for iron
  // vector[N_cluster] mu;
  vector[N] mu_all;
  real Rlim;
  vector[N] Rlim_eff;
  real Vmin;
  real Vmax;

  real omega_dist_init;
  real xi_dist_init;
  real V0;
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
      R_[index_]=R_ABSMAG_SB26[index_]-Rlim_eff[i];
      index_=index_+1;
    }
  }
}


// from eyeball look at data expect b ~ -20, a ~ -7
// average logV ~ 2
parameters {
  // vector<lower=-pi()/2/angle_dispersion, upper=pi()/2/angle_dispersion>[N] epsilon_raw;    // angle error. There is a 1/cos so avoid extreme
  // vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] epsilon_unif;   
  vector[N] logL_raw;       // latent parameter
  // if (flatDistribution == 0)
  // {
  // parameters for SkewNormal
  // real<lower=-3, upper=3> alpha_dist;
  real<lower=0.4, upper=2> omega_dist;  
  // real<lower=12, upper=18> xi_dist;
  real<lower=-1.0, upper=2.0> xi_dist;
  // real<lower=1.5, upper=2.7> xi_dist; 
  // }

  real<lower=atan(-10) , upper=atan(-5)> atanAR; // negative slope positive cosine

  vector[N_cluster] bR;
  // vector[N_cluster] bR_offset;

  vector[N] random_realization_raw;
  real<lower=0.01> sigR;
  // real sigR;

  // special case for letting dispersion axis free dispersion_case=5
  real<lower=-pi()/2,upper=pi()/2> theta_2;
}
model {
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
    sinth_r=sin(theta_2); costh_r=cos(theta_2);
  }

  if (flatDistribution==0) {
    logL=omega_dist*logL_raw+xi_dist/costh;
    // logL = omega_dist * logL_raw;
  } else {
    // logL=logL_raw*omega_dist_init + xi_dist_init;
    logL = omega_dist * logL_raw;
  } 

  vector[N] VtoUse = pow(10, costh * logL  + random_realization * costh_r);
  // if (angle_error == 1){
  //   VtoUse = V_fiber(VtoUse,epsilon);
  //  } 

    int N_y = 9;
      vector[N_y] y_int = to_vector({0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9});
      vector[N_y] int_holder;
      real V_holder;
      for (n in 1:N) {
        for (i in 1:N_y) {
            V_holder =  VtoUse[n]/cos(angle_dispersion*tan(pi()*(y_int[i]-0.5)));
            int_holder[i] = normal_lpdf(V_0p4R26_lognorm[n] | V_holder , V_0p4R26_lognorm_err[n]) -
                log_diff_exp(normal_lcdf(Vmax  | V_holder , V_0p4R26_lognorm_err[n]), normal_lcdf(Vmin| V_holder , V_0p4R26_lognorm_err[n]));
          }
        target += log_sum_exp(int_holder);
      }

  vector[N] m_realize = sinth * logL  + random_realization*sinth_r - xi_dist * tan(atanAR);
  vector[N_cluster] a_term = bR - Rlim;
  int index=1;
  for (i in 1:N_cluster){  
    for (j in 1:N_per_cluster[i]){
      m_realize[index]= a_term[i] + m_realize[index];
      index=index+1;
    }
    // R_ ~ normal(m_realize, R_err) T[,Rlim_eff[i]];
  }
  

  R_ ~ normal(m_realize - R_correction, sqrt(R_MAG_SB26_ERR^2 + R_correction_err^2)) T[,mean(mu_all)];

  // V_0p4R26_lognorm ~ normal(VtoUse, V_0p4R26_lognorm_err) T[Vmin,Vmax];

  if (flatDistribution==0)
  {
      // logL_raw ~ skew_normal(0, 1 ,alpha_dist);
      logL_raw ~ normal(0,1);
      target+= -N * log(omega_dist);
  } else {
      logL_raw ~ normal(0, 10);
  }


  random_realization_raw ~ normal (0, 1);

  // if (angle_error==1){
  //   // epsilon_raw ~ cauchy(0,1);
  // }
}
generated quantities {
   real aR=tan(atanAR);
}
