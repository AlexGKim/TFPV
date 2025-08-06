// make ~/Projects/TFPV/fit
// ./fit sample algorithm=hmc engine=nuts num_warmup=500 num_samples=500 num_chains=4 init="data_fit/Y1/fit_init.json"  data file="data_fit/Y1/fit.json" output file="output_fit/Y1/fit.csv"
data {
    int<lower=1> N; // number of data points
    vector[N] V_0p4R26;
    vector[N] V_0p4R26_ERR;
    vector[N] R_MAG_SB26;
    vector[N] R_MAG_SB26_ERR;

    vector[N] Rlim_eff;
    real Vmin;
    real Vmax;

    real V0;
    real bR0;

    // Note that these are not the fit parameters in the training fit so they need to be transformed first
    // Alternatively this code could be modified to use the native training parameters

    // vector[6] pop_mn; // in order of  atanAR; bR; sigR; xi_dist; omega_dist; theta_2;
    // matrix[6,6] pop_cov_L; // Cholesky decomposition of covariance matrix
    vector[5] pop_mn; // in order of  atanAR; bR; sigR; xi_dist; omega_dist; theta_2;
    matrix[5,5] pop_cov_L; // Cholesky decomposition of covariance matrix    
}

transformed data {

  // 4 cases
  // 1 : line fit only
  // 2 : log-V dispersion
  // 3 : mag dispersion
  // 4 : perp dispersion
  // 5 : free NOTE HARD MODEL IS HARDWIRED TO 5!!!!!

  // int dispersion_case=5; 

  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 4.217219770730775;
  real angle_dispersion = angle_dispersion_deg/180*pi();

  // shifted data to align to common magnitude cutoff.  Allows vectorization
  vector[N] R_ =R_MAG_SB26-Rlim_eff;
  vector[N] V_0p4R26_0 = V_0p4R26 / V0;
  vector[N] V_0p4R26_ERR_0 = V_0p4R26_ERR / V0;
  real Vmin_0 = Vmin/V0;
  real Vmax_0 = Vmax/V0;
}

parameters {
    vector[N] mu;
    vector[N] logL_raw;
    vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] epsilon_unif;   
    vector[N] random_realization_raw;

    // Latent variables that were fit parameters in the training
    vector[N] atanAR;
    // vector[N] bR;
    vector<lower=0>[N] sigR;
    vector[N] xi_dist;
    vector<lower=0>[N] omega_dist;
    vector[N] theta_2;
}


model {
    vector[N] epsilon = angle_dispersion * tan(epsilon_unif);

    vector[N] sinth = sin(atanAR);
    vector[N] costh = cos(atanAR);
    vector[N] sinth_r = sin(theta_2);
    vector[N] costh_r = cos(theta_2);

    vector[N] logL = omega_dist.*logL_raw + xi_dist./costh;
    vector[N] random_realization=random_realization_raw .* sigR;

    vector[N] VtoUse = pow(10, costh .* logL  + random_realization .* costh_r ) ./ cos(epsilon) ;
    vector[N] m_realize = bR0 + mu - Rlim_eff + sinth .* logL  + random_realization .* sinth_r - xi_dist .* tan(atanAR);


    // Truncated normal likelihoods
    R_ ~ normal(m_realize, R_MAG_SB26_ERR) T[,0];
    V_0p4R26_0 ~ normal(VtoUse, V_0p4R26_ERR_0) T[Vmin_0,Vmax_0];

    // CONTAINERS
    vector[5] pars;
    for (n in 1:N) {
        // Priors on latent variables
        pars[1] = atanAR[n];
        // pars[2] = bR[n];
        pars[2] = sigR[n];
        pars[3] = xi_dist[n];
        pars[4] = omega_dist[n];
        pars[5] = theta_2[n];
        pars ~ multi_normal_cholesky(pop_mn, pop_cov_L);

        // print(normal_lpdf(R_[n]| m_mod - Rlim_eff[n], R_MAG_SB26_ERR[n]), " ", normal_lpdf(V_0p4R26_0[n] | V_mod, V_0p4R26_ERR_0[n])," ", multi_normal_cholesky_lpdf(pars | pop_mn, pop_cov_L));
    }
    random_realization_raw ~ normal(0,1);
    logL_raw ~ normal(0,1);
    target+= -log(omega_dist);
    target += - log(sigR);    
}