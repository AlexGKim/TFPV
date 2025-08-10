// make ~/Projects/TFPV/fit
// ./fit sample algorithm=hmc engine=nuts num_warmup=500 num_samples=500 num_chains=4 init="data_fit/Y1/fit_init.json"  data file="data_fit/Y1/fit.json" output file="output_fit/Y1/fit.csv"
// $CONDA_PREFIX/bin/cmdstan/bin/stansummary output_fit/Y1/fit_?.csv

data {
    int<lower=1> N; // number of data points
    vector[N] V_0p4R26;
    vector[N] V_0p4R26_ERR;
    vector[N] R_MAG_SB26;
    vector[N] R_MAG_SB26_ERR;

    vector[N] Rlim_eff;
    vector[N] Vlim_eff;

    real Vmin;
    real Vmax;

    real V0;
    real bR0;

    // Don't think this is necessary
    // real xiprodb;

    // Note that these are not the fit parameters in the training fit so they need to be transformed first
    // Alternatively this code could be modified to use the native training parameters

    // vector[6] pop_mn; // in order of  atanAR; bR; sigR; xi_dist; omega_dist; theta_2;
    // matrix[6,6] pop_cov_L; // Cholesky decomposition of covariance matrix
    vector[3] pop_mn; // in order of  atanAR; bR; sigR; xi_dist; omega_dist; theta_2;
    matrix[3,3] pop_cov_L; // Cholesky decomposition of covariance matrix    
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
  vector[N] R_ = R_MAG_SB26 - Rlim_eff;
  vector[N] V_0p4R26_0 = V_0p4R26 / V0;
  vector[N] V_0p4R26_ERR_0 = V_0p4R26_ERR / V0;
  vector[N] Vlim_eff_0 = Vlim_eff / V0;
  real Vmin_0 = Vmin/V0;
  real Vmax_0 = Vmax/V0;


}

parameters {
    vector[N] mu;
    // vector[N] logL_raw;
    vector[N] logL;
    vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] epsilon_unif;   
    vector[N] random_realization_raw;

    // Latent variables that were fit parameters in the training
    array[N] vector[3] alpha;
}


model {

    // training parameters
    for (n in 1:N) {
        alpha[n] ~ std_normal();  
    }
    vector[3] pars;
    vector[N] atanAR;
    // vector[N] bR;
    vector[N] sigR;
    // vector[N] xi_dist;
    // vector<lower=0>[N] omega_dist;
    vector[N] theta_2;
    // Priors on latent variables
    for (n in 1:N) {
        pars = pop_mn + pop_cov_L * alpha[n];
        atanAR[n] = pars[1];
        sigR[n] = pars[2];
        theta_2[n] = pars[3];
    }

    vector[N] epsilon = angle_dispersion * tan(epsilon_unif);
    vector[N] random_realization=random_realization_raw .* sigR;

    vector[N] sinth = sin(atanAR);
    vector[N] costh = cos(atanAR);
    vector[N] sinth_r = sin(theta_2);
    vector[N] costh_r = cos(theta_2);

    // vector[N] logL = omega_dist.*logL_raw + xi_dist./costh;

    vector[N] VtoUse = pow(10, costh .* logL  + random_realization .* costh_r ) ./ cos(epsilon) ;
    // vector[N] m_realize = bR0 + mu - Rlim_eff + sinth .* logL  + random_realization .* sinth_r - xi_dist .* tan(atanAR);
    vector[N] m_realize = mu + bR0  + sinth .* logL  + random_realization .* sinth_r  - Rlim_eff;

    // print(mu," ", R_, " ", m_realize, " ", R_-m_realize, " ", V_0p4R26_0," ",VtoUse);
    // Truncated normal likelihoods
    R_ ~ normal(m_realize, R_MAG_SB26_ERR) T[,0];

    for (n in 1:N) {
        V_0p4R26_0[n] ~ normal(VtoUse[n], V_0p4R26_ERR_0[n]) T[Vmin_0,Vlim_eff_0[n]];
    }

    random_realization_raw ~ normal(0,1);
    // logL_raw ~ normal(0,1);
    // target+= -log(omega_dist);
    target += - log(sigR);

}