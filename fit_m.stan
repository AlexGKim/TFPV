// make ~/Projects/TFPV/fit
// ./fit_m sample algorithm=hmc engine=nuts num_warmup=500 num_samples=500 num_chains=4 init="data_fit/unclustered/fit_init.json"  data file="data_fit/unclustered/fit.json" output file="output_fit/unclustered/fit.csv"
// $CONDA_PREFIX/bin/cmdstan/bin/stansummary output_fit/Y1/fit_?.csv

data {
    int<lower=1> N; // number of data points
    vector[N] V_0p4R26;
    vector[N] V_0p4R26_ERR;
    vector[N] R_MAG_SB26;
    vector[N] R_MAG_SB26_ERR;

    vector[N] Rlim_eff;
    vector[N] Vlim_eff;
    vector[N] Vlim_min;

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
    // vector[5] pop_mn; // in order of  atanAR; bR; sigR; xi_dist; omega_dist; theta_2;
    // matrix[5,5] pop_cov_L; // Cholesky decomposition of covariance matrix

    // Random realization of TF relation and other parameters to integrate over
    int N_s;
    vector[N_s] atanAR;
    vector[N_s] sigR;
    vector[N_s] theta_2;
    vector[N_s] random_realization;
    vector[N_s] epsilon_unif;
}

transformed data {

  // 4 cases
  // 1 : line fit only
  // 2 : log-V dispersion
  // 3 : mag dispersion
  // 4 : perp dispersion
  // 5 : free NOTE HARD MODEL IS HARDWIRED TO 4!!!!!

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
  vector[N] Vlim_min_0 = Vlim_min / V0;

  real Vmin_0 = Vmin/V0;
  // real Vmax_0 = Vmax/V0;

  // V_0p4R26_0 = V_0p4R26_0 - Vmin_0;
  // Vmax_0 = Vmax_0 - Vmin_0;

  vector[N_s] sinth = sin(atanAR);
  vector[N_s] costh = cos(atanAR);
  vector[N_s] random_sinth_r = -costh .* random_realization;
  vector[N_s] random_costh_r = sinth .* random_realization;    
  vector[N_s] cos_epsilon = cos(angle_dispersion * tan(epsilon_unif));
}

parameters {
    vector<lower=32, upper=40> [N] mu;
    vector[N] logL;
}

model {
    vector[N_s] logexp;
    vector[N] VtoUse;
    vector[N] m_realize;
    for (m in 1:N_s) {
        
        VtoUse = pow(10, costh[m] * logL  + random_costh_r[m] ) ./ cos_epsilon[m] ;
        m_realize = mu + bR0  + sinth[m] * logL  + random_sinth_r[m]  - Rlim_eff;

        // Truncated normal likelihoods
        logexp[m] = normal_lpdf(R_ | m_realize, R_MAG_SB26_ERR) - normal_lcdf(0 | m_realize, R_MAG_SB26_ERR)
                + normal_lpdf(V_0p4R26_0 | VtoUse, V_0p4R26_ERR_0) 
                - normal_lccdf(Vlim_min_0 | VtoUse, V_0p4R26_ERR_0)
                - normal_lcdf(Vlim_eff_0 | VtoUse, V_0p4R26_ERR_0);        
    }
    target += log_sum_exp(logexp);

}

// generated quantities {
//    vector[N] V_TF ;
//    for (n in 1:N) {
//     V_TF[n]=0;
//    }
//    for (m in 1:N_s) {
//         V_TF = V_TF + V0 * pow(10, cos(posterior_samples[m,1]) .* logL ) ./ cos(angle_dispersion * tan(posterior_samples[m,5])) ;
//     }
//     V_TF = V_TF/N_s;
// }