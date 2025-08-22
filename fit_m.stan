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
    int N_s;
    array[N_s] vector[5] posterior_samples;
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
  vector[N] Vlim_min_0 = Vlim_min / V0;

  real Vmin_0 = Vmin/V0;
  // real Vmax_0 = Vmax/V0;

  // V_0p4R26_0 = V_0p4R26_0 - Vmin_0;
  // Vmax_0 = Vmax_0 - Vmin_0;
}

parameters {
    vector<lower=30, upper=42> [N] mu;
    vector[N] logL;
}

model {
    real sinth;
    real costh ;
    real sinth_r;
    real costh_r;    
    real random_realization;
    real epsilon;


    for (m in 1:N_s) {
        sinth = sin(posterior_samples[m,1]);
        costh = cos(posterior_samples[m,1]);
        // sinth_r = sin(posterior_samples[m,3]);
        // costh_r = cos(posterior_samples[m,3]);
        sinth_r = - costh;
        costh_r = sinth;
        random_realization=posterior_samples[m,4]* posterior_samples[m,2];

        epsilon = angle_dispersion * tan(posterior_samples[m,5]);
        
        // vector[N] logL = omega_dist.*logL_raw + xi_dist./costh;
        vector[N] VtoUse = pow(10, costh .* logL  + random_realization .* costh_r ) ./ cos(epsilon) ;
        // vector[N] m_realize = bR0 + mu - Rlim_eff + sinth .* logL  + random_realization .* sinth_r - xi_dist .* tan(atanAR);
        vector[N] m_realize = mu + bR0  + sinth .* logL  + random_realization .* sinth_r  - Rlim_eff;

        // print(mu," ", R_, " ", m_realize, " ", R_-m_realize, " ", V_0p4R26_0," ",VtoUse);
        // Truncated normal likelihoods
        R_ ~ normal(m_realize, R_MAG_SB26_ERR) T[,0];

        for (n in 1:N) {
            V_0p4R26_0[n] ~ normal(VtoUse[n], V_0p4R26_ERR_0[n]) T[Vlim_min_0[n],Vlim_eff_0[n]];
            // V_0p4R26_0[n] ~ normal(VtoUse[n] - Vmin_0, V_0p4R26_ERR_0[n]) T[0,Vlim_eff_0[n]];
        }
        // print(V_0p4R26_0[1] - Vmin_0, " ", VtoUse[1] - Vmin_0," ",(VtoUse[1]-V_0p4R26_0[1])/V_0p4R26_ERR_0[1], " ",cos(epsilon), " ",random_realization .* costh_r );

        // logL ~ normal(xi_dist, omega_dist);

        // random_realization_raw ~ normal(0,1);
        // target += - N*log(posterior_samples[m,2]);
    }

}

generated quantities {
   vector[N] V_TF ;
   for (n in 1:N) {
    V_TF[n]=0;
   }
   for (m in 1:N_s) {
        V_TF = V_TF + V0 * pow(10, cos(posterior_samples[m,1]) .* logL ) ./ cos(angle_dispersion * tan(posterior_samples[m,5])) ;
    }
    V_TF = V_TF/N_s;
}