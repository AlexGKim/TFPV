// make ~/Projects/TFPV/fit
// ./fit sample algorithm=hmc engine=nuts num_chains=4 init="data_fit/Y1/fit_init.json"  data file="data_fit/Y1/fit.json" output file="output_fit/Y1/fit.csv" 

data {
    int<lower=1> N; // number of data points
    vector[N] V_0p4R26;
    vector[N] V_0p4R26_ERR;
    vector[N] R_MAG_SB26;
    vector[N] R_MAG_SB26_ERR;

    vector[N] Rcut;
    real Vmin;
    real Vmax;

    real V0;

    // Note that these are not the fit parameters in the training fit so they need to be transformed first
    // Alternatively this code could be modified to use the native training parameters

    vector[6] pop_mn; // in order of  atanAR; bR; sigR; xi_dist; omega_dist; theta_2;
    matrix[6,6] pop_cov_L; // Cholesky decomposition of covariance matrix
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
  vector[N] R_ =R_MAG_SB26-Rcut;
  vector[N] V_0p4R26_0 = V_0p4R26 / V0;
  vector[N] V_0p4R26_ERR_0 = V_0p4R26_ERR / V0;
  real Vmin_0 = Vmin/V0;
  real Vmax_0 = Vmax/V0;
}

parameters {
    vector[N] mu;
    vector[N] logL_raw;
    vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] delta_phi_unif;   
    vector[N] epsilon;

    // Latent variables that were fit parameters in the training
    vector[N] atanAR;
    vector[N] bR;
    vector<lower=0>[N] sigR;
    vector[N] xi_dist;
    vector<lower=0>[N] omega_dist;
    vector[N] theta_2;
}


model {
    vector[N] sinth = sin(atanAR);
    vector[N] costh = cos(atanAR);
    vector[N] sinth_2 = sin(theta_2);
    vector[N] costh_2 = cos(theta_2);

    vector[N] logL = omega_dist.*logL_raw + xi_dist./costh;
    vector[N] cos_delta_phi = cos(angle_dispersion * tan(delta_phi_unif));
    vector[6] pars;
    real V_mod;
    real m_mod;

    for (n in 1:N) {
        // Priors on latent variables
        epsilon[n] ~ normal(0, sigR[n]);

        // Likelihoods
        // Compute the deterministic functions
        V_mod = pow(10, costh[n] * logL[n] + costh_2[n] * epsilon[n]) * (1.0 / cos_delta_phi[n]);
        m_mod = mu[n] + bR[n] + sinth[n] * logL[n] + sinth_2[n] * epsilon[n];

        // Truncated normal likelihoods
        R_[n] ~ normal(m_mod - Rcut[n], R_MAG_SB26_ERR[n]) T[,0];
        V_0p4R26_0[n] ~ normal(V_mod, V_0p4R26_ERR_0[n]) T[Vmin_0,Vmax_0];

        pars[1] = atanAR[n];
        pars[2] = bR[n];
        pars[3] = sigR[n];
        pars[4] = xi_dist[n];
        pars[5] = omega_dist[n];
        pars[6] = theta_2[n];
        pars ~ multi_normal_cholesky(pop_mn, pop_cov_L);

        // print(normal_lpdf(R_[n]| m_mod - Rcut[n], R_MAG_SB26_ERR[n]), " ", normal_lpdf(V_0p4R26_0[n] | V_mod, V_0p4R26_ERR_0[n])," ", multi_normal_cholesky_lpdf(pars | pop_mn, pop_cov_L));
    }
    logL_raw ~ normal(0,1);
    target+= -log(omega_dist);
}