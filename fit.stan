// make ~/Projects/TFPV/fit
// ./fit sample algorithm=hmc engine=nuts num_chains=4 data file="data_fit/Y1/fit.json" output file="output_fit/Y1/fit.csv"

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

    vector[7] pop_mn; // in order of theta_1 : tan(atanAR), theta_2, b : bR, sigR,  logL0 :xi_dist/cos(theta_1), sigma_logL0 (omega_dist)  
    matrix[7,7] pop_cov_L; // Cholesky decomposition of covariance matrix
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
    vector[N] logL;
    vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] delta_phi_unif;   
    vector[N] epsilon;

    // Latent variables that were fit parameters in the training
    vector[N] theta_1;
    vector[N] theta_2;
    vector[N] b;
    vector[N] sigR;
    vector[N] logL0;
    vector<lower=0>[N] sigma_logL0;
}

model {

    vector[7] pars;
    vector[N] delta_phi = angle_dispersion * tan(delta_phi_unif);
    real V_mod;
    real m_mod;    

    for (n in 1:N) {
        // Priors on latent variables
        logL[n] ~ normal(logL0[n], sigma_logL0[n]);
        epsilon[n] ~ normal(0, sigR[n]);

        // Likelihoods
        // Compute the deterministic functions
        V_mod = pow(10, cos(theta_1[n]) * logL[n] + cos(theta_2[n]) * epsilon[n]) * (1.0 / cos(delta_phi[n]));
        m_mod = mu[n] + b[n] + sin(theta_1[n]) * logL[n] + sin(theta_2[n]) * epsilon[n];

        // Truncated normal likelihoods
        R_ ~ normal(m_mod - Rcut[n], R_MAG_SB26_ERR) T[,0];
        V_0p4R26_0 ~ normal(V_mod, V_0p4R26_ERR_0) T[Vmin_0,Vmax_0];

        pars[1] = theta_1[n];
        pars[2] = theta_2[n];
        pars[3] = b[n];
        pars[4] = sigR[n];
        pars[5] = logL0[n];
        pars[6] = sigma_logL0[n];
        pars ~ multi_normal_cholesky(pop_mn, pop_cov_L);
    }
}