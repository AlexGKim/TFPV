// ./base sample num_samples=500 num_chains=4 data file=TF_mock_input.json init=TF_mock_init.json output file=output_base.csv
// ../cmdstan/bin/stansummary output_base_?.csv -i slope -i intercept.1 -i sigma_int_x -i sigma_int_y
// ../cmdstan/bin/diagnose output_base*.csv

// Tully-Fisher Relation (TFR) model with multiple redshift bins
// 
// Data structure:
// - x_lst: list of N absolute magnitude arrays (one per redshift bin)
// - sigma_x_lst: list of N absolute magnitude uncertainty arrays (optional)
// - y_lst: list of N log(Vrot/V0) arrays (one per redshift bin)
// - sigma_y_lst: list of N log(Vrot/V0) uncertainty arrays (optional)
//
// Parameters:
// - slope: common TFR slope across all redshift bins
// - sigma_int_x: intrinsic scatter in x-axis (absolute magnitude)
// - sigma_int_y: intrinsic scatter in y-axis (log velocity)
// - intercept[i]: TFR intercept for the i-th redshift bin

data {
  // Number of redshift bins
  int<lower=1> N_bins; // For the momment N_bins = 1
  
  // Total number of galaxies across all bins
  int<lower=0> N_total;
  
  // Absolute magnitude data (flattened array with ragged structure)
  vector[N_total] x;
  
  // Absolute magnitude uncertainties (optional, set to zero if not available)
  vector<lower=0>[N_total] sigma_x;
  
  // log(Vrot/V0) data (flattened array with ragged structure)
  vector[N_total] y;
  
  // log(Vrot/V0) uncertainties (optional, set to zero if not available)
  vector<lower=0>[N_total] sigma_y;
  
  // Bin assignment for each galaxy (maps galaxy index to redshift bin)
  // array[N_total] int<lower=1, upper=N_bins> bin_idx;
}
// standardizing predictor variable
transformed data {
  real mean_x = mean(x);
  real sd_x = sd(x);
  real sd_y = sd(y);
  vector[N_total] x_std = (x - mean_x) / sd_x;
  vector[N_total] sigma_x_std = sigma_x / sd_x;
  
  // properties of dataset
  real y_lb = -23; //-23.361639168868468; // min(y) + 0.09;  FROM ARIEL FEB 2 2026
  real y_ub = -15; //-14.623998117629371; // max(y) - 0.09; // small buffer below max
  real haty_max = -17;
  
  // variables used in more complicaed models
  real log_lb = log(haty_max - y_lb);
  real log_ub = log(y_ub - haty_max);
  vector[N_total] sigma_x_std_sq = square(sigma_x_std);
  vector[N_total] sigma_y_sq = square(sigma_y);
  
  int bin_idx = 1;
  
  // run configuration parameters
  int y_TF_limits = 1;
  int y_selection = 1;
  
  int fit_sigmas = 1;
  // real theta_int; // if fit_sigmas ==0
}
parameters {
  // Common slope across all redshift bins
  real<lower=-12 * sd_x, upper=-4 * sd_x> slope_std;
  
  // Intercept for each redshift bin
  vector<upper=0>[N_bins] intercept_std;
  
  // Intrinsic scatter in x-direction (absolute magnitude)
  // real<lower=0> sigma_int_x_std;
  
  // Intrinsic scatter in y-direction (log velocity)
  // real<lower=0> sigma_int_y;
  
  // Reparameterized intrinsic scatter
  real<lower=0> sigma_int_tot_y; // total intrinsic scatter (projected to y)
  // if fit_sigmas != 0
  real<lower=0, upper=pi() / 2> theta_int; // partitioning angle between x and y
}
transformed parameters {
  real sigma_int_y;
  real sigma_int_x_std;
  if (fit_sigmas == 0) {
    sigma_int_y = sigma_int_tot_y;
    sigma_int_x_std = sigma_int_y / sd_x;
  } else {
    sigma_int_x_std = sigma_int_tot_y * cos(theta_int) / abs(slope_std);
    sigma_int_y = sigma_int_tot_y * sin(theta_int);
  }
}
model {
  // likelihood given flat prior in y_TF
  vector[N_total] yfromxstd = intercept_std[bin_idx] + slope_std * x_std;
  vector[N_total] sigmasq1_std = square(sigma_int_x_std) + sigma_x_std_sq;
  vector[N_total] sigmasq2 = square(sigma_int_y) + sigma_y_sq;
  // vector[N_total] sigmasq_tot = square(slope_std) * sigmasq1_std + sigmasq2;
  vector[N_total] sigmasq_tot = square(slope_std)
                                * (square(sigma_int_x_std) + sigma_x_std_sq)
                                + (square(sigma_int_y) + sigma_y_sq);
  
  if (y_TF_limits == 0) {
    // No prior limits; without selection
    y ~ normal(yfromxstd, sqrt(sigmasq_tot));
    target += log(abs(slope_std)) * N_total;
  }
  
  if (y_TF_limits != 0) {
    vector[N_total] mu_star = (yfromxstd .* sigmasq2
                               + y * square(slope_std) .* sigmasq1_std)
                              ./ sigmasq_tot;
    
    vector[N_total] sqrt_sigmasq_star = abs(slope_std)
                                        * sqrt(
                                               (sigmasq1_std .* sigmasq2)
                                               ./ sigmasq_tot);
    
    y ~ normal(yfromxstd, sqrt(sigmasq_tot));
    target += log(abs(slope_std)) * N_total;
    
    // containers used for multiple purposes
    vector[N_total] term_lb;
    vector[N_total] term_ub;
    
    // Term for the TFR limits
    for (n in 1 : N_total) {
      term_lb[n] = normal_lcdf(y_lb | mu_star[n], sqrt_sigmasq_star[n]);
      term_ub[n] = normal_lcdf(y_ub | mu_star[n], sqrt_sigmasq_star[n]);
    }
    target += log_diff_exp(term_ub, term_lb); // done with this use of term_lb/ub
    
    // Term for the selection function
    if (y_selection != 0) {
      // vector[N_total] sigma2 = sqrt(sigmasq2);
      vector[N_total] sigma2 = sqrt(square(sigma_int_y) + sigma_y_sq);

      term_lb = (haty_max - y_lb) / sigma2;
      term_ub = (y_ub - haty_max) / sigma2;
      
      vector[N_total] logsigma2 = 0.5 * log(square(sigma_int_y) + sigma_y_sq);
      
      // standard‑normal arguments for the lower‑ and upper‑bound CDFs
      vector[3] lse_terms;
      for (n in 1 : N_total) {
        lse_terms[1] = log_lb + log(Phi_approx(term_lb[n]));
        lse_terms[2] = logsigma2[n] + std_normal_lpdf(term_lb[n]);
        lse_terms[3] = log_ub + log(Phi_approx(term_ub[n]));
        term_lb[n] = log_sum_exp(lse_terms);
        term_ub[n] = logsigma2[n] + std_normal_lpdf(term_ub[n]);
      }
      
      target += -log_diff_exp(term_lb, term_ub);
    }
  }
  
  // Priors
  // It is standard practice to use half-normal priors for dispersion parameters
  // sigma_int_x_std ~ cauchy(0, 0.03 * 100 / sd_x);
  // sigma_int_y ~ cauchy(0, 0.03 * 100);
  sigma_int_tot_y ~ cauchy(0, 0.03 * 1000);
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
  real sigma_int_x = sigma_int_x_std * sd_x;
}
