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
  int<lower=1> N_bins;
  
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
  array[N_total] int<lower=1, upper=N_bins> bin_idx;
}
// standardizing predictor variable
transformed data {
  real mean_x = mean(x);
  real sd_x = sd(x);
  real sd_y = sd(y);
  vector[N_total] x_std = (x - mean_x) / sd_x;
  vector[N_total] sigma_x_std = sigma_x / sd_x;
  real y_lb = min(y);
  real y_ub = max(y) - 0.01; // small buffer below max
  
  real haty_max = max(y); // in implementation y_ub > haty_max is requred
  
  // run configuration parameters
  int y_TF_limits = 1;
  int y_selection = 1;
  
  int fit_sigmas = 0;
  real theta_int; // if fit_sigmas ==0
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
  // real<lower=0, upper=pi() / 2> theta_int; // partitioning angle between x and y
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
  vector[N_total] yfromxstd = intercept_std[1] + slope_std * x_std;
  vector[N_total] sigmasq1_std = square(sigma_int_x_std)
                                 + square(sigma_x_std);
  vector[N_total] sigmasq2 = square(sigma_int_y) + square(sigma_y);
  vector[N_total] sigmasq_tot = square(slope_std) * sigmasq1_std + sigmasq2;
  
  if (y_TF_limits == 0) {
    // No prior limits; without selection
    y ~ normal(yfromxstd, sqrt(sigmasq_tot));
    target += log(abs(slope_std)) * N_total;
  }
  
  if (y_TF_limits != 0) {
    vector[N_total] mu_star = (yfromxstd .* sigmasq2
                               + y * square(slope_std) .* sigmasq1_std)
                              ./ sigmasq_tot;
    
    vector[N_total] sqrt_sigmasq_star = sqrt(
                                             (square(slope_std)
                                              * sigmasq1_std .* sigmasq2)
                                             ./ sigmasq_tot);
    
    y ~ normal(yfromxstd, sqrt(sigmasq_tot));
    target += log(abs(slope_std)) * N_total;
    // Prior limits without selection      
    target += log_diff_exp(normal_lcdf(y_ub | mu_star, sqrt_sigmasq_star),
                           normal_lcdf(y_lb | mu_star, sqrt_sigmasq_star));
    if (y_selection != 0) {
      vector[N_total] sigma2 = sqrt(sigmasq2);
      
      // standard‑normal arguments for the lower‑ and upper‑bound CDFs
      vector[N_total] z_lb = (haty_max - y_lb) ./ sigma2;
      vector[N_total] z_ub = (haty_max - y_ub) ./ sigma2;
      
      // log‑CDF (normal_lcdf) – note that normal_lcdf = log(Phi)
      real log_lcdf_lb = normal_lcdf(z_lb | 0, 1);
      real log_lcdf_ub = normal_lcdf(z_ub | 0, 1);
      
      // log‑PDF (normal_lpdf) – explicit normal‑density formula
      real log_lpdf_lb = normal_lpdf(z_lb | 0, 1);
      real log_lpdf_ub = normal_lpdf(z_ub | 0, 1);
      
      // log‑normalising constants (the “lnZ” terms)
      vector[N_total] lnZ_lb = log_sum_exp(
                                           log(haty_max - y_lb) + log_lcdf_lb,
                                           log(sigma2) + log_lpdf_lb);
      
      vector[N_total] lnZ_ub = log_sum_exp(
                                           log(haty_max - y_ub) + log_lcdf_ub,
                                           log(sigma2) + log_lpdf_ub);
      
      // add the whole contribution to the target in one go
      target += -sum(log_diff_exp(lnZ_lb, lnZ_ub));
    }
  }
  // Priors
  // It is standard practice to use half-normal priors for dispersion parameters
  // sigma_int_x_std ~ cauchy(0, 5);
  sigma_int_tot_y ~ cauchy(0, 0.03 * 100);
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
  real sigma_int_x = sigma_int_x_std * sd_x;
}
