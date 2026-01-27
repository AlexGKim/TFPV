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
  real x_sd_lb = min(x_std)*1.1;
  real x_sd_ub = max(x_std)*1.1;
}
parameters {
  // Common slope across all redshift bins
  real<lower=-12 * sd_x, upper=-4 * sd_x> slope_std;
  
  // Intercept for each redshift bin
  vector<upper=0>[N_bins] intercept_std;
  
  // Intrinsic scatter in x-direction (absolute magnitude)
  real<lower=0> sigma_int_x_std;
  
  // Intrinsic scatter in y-direction (log velocity)
  real<lower=0> sigma_int_y;
  
  // Underlying (latent) x for each galaxy
  // vector[N_total] x_TF_std_stdnormal;
  
  // vector[N_total] delta_x_TF_std;
  // vector[N_total] delta_y_TF;
  
  // // Underlying x_TF_std distribution parameters
  // vector[N_total] x_TF_std;
}
model {
  // marginalize x_TF_std flat
  vector[N_total] sigmasq1_std = square(sigma_int_x_std)
                                 + square(sigma_x_std);
  vector[N_total] sigmasq2 = square(sigma_int_y) + square(sigma_y);
  
  vector[N_total] mu_star = (x_std .* sigmasq2
                             + slope_std * (y - intercept_std[1])
                               .* sigmasq1_std)
                            ./ (sigmasq2 + square(slope_std) * sigmasq1_std);
  vector[N_total] sqrt_sigmasq_star = sqrt((sigmasq1_std .* sigmasq2)
                                 ./ (sigmasq2
                                     + square(slope_std) * sigmasq1_std));
  vector[N_total] sigmasq_tot = square(slope_std) * sigmasq1_std + sigmasq2;

  y ~ normal(intercept_std[1] + slope_std * x_std,
               sqrt(sigmasq_tot));
  target += log_diff_exp(normal_lcdf(x_sd_ub|mu_star , sqrt_sigmasq_star),normal_lcdf(x_sd_lb|mu_star , sqrt_sigmasq_star));

  // vector[N_total] x_TF_std = mu_x_TF_std + sigma_x_TF_std * x_TF_std_stdnormal;
  // vector[N_total] y_TF = intercept_std[1] + slope_std * x_TF_std;
  // // //   for (i in 1 : N_total) {
  // // //     int bin = bin_idx[i];
  // // //     y_TF[i] = intercept_std[bin] + slope_std * x_TF_std[i];
  // // //   }
  // // }
  
  // // Measurement model: observed values given true values
  // x_std ~ normal(x_TF_std, sqrt(sigma_int_x_std^2 + sigma_x_std^2));
  // y ~ normal(y_TF, sqrt(sigma_int_y ^ 2 + sigma_y ^ 2));
  
  // x_std ~ normal(x_TF_std + delta_x_TF_std * sigma_int_x_std, sigma_x_std);
  // y ~ normal(y_TF + delta_y_TF * sigma_int_y, sigma_y);
  // delta_x_TF_std ~ std_normal();
  // delta_y_TF ~ std_normal();
  
  // Priors
  // It is standard practice to use half-normal priors for dispersion parameters
  sigma_int_x_std ~ normal(0, 5);
  sigma_int_y ~ normal(0, 5 * sd_y);
  // x_TF_std_stdnormal ~ std_normal();
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
  real sigma_int_x = sigma_int_x_std * sd_x;
}
