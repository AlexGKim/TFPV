// ./base sample data file=TF_mock_input.json output file=output_base.csv

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
  vector[N_total] x_std = (x - mean_x) / sd_x;
  vector[N_total] sigma_x_std = sigma_x / sd_x;
}
parameters {
  // Common slope across all redshift bins
  real slope_std;
  
  // Intercept for each redshift bin
  vector[N_bins] intercept_std;
  
  // Intrinsic scatter in x-direction (absolute magnitude)
  real<lower=0> sigma_int_x_std;
  
  // Intrinsic scatter in y-direction (log velocity)
  real<lower=0> sigma_int_y;
  
  // Underlying (latent) x for each galaxy
  vector<lower=0>[N_total] x_TF_std;
  
  // True (latent) x differences for each galaxy
  // this parameterization is efficient in STAN
  vector[N_total] dx;
  
  // True (latent) y differences for each galaxy
  vector[N_total] dy;
}
transformed parameters {
  vector[N_total] y_TF;
  for (i in 1 : N_total) {
    int bin = bin_idx[i];
    y_TF[i] = intercept_std[bin] + slope_std * x_TF_std[i];
  }
}
model {
  // True (latent) y values for each galaxy
  vector[N_total] x_true_std = x_TF_std + dx * sigma_int_x_std;
  vector[N_total] y_true = y_TF + dy * sigma_int_y;
  
  // Measurement model: observed values given true values
  x ~ normal(x_true_std, sigma_x_std+1e-8);
  y ~ normal(y_true, sigma_y+1e-8);
  
  dx ~ std_normal();
  dy ~ std_normal();
  
  // Priors
  // It is standard practice to use half-Cauchy priors for dispersion parameters
  sigma_int_x_std ~ cauchy(0, 10);
  sigma_int_y ~ cauchy(0, 10);
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
}
