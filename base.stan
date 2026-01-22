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
parameters {
  // Common slope across all redshift bins
  real slope;
  
  // Intercept for each redshift bin
  vector[N_bins] intercept;
  
  // Intrinsic scatter in x-direction (absolute magnitude)
  real<lower=0> sigma_int_x;
  
  // Intrinsic scatter in y-direction (log velocity)
  real<lower=0> sigma_int_y;
  
  // Underlying (latent) x for each galaxy
  vector<lower=0>[N_total] x_TF;
  
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
    y_TF[i] = intercept[bin] + slope * x_TF[i];
  }
}
model {
  // True (latent) y values for each galaxy
  vector[N_total] x_true = x_TF + dx * sigma_int_x;
  vector[N_total] y_true = y_TF + dy * sigma_int_y;
  
  // Measurement model: observed values given true values
  x ~ normal(x_true, sigma_x);
  y ~ normal(y_true, sigma_y);
  
  dx ~ std_normal();
  dy ~ std_normal();
  
  // Priors
  // It is standard practice to use half-Cauchy priors for dispersion parameters
  sigma_int_x ~ cauchy(0, 10);
  sigma_int_y ~ cauchy(0, 10);
}
