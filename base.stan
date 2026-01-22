// Tully-Fisher Relation (TFR) model with multiple redshift bins
// 
// Data structure:
// - x_lst: list of N absolute magnitude arrays (one per redshift bin)
// - x_unc_lst: list of N absolute magnitude uncertainty arrays (optional)
// - y_lst: list of N log(Vrot/V0) arrays (one per redshift bin)
// - y_unc_lst: list of N log(Vrot/V0) uncertainty arrays (optional)
//
// Parameters:
// - slope: common TFR slope across all redshift bins
// - sigma_int_x: intrinsic scatter in x-axis (absolute magnitude)
// - sigma_int_y: intrinsic scatter in y-axis (log velocity)
// - intercept[i]: TFR intercept for the i-th redshift bin

data {
  // Number of redshift bins
  int<lower=1> N_bins;
  
  // Number of galaxies in each redshift bin
  array[N_bins] int<lower=0> N_gal;
  
  // Total number of galaxies across all bins
  int<lower=0> N_total;
  
  // Absolute magnitude data (flattened array with ragged structure)
  vector[N_total] x;
  
  // Absolute magnitude uncertainties (optional, set to zero if not available)
  vector<lower=0>[N_total] x_unc;
  
  // log(Vrot/V0) data (flattened array with ragged structure)
  vector[N_total] y;
  
  // log(Vrot/V0) uncertainties (optional, set to zero if not available)
  vector<lower=0>[N_total] y_unc;
  
  // Bin assignment for each galaxy (maps galaxy index to redshift bin)
  array[N_total] int<lower=1, upper=N_bins> bin_idx;
}

transformed data {
  // Compute start indices for each bin (for efficient indexing)
  array[N_bins] int start_idx;
  array[N_bins] int end_idx;
  
  start_idx[1] = 1;
  end_idx[1] = N_gal[1];
  
  for (i in 2:N_bins) {
    start_idx[i] = end_idx[i-1] + 1;
    end_idx[i] = start_idx[i] + N_gal[i] - 1;
  }
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
  
}

model {
  // Priors
  slope ~ normal(0, 10);
  intercept ~ normal(0, 10);
  sigma_int_x ~ normal(0, 1);
  sigma_int_y ~ normal(0, 1);
  
  // Loop over all galaxies
  for (i in 1:N_total) {
    int bin = bin_idx[i];
    
    // Measurement model: observed values given true values
    x[i] ~ normal(x_true[i], x_unc[i]);
    y[i] ~ normal(y_true[i], y_unc[i]);
    
    // TFR model: y_true = intercept[bin] + slope * x_true
    // We model x_true with intrinsic scatter and y_true follows the relation
    y_true[i] ~ normal(intercept[bin] + slope * x_true[i], sigma_int_y);
  }
  
  // Prior on true x values (can be adjusted based on expected magnitude range)
  x_true ~ normal(-20, 5);
}