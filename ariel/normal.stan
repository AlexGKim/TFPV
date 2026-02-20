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
// - sigma_int_x: intrinsic scatter in x-axis (log velocity)
// - sigma_int_y: intrinsic scatter in y-axis (absolute magnitude)
// - intercept[i]: TFR intercept for the i-th redshift bin

functions {
  real binormal_cdf(tuple(real, real) z, real rho) {
    real z1 = z.1;
    real z2 = z.2;
    if (z1 == 0 && z2 == 0) {
      return 0.25 + asin(rho) / (2 * pi());
    }
    real denom = sqrt((1 + rho) * (1 - rho));
    real term1 = z1 == 0 ? (z2 > 0 ? 0.25 : -0.25)
                 : owens_t(z1, (z2 / z1 - rho) / denom);
    real term2 = z2 == 0 ? (z1 > 0 ? 0.25 : -0.25)
                 : owens_t(z2, (z1 / z2 - rho) / denom);
    real z1z2 = z1 * z2;
    real delta = z1z2 < 0 || (z1z2 == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) + Phi(z2) - delta) - term1 - term2;
  }
  
  real P_binormal_strip(real mu_y_TF,
                        real tau, // SD of y_TF  (i.e., y_TF ~ Normal(mu_y_TF, Sigma))
                        real haty_max,
                        real s,
                        real c,
                        real s_plane, // \bar{s}
                        real c1_plane, // \bar{c}_1
                        real c2_plane, // \bar{c}_2
                        real sigma1, // \sigma_{1,i}
                        real sigma2 // \sigma_{2,i}
                        ) {
    real delta_c = c2_plane - c1_plane;
    real a;
    real mu_yhat;
    real mu_u;
    real var_yhat;
    real var_u;
    real cov_yhat_u;
    real rho;
    real beta;
    real gamma0;
    real gamma1;
    
    // empty / invalid strip
    if (delta_c <= 0) 
      return 0;
    
    // a = (1 - \bar{s}/s)
    a = 1 - s_plane / s;
    
    // Means
    mu_yhat = mu_y_TF;
    mu_u = a * mu_y_TF + (s_plane / s) * c - c1_plane;
    
    // Variances and covariance
    var_yhat = square(tau) + square(sigma2);
    var_u = square(a) * square(tau) + square(sigma2)
            + square(s_plane * sigma1);
    cov_yhat_u = a * square(tau) + square(sigma2);
    
    // Correlation (clamp away from exactly +/-1 for numerical stability)
    rho = cov_yhat_u / sqrt(var_yhat * var_u);
    rho = fmin(1 - 1e-12, fmax(-1 + 1e-12, rho));
    
    // Standardized limits
    beta = (haty_max - mu_yhat) / sqrt(var_yhat);
    gamma0 = (0 - mu_u) / sqrt(var_u);
    gamma1 = (delta_c - mu_u) / sqrt(var_u);
    
    // Probability of the strip under bivariate normal
    return binormal_cdf((beta, gamma1) | rho)
           - binormal_cdf((beta, gamma0) | rho);
  }
}
data {
  // Number of redshift bins
  int<lower=1> N_bins; // For the momment N_bins = 1
  
  // Total number of galaxies across all bins
  int<lower=0> N_total;
  
  // Absolute magnitude data (flattened array with ragged structure)
  vector[N_total] y;
  
  // Absolute magnitude uncertainties (optional, set to zero if not available)
  vector<lower=0>[N_total] sigma_y;
  
  // log(Vrot/V0) data (flattened array with ragged structure)
  vector[N_total] x;
  
  // log(Vrot/V0) uncertainties (optional, set to zero if not available)
  vector<lower=0>[N_total] sigma_x;
  
  // Selection function parameter
  real haty_max;
  real slope_plane;
  real intercept_plane;
  real intercept_plane2;
  
  // Properties of dataset
  real<upper=haty_max> y_min;
  real<lower=haty_max> y_max;
  
  real mu_y_TF;
  real<lower=0> tau;
  
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
  
  // variables used in more complicated models
  real log_lb = log(haty_max - y_min);
  real log_minus_ub = log(y_max - haty_max);
  vector[N_total] sigma_x_std_sq = square(sigma_x_std); //////// DEBUG
  vector[N_total] sigma_y_sq = square(sigma_y);
  
  int bin_idx = 1;
  
  // run configuration parameters
  int y_TF_limits = 1;
  int y_selection = 1;
  int plane_cut = 1;
  
  int fit_sigmas = 1;
  // real theta_int; // if fit_sigmas ==0
  
  // for now put the slice cut here
  
  real slope_plane_std = slope_plane * sd_x;
  real intercept_plane_std = intercept_plane
                             + slope_plane_std * mean_x / sd_x;
  real intercept_plane2_std = intercept_plane2
                              + slope_plane_std * mean_x / sd_x;
}
parameters {
  // Common slope across all redshift bins
  real<lower=-18 * sd_x, upper=-4 * sd_x> slope_std;
  
  // Intercept for each redshift bin
  
  vector<lower=-24 + slope_std * mean_x / sd_x,
         upper=-14 + slope_std * mean_x / sd_x>[N_bins] intercept_std;
  
  // Intrinsic scatter in x-direction (absolute magnitude)
  real<lower=0, upper=1> sigma_int_x; // in x-units
  real<lower=0, upper=40> sigma_int_y; // in y-units
}
transformed parameters {
  // real sigma_int_y;
  real sigma_int_x_std;
  if (fit_sigmas == 0) {
    sigma_int_x_std = sigma_int_y / sd_x;
  } else {
    sigma_int_x_std = sigma_int_x / sd_x;
  }
}
model {
  // likelihood given Gaussian prior y_TF ~ Normal(mu_y_TF, tau)
  vector[N_total] yfromxstd = intercept_std[bin_idx] + slope_std * x_std;
  vector[N_total] sigmasq1_std = square(sigma_int_x_std) + sigma_x_std_sq;
  vector[N_total] sigmasq2 = square(sigma_int_y) + sigma_y_sq;
  vector[N_total] sigmasq_tot = square(slope_std)
                                * (square(sigma_int_x_std) + sigma_x_std_sq)
                                + (square(sigma_int_y) + sigma_y_sq);
  
  // Bivariate normal likelihood: (x_std_i, y_i) ~ N2(mu_i, Sigma_i)
  // where y_TF ~ N(mu_y_TF, tau) is integrated out analytically.
  //
  // In standardized x-coords the covariance matrix is:
  //   Sigma_i[1,1] = tau^2/slope_std^2 + sigmasq1_std_i
  //   Sigma_i[1,2] = tau^2/slope_std
  //   Sigma_i[2,2] = tau^2 + sigmasq2_i
  //
  // mean: mu_x_std = (mu_y_TF - intercept_std[bin_idx]) / slope_std
  //        mu_y    = mu_y_TF
  {
    real tausq = square(tau);
    real inv_slope = inv(slope_std);
    real mu_x_std_prior = (mu_y_TF - intercept_std[bin_idx]) * inv_slope;
    vector[2] mu_prior = [mu_x_std_prior, mu_y_TF]';
    
    for (n in 1 : N_total) {
      matrix[2, 2] Sigma_i;
      Sigma_i[1, 1] = tausq * square(inv_slope) + sigmasq1_std[n];
      Sigma_i[1, 2] = tausq * inv_slope;
      Sigma_i[2, 1] = tausq * inv_slope;
      Sigma_i[2, 2] = tausq + sigmasq2[n];
      
      target += multi_normal_lpdf([x_std[n], y[n]]' | mu_prior, Sigma_i);
      
      target += -log(
                     P_binormal_strip(mu_y_TF, tau, haty_max, slope_std,
                                      intercept_std[bin_idx],
                                      slope_plane_std, intercept_plane_std,
                                      intercept_plane2_std,
                                      sqrt(sigmasq1_std[n]),
                                      sqrt(sigmasq2[n])));
    }
  }

  // Priors
  sigma_int_x ~ cauchy(0, 0.3);
  sigma_int_y ~ cauchy(0, 2);
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
  // real sigma_int_x = sigma_int_x_std * sd_x;
}
