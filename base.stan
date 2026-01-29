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
  real binormal_cdf_diff(real zmin, real zmax, real w, real rho) {
    real sqrt1mrho2 = sqrt((1 - rho) * (1 + rho));
    
    // Phi(w) cancels exactly!
    real phi_diff = 0.5 * (Phi(zmax) - Phi(zmin));
    
    // First T terms: different first arguments
    real a_max = (w - rho * zmax) / (zmax * sqrt1mrho2);
    real a_min = (w - rho * zmin) / (zmin * sqrt1mrho2);
    real T_diff_1 = owens_t(zmax, a_max) - owens_t(zmin, a_min);
    
    // Second T terms: same first argument, use integral form
    real b_max = (zmax - rho * w) / (w * sqrt1mrho2);
    real b_min = (zmin - rho * w) / (w * sqrt1mrho2);
    real T_diff_2 = owens_t_diff(w, b_min, b_max); // Custom function
    
    // Delta terms
    real delta_max = (zmax * w < 0) || (zmax * w == 0 && (zmax + w) < 0);
    real delta_min = (zmin * w < 0) || (zmin * w == 0 && (zmin + w) < 0);
    
    return phi_diff - T_diff_1 - T_diff_2 - 0.5 * (delta_max - delta_min);
  }
  
  real owens_t_diff(real h, real a_lo, real a_hi) {
    // Direct integration avoids subtraction
    real mid = 0.5 * (a_hi + a_lo);
    real half_width = 0.5 * (a_hi - a_lo);
    real h2 = square(h);
    
    array[5] real nodes = {-0.906179845938664, -0.538469310105683, 0.0,
                           0.538469310105683, 0.906179845938664};
    array[5] real weights = {0.236926885056189, 0.478628670499366,
                             0.568888888888889, 0.478628670499366,
                             0.236926885056189};
    
    real result = 0;
    for (i in 1 : 5) {
      real t = mid + half_width * nodes[i];
      result += weights[i] * exp(-0.5 * h2 * (1 + square(t)))
                / (1 + square(t));
    }
    
    return half_width * result / (2 * pi());
  }
}
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
  real y_ub = max(y) + 0.2;
  
  real haty_max = max(y);
  real theta_int = pi() / 4; // initial guess
  
  int y_TF_limits = 1;
  int y_selection = 0;
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
  // real<lower=0, upper=pi() / 2> theta_int; // partitioning angle between x and y
}
transformed parameters {
  // real sigma_int_x_std = sigma_int_tot_y * cos(theta_int) / abs(slope_std);
  // real sigma_int_y = sigma_int_tot_y * sin(theta_int);
  real sigma_int_y = sigma_int_tot_y;
  real sigma_int_x_std = sigma_int_y / sd_x;
}
model {
  // likelihood given flat prior in y_TF
  vector[N_total] yfromxstd = intercept_std[1] + slope_std * x_std;
  vector[N_total] sigmasq1_std = square(sigma_int_x_std)
                                 + square(sigma_x_std);
  vector[N_total] sigmasq2 = square(sigma_int_y) + square(sigma_y);
  vector[N_total] sigmasq_tot = square(slope_std) * sigmasq1_std + sigmasq2;
  
  if (y_TF_limits == 0) {
    if (y_selection == 0) {
      // No prior limits; without selection
      y ~ normal(yfromxstd, sqrt(sigmasq_tot));
    } else {
      // No prior limits; with selection
      y ~ normal(yfromxstd, sqrt(sigmasq_tot)) T[ , haty_max];
    }
  }
  
  if (y_TF_limits != 0) {
    vector[N_total] mu_star = (yfromxstd .* sigmasq2
                               + y * square(slope_std) .* sigmasq1_std)
                              ./ sigmasq_tot;
    
    vector[N_total] sqrt_sigmasq_star = sqrt(
                                             (square(slope_std)
                                              * sigmasq1_std .* sigmasq2)
                                             ./ sigmasq_tot);
    if (y_selection == 0) {
      // Prior limits without selection
      y ~ normal(yfromxstd, sqrt(sigmasq_tot));
      
      target += log_diff_exp(normal_lcdf(y_ub | mu_star, sqrt_sigmasq_star),
                             normal_lcdf(y_lb | mu_star, sqrt_sigmasq_star));
    } else {
      // Prior limits with selection
      // Extra terms needed for sample selection
      vector[N_total] sigma_tot = sqrt(sigmasq_tot);
      vector[N_total] sigma1_std = sqrt(sigmasq1_std);
      
      vector[N_total] zmin = (y_lb - yfromxstd) / abs(slope_std)
                             ./ sigma1_std;
      vector[N_total] zmax = (y_ub - yfromxstd) / abs(slope_std)
                             ./ sigma1_std;
      vector[N_total] w = (haty_max - yfromxstd) / abs(slope_std)
                          ./ sigma_tot;
      vector[N_total] rho = abs(slope_std) * sigma1_std ./ sigma_tot;
      for (n in 1 : N_total) {
        // // print(zmin[n], w[n], zmax[n], rho[n]);
        tuple(real, real) zminw = (zmin[n], w[n]);
        tuple(real, real) zmaxw = (zmax[n], w[n]);
        // print((binormal_lcdf(zmaxw | rho[n]), binormal_lcdf(zminw | rho[n])));
        // target += -log_diff_exp(binormal_lcdf(zmaxw | rho[n]),
        //                         binormal_lcdf(zminw | rho[n]));
        print(log(binormal_cdf(zmaxw | rho[n])), " ",
              log(binormal_cdf(zminw | rho[n])));
        
        target += -log_diff_exp(log(binormal_cdf(zmaxw | rho[n])),
                                log(binormal_cdf(zminw | rho[n])));
      }
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
