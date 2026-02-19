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
  real integrate_binormal_trapez(real y_min,
                                 real y_max,
                                 real haty_max,
                                 real s,
                                 real c,
                                 real s_plane,
                                 real c_plane,
                                 real sigma1,
                                 real sigma2,
                                 int N) {
    real h = (y_max - y_min) / N;
    real sigma_tot = sqrt(sigma2 ^ 2 + s_plane ^ 2 * sigma1 ^ 2);
    real rho = sigma2 / sigma_tot;
    real alpha_n;
    real beta_n;
    real sum;
    
    // Evaluate integrand at left endpoint
    alpha_n = (c_plane - (y_min - s_plane * (y_min - c) / s)) / sigma_tot;
    beta_n = (haty_max - y_min) / sigma2;
    sum = binormal_cdf((-alpha_n, beta_n) | -rho);
    
    // Evaluate integrand at interior points
    for (n in 1 : (N - 1)) {
      real y_TF = y_min + n * h;
      alpha_n = (c_plane - (y_TF - s_plane * (y_TF - c) / s)) / sigma_tot;
      beta_n = (haty_max - y_TF) / sigma2;
      sum += 2.0 * binormal_cdf((-alpha_n, beta_n) | -rho);
    }
    
    // Evaluate integrand at right endpoint
    alpha_n = (c_plane - (y_max - s_plane * (y_max - c) / s)) / sigma_tot;
    beta_n = (haty_max - y_max) / sigma2;
    sum += binormal_cdf((-alpha_n, beta_n) | -rho);
    
    // Trapezoidal rule with top-hat normalization
    return (h / 2.0) * sum / (y_max - y_min);
  }
  
  // Two-sided (parallel) half-plane strip: c1 <= y - s_plane*x <= c2, plus y <= haty_max
  real integrate_binormal_strip_trapez(
         real y_min,
         real y_max,
         real haty_max,
         real s,
         real c,
         real s_plane,
         real c1_plane,
         real c2_plane,
         real sigma1,
         real sigma2,
         int N
       ) {
    real h = (y_max - y_min) / (N - 1);
    
    real sigma_tot = sqrt(square(sigma2) + square(s_plane) * square(sigma1));
    real rho = sigma2 / sigma_tot;
    
    real sum = 0;
    
    for (i in 1 : N) {
      real y_TF = y_min + (i - 1) * h;
      
      real mu = y_TF - s_plane * (y_TF - c) / s;
      real alpha1 = (c1_plane - mu) / sigma_tot;
      real alpha2 = (c2_plane - mu) / sigma_tot;
      real beta = (haty_max - y_TF) / sigma2;
      
      real term = binormal_cdf((-alpha1, beta) | -rho)
                  - binormal_cdf((-alpha2, beta) | -rho);
      
      real w = (i == 1 || i == N) ? 1.0 : 2.0;
      sum += w * term;
    }
    
    return (h / 2.0) * sum / (y_max - y_min);
  }
  
  real integrate_binormal_strip_gl16(
         real y_min,
         real y_max,
         real haty_max,
         real s,
         real c,
         real s_plane,
         real c1_plane,
         real c2_plane,
         real sigma1,
         real sigma2,
         vector x,
         vector w
       ) {
    // 16-point Gauss-Legendre nodes/weights on [-1, 1]
    
    real L = y_max - y_min;
    real mid = 0.5 * (y_max + y_min);
    real half = 0.5 * L;
    
    // precompute constants
    real sigma_tot = sqrt(square(sigma2) + square(s_plane) * square(sigma1));
    real inv_sigma_tot = inv(sigma_tot);
    real inv_sigma2 = inv(sigma2);
    real rho = sigma2 * inv_sigma_tot;
    
    // mu(y) = y - s_plane*(y-c)/s = (1 - s_plane/s)*y + (s_plane*c/s)
    real a = 1.0 - s_plane / s;
    real b = (s_plane * c) / s;
    
    // alpha2 = alpha1 + d_alpha
    real d_alpha = (c2_plane - c1_plane) * inv_sigma_tot;
    
    real acc = 0;
    for (k in 1 : 16) {
      real y_TF = mid + half * x[k];
      
      real mu = a * y_TF + b;
      real alpha1 = (c1_plane - mu) * inv_sigma_tot;
      real beta = (haty_max - y_TF) * inv_sigma2;
      
      real term = binormal_cdf((-alpha1, beta) | -rho)
                  - binormal_cdf((-(alpha1 + d_alpha), beta) | -rho);
      
      acc += w[k] * term;
    }
    
    // integral ≈ half * acc; divide by (y_max - y_min) for Uniform averaging -> (half/L)*acc = 0.5*acc
    return 0.5 * acc;
  }
  
  // Fast path assumes z1 != 0 and z2 != 0 so we can avoid special-case branching.
  // It also assumes denom = sqrt(1 - rho^2) is provided.
  real binormal_cdf_diff_z1_fast(real z1a, real z1b, real z2, real rho,
                                 real denom) {
    // delta simplifies when z1 and z2 are nonzero: delta = (z1*z2 < 0)
    int delta_a = (z1a * z2 < 0);
    int delta_b = (z1b * z2 < 0);
    
    // Owen's T pieces
    real t1 = owens_t(z1a, (z2 / z1a - rho) / denom)
              - owens_t(z1b, (z2 / z1b - rho) / denom);
    
    real t2 = owens_t(z2, (z1a / z2 - rho) / denom)
              - owens_t(z2, (z1b / z2 - rho) / denom);
    
    // Note Phi(z2) cancels in the difference, so don't compute it.
    return 0.5 * (Phi_approx(z1a) - Phi_approx(z1b) - (delta_a - delta_b))
           - t1 - t2;
  }
  
  real integrate_binormal_strip_gl_fast(
         real y_min,
         real y_max,
         real haty_max,
         real s,
         real c,
         real s_plane,
         real c1_plane,
         real c2_plane,
         real sigma1,
         real sigma2,
         vector x,
         vector w
       ) {
    real L = y_max - y_min;
    real mid = 0.5 * (y_max + y_min);
    real half = 0.5 * L;
    
    real sp_sig1 = s_plane * sigma1;
    real sigma_tot = sqrt(square(sigma2) + square(sp_sig1));
    real inv_sigma_tot = 1.0 / sigma_tot;
    real inv_sigma2 = 1.0 / sigma2;
    
    real rho_bvn = -(sigma2 * inv_sigma_tot);
    real denom = abs(sp_sig1) * inv_sigma_tot;
    
    real a_mu = 1.0 - s_plane / s;
    real b_mu = (s_plane * c) / s;
    
    real mu_mid = a_mu * mid + b_mu;
    real mu_slope = a_mu * half;
    
    real alpha_base = (c1_plane - mu_mid) * inv_sigma_tot;
    real alpha_slope = (-mu_slope) * inv_sigma_tot;
    
    real beta_base = (haty_max - mid) * inv_sigma2;
    real beta_slope = (-half) * inv_sigma2;
    
    real d_alpha = (c2_plane - c1_plane) * inv_sigma_tot;
    
    int n_pts = rows(x); // <-- replaces hardcoded 16
    real acc = 0;
    for (k in 1 : n_pts) {
      real xk = x[k];
      real alpha1 = alpha_base + alpha_slope * xk;
      real beta = beta_base + beta_slope * xk;
      real z1a = -alpha1;
      real z1b = z1a - d_alpha;
      acc += w[k] * binormal_cdf_diff_z1_fast(z1a, z1b, beta, rho_bvn, denom);
    }
    
    return 0.5 * acc;
  }
  
  real integrate_binormal_strip_simpson_fast(
         real y_min,
         real y_max,
         real haty_max,
         real s,
         real c,
         real s_plane,
         real c1_plane,
         real c2_plane,
         real sigma1,
         real sigma2,
         int n_intervals // must be even for Simpson's rule
         ) {
    // Step size
    real h = (y_max - y_min) / n_intervals;
    
    // Precompute constants once
    real sp_sig1 = s_plane * sigma1;
    real sigma_tot = sqrt(square(sigma2) + square(sp_sig1));
    real inv_sigma_tot = 1.0 / sigma_tot;
    real inv_sigma2 = 1.0 / sigma2;
    
    real rho_bvn = -(sigma2 * inv_sigma_tot);
    real denom = abs(sp_sig1) * inv_sigma_tot;
    
    // mu(y) coefficients: mu(y) = a_mu * y + b_mu
    real a_mu = 1.0 - s_plane / s;
    real b_mu = (s_plane * c) / s;
    
    // constant strip width in alpha-space
    real d_alpha = (c2_plane - c1_plane) * inv_sigma_tot;
    
    real acc = 0;
    
    // First point (i=0, weight=1)
    {
      real y = y_min;
      real mu_y = a_mu * y + b_mu;
      real alpha1 = (c1_plane - mu_y) * inv_sigma_tot;
      real beta = (haty_max - y) * inv_sigma2;
      real z1a = -alpha1;
      real z1b = z1a - d_alpha;
      
      acc += binormal_cdf_diff_z1_fast(z1a, z1b, beta, rho_bvn, denom);
    }
    
    // Interior points: odd indices get weight 4, even indices get weight 2
    for (i in 1 : (n_intervals - 1)) {
      real y = y_min + i * h;
      real mu_y = a_mu * y + b_mu;
      real alpha1 = (c1_plane - mu_y) * inv_sigma_tot;
      real beta = (haty_max - y) * inv_sigma2;
      real z1a = -alpha1;
      real z1b = z1a - d_alpha;
      
      real f_val = binormal_cdf_diff_z1_fast(z1a, z1b, beta, rho_bvn, denom);
      
      // Weight is 4 for odd i, 2 for even i
      // This can be written as: 2 + 2*(i%2) or using bit operations
      real weight = 2.0 + 2.0 * (i % 2);
      acc += weight * f_val;
    }
    
    // Last point (i=n_intervals, weight=1)
    {
      real y = y_max;
      real mu_y = a_mu * y + b_mu;
      real alpha1 = (c1_plane - mu_y) * inv_sigma_tot;
      real beta = (haty_max - y) * inv_sigma2;
      real z1a = -alpha1;
      real z1b = z1a - d_alpha;
      
      acc += binormal_cdf_diff_z1_fast(z1a, z1b, beta, rho_bvn, denom);
    }
    
    // Simpson's rule: integral ≈ (h/3) * acc
    // Divide by L to get average: (h/3) * acc / L = acc / (3 * n_intervals)
    return acc / (3.0 * n_intervals);
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
  vector[N_total] sigma_x_std_sq = square(sigma_x_std);
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
  
  int N_gl = 64; // number of GL points for strip integral
  //   import numpy as np
  
  // nodes, weights = np.polynomial.legendre.leggauss(64)
  
  // lines_x = ",\n                         ".join(f"{v:.16e}" for v in nodes)
  // lines_w = ",\n                         ".join(f"{v:.16e}" for v in weights)
  
  // print(f"array[64] real x_ = {{{lines_x}}};")
  // print(f"array[64] real w_ = {{{lines_w}}};")
  vector[N_gl] gl_x;
  vector[N_gl] gl_w;
  if (N_gl == 16) {
    array[16] real x_ = {-0.9894009349916499, -0.9445750230732326,
                         -0.8656312023878317, -0.7554044083550030,
                         -0.6178762444026437, -0.4580167776572274,
                         -0.2816035507792589, -0.09501250983763744,
                         0.09501250983763744, 0.2816035507792589,
                         0.4580167776572274, 0.6178762444026437,
                         0.7554044083550030, 0.8656312023878317,
                         0.9445750230732326, 0.9894009349916499};
    array[16] real w_ = {0.027152459411754095, 0.06225352393864789,
                         0.09515851168249278, 0.12462897125553387,
                         0.14959598881657673, 0.16915651939500254,
                         0.18260341504492359, 0.18945061045506850,
                         0.18945061045506850, 0.18260341504492359,
                         0.16915651939500254, 0.14959598881657673,
                         0.12462897125553387, 0.09515851168249278,
                         0.06225352393864789, 0.027152459411754095};
    gl_x = to_vector(x_);
    gl_w = to_vector(w_);
  } else if (N_gl == 64) {
    array[64] real x_ = {-9.9930504173577217e-01, -9.9634011677195522e-01,
                         -9.9101337147674429e-01, -9.8333625388462598e-01,
                         -9.7332682778991098e-01, -9.6100879965205377e-01,
                         -9.4641137485840277e-01, -9.2956917213193957e-01,
                         -9.1052213707850282e-01, -8.8931544599511414e-01,
                         -8.6599939815409277e-01, -8.4062929625258032e-01,
                         -8.1326531512279754e-01, -7.8397235894334139e-01,
                         -7.5281990726053194e-01, -7.1988185017161088e-01,
                         -6.8523631305423327e-01, -6.4896547125465731e-01,
                         -6.1115535517239328e-01, -5.7189564620263400e-01,
                         -5.3127946401989457e-01, -4.8940314570705296e-01,
                         -4.4636601725346409e-01, -4.0227015796399163e-01,
                         -3.5722015833766813e-01, -3.1132287199021097e-01,
                         -2.6468716220876742e-01, -2.1742364374000708e-01,
                         -1.6964442042399283e-01, -1.2146281929612056e-01,
                         -7.2993121787799042e-02, -2.4350292663424433e-02,
                         2.4350292663424433e-02, 7.2993121787799042e-02,
                         1.2146281929612056e-01, 1.6964442042399283e-01,
                         2.1742364374000708e-01, 2.6468716220876742e-01,
                         3.1132287199021097e-01, 3.5722015833766813e-01,
                         4.0227015796399163e-01, 4.4636601725346409e-01,
                         4.8940314570705296e-01, 5.3127946401989457e-01,
                         5.7189564620263400e-01, 6.1115535517239328e-01,
                         6.4896547125465731e-01, 6.8523631305423327e-01,
                         7.1988185017161088e-01, 7.5281990726053194e-01,
                         7.8397235894334139e-01, 8.1326531512279754e-01,
                         8.4062929625258032e-01, 8.6599939815409277e-01,
                         8.8931544599511414e-01, 9.1052213707850282e-01,
                         9.2956917213193957e-01, 9.4641137485840277e-01,
                         9.6100879965205377e-01, 9.7332682778991098e-01,
                         9.8333625388462598e-01, 9.9101337147674429e-01,
                         9.9634011677195522e-01, 9.9930504173577217e-01};
    array[64] real w_ = {1.7832807216943839e-03, 4.1470332605647742e-03,
                         6.5044579689795528e-03, 8.8467598263643459e-03,
                         1.1168139460131439e-02, 1.3463047896718672e-02,
                         1.5726030476025114e-02, 1.7951715775697274e-02,
                         2.0134823153529990e-02, 2.2270173808383049e-02,
                         2.4352702568711051e-02, 2.6377469715054586e-02,
                         2.8339672614259511e-02, 3.0234657072402474e-02,
                         3.2057928354851342e-02, 3.3805161837141356e-02,
                         3.5472213256882254e-02, 3.7055128540240040e-02,
                         3.8550153178615480e-02, 3.9953741132720280e-02,
                         4.1262563242623354e-02, 4.2473515123653542e-02,
                         4.3583724529323332e-02, 4.4590558163756462e-02,
                         4.5491627927418031e-02, 4.6284796581314271e-02,
                         4.6968182816209882e-02, 4.7540165714830163e-02,
                         4.7999388596458199e-02, 4.8344762234802822e-02,
                         4.8575467441503317e-02, 4.8690957009139578e-02,
                         4.8690957009139578e-02, 4.8575467441503317e-02,
                         4.8344762234802822e-02, 4.7999388596458199e-02,
                         4.7540165714830163e-02, 4.6968182816209882e-02,
                         4.6284796581314271e-02, 4.5491627927418031e-02,
                         4.4590558163756462e-02, 4.3583724529323332e-02,
                         4.2473515123653542e-02, 4.1262563242623354e-02,
                         3.9953741132720280e-02, 3.8550153178615480e-02,
                         3.7055128540240040e-02, 3.5472213256882254e-02,
                         3.3805161837141356e-02, 3.2057928354851342e-02,
                         3.0234657072402474e-02, 2.8339672614259511e-02,
                         2.6377469715054586e-02, 2.4352702568711051e-02,
                         2.2270173808383049e-02, 2.0134823153529990e-02,
                         1.7951715775697274e-02, 1.5726030476025114e-02,
                         1.3463047896718672e-02, 1.1168139460131439e-02,
                         8.8467598263643459e-03, 6.5044579689795528e-03,
                         4.1470332605647742e-03, 1.7832807216943839e-03};
    gl_x = to_vector(x_);
    gl_w = to_vector(w_);
  }
}
parameters {
  // Common slope across all redshift bins
  real<lower=-14 * sd_x, upper=-2 * sd_x> slope_std;
  
  // Intercept for each redshift bin
  
  vector<lower=-24 + slope_std * mean_x / sd_x,
         upper=-14 + slope_std * mean_x / sd_x>[N_bins] intercept_std;
  
  // Intrinsic scatter in x-direction (absolute magnitude)
  real<lower=0, upper=1> sigma_int_x; // in x-units
  real<lower=0, upper=1> sigma_int_y; // in y-units
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
  // likelihood given flat prior in y_TF
  vector[N_total] yfromxstd = intercept_std[bin_idx] + slope_std * x_std;
  vector[N_total] sigmasq1_std = square(sigma_int_x_std) + sigma_x_std_sq;
  vector[N_total] sigmasq2 = square(sigma_int_y) + sigma_y_sq;
  // vector[N_total] sigmasq_tot = square(slope_std) * sigmasq1_std + sigmasq2;
  vector[N_total] sigmasq_tot = square(slope_std)
                                * (square(sigma_int_x_std) + sigma_x_std_sq)
                                + (square(sigma_int_y) + sigma_y_sq);
  
  //  term that applies to all cases
  y ~ normal(yfromxstd, sqrt(sigmasq_tot));
  target += log(abs(slope_std)) * N_total;
  
  // if there is a non-zero range of y values allowed by the TFR limits, then we need to apply the selection function
  if (y_TF_limits != 0) {
    vector[N_total] mu_star = (yfromxstd .* sigmasq2
                               + y * square(slope_std) .* sigmasq1_std)
                              ./ sigmasq_tot;
    
    vector[N_total] sqrt_sigmasq_star = abs(slope_std)
                                        * sqrt(
                                               (sigmasq1_std .* sigmasq2)
                                               ./ sigmasq_tot);
    
    // containers used for multiple purposes
    vector[N_total] term_lb;
    vector[N_total] term_ub;
    // // Term for the TFR limits
    for (n in 1 : N_total) {
      // log(Phi_approx) lacks precision for this step
      term_lb[n] = normal_lcdf(y_min | mu_star[n], sqrt_sigmasq_star[n]);
      term_ub[n] = normal_lcdf(y_max | mu_star[n], sqrt_sigmasq_star[n]);
    }
    
    target += log_diff_exp(term_ub, term_lb); // done with this use of term_lb/ub
    
    // Term for the selection function
    if (y_selection != 0 && plane_cut == 0) {
      // vector[N_total] sigma2 = sqrt(sigmasq2);
      vector[N_total] sigma2 = sqrt(square(sigma_int_y) + sigma_y_sq);
      
      term_lb = (haty_max - y_min) / sigma2;
      term_ub = (haty_max - y_max) / sigma2;
      
      vector[N_total] logsigma2 = 0.5 * log(square(sigma_int_y) + sigma_y_sq);
      
      // standard‑normal arguments for the lower‑ and upper‑bound CDFs
      vector[3] lse_terms;
      for (n in 1 : N_total) {
        lse_terms[1] = log_lb + std_normal_lcdf(term_lb[n]);
        lse_terms[2] = logsigma2[n] + std_normal_lpdf(term_lb[n]);
        lse_terms[3] = log_minus_ub + std_normal_lcdf(term_ub[n]);
        term_lb[n] = log_sum_exp(lse_terms);
        term_ub[n] = logsigma2[n] + std_normal_lpdf(term_ub[n]);
      }
      
      target += -log_diff_exp(term_lb, term_ub);
    } else if (y_selection != 0 && plane_cut == 1) {
      for (n in 1 : N_total) {
        target += -log(
                       integrate_binormal_strip_gl_fast(y_min, y_max,
                         haty_max, slope_std, intercept_std[bin_idx],
                         slope_plane_std, intercept_plane_std,
                         intercept_plane2_std, sqrt(sigmasq1_std[n]),
                         sqrt(sigmasq2[n]), gl_x, gl_w));
      }
    }
  }
  
  // Priors
  sigma_int_x ~ cauchy(0, 0.5);
  sigma_int_y ~ cauchy(0, 0.5);
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
  // real sigma_int_x = sigma_int_x_std * sd_x;
}
