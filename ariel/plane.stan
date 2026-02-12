// Tully-Fisher Relation (TFR) model with multiple redshift bins
// Implements selection domain: upper limit in \hat{y} AND a fixed half-plane cut
//
// Selection region for each galaxy i (in measured space):
//   S_i :  (i)  \hat y_i <= haty_max
//         (ii)  \hat y_i - a_hp * \hat x_i - b_hp >= 0   <=>  \hat y_i >= a_hp * \hat x_i + b_hp
//
// This code follows the pattern of your base model and applies a selection correction
// under the same "flat prior in latent y between [y_lb, y_ub]" setup used in your code.
//
// NOTE: This implementation treats the half-plane threshold as a lower limit in \hat y
// conditional on the measured x_i appearing in the dataset, i.e. y_hp_i = a_hp * x[i] + b_hp.

functions {
  // h(t) = \int Phi(t) dt = t*Phi(t) + phi(t), always positive
  // We compute log(h(t)) stably using log-sum-exp / log-diff-exp.
  real log_h(real t) {
    real log_phi = std_normal_lpdf(t);   // log phi(t)
    real log_Phi = std_normal_lcdf(t);   // log Phi(t)

    if (t >= 0) {
      // h(t) = phi(t) + t*Phi(t)
      return log_sum_exp(log_phi, log(t) + log_Phi);
    } else {
      // h(t) = phi(t) - (-t)*Phi(t), and for t<0 we have phi(t) > (-t)*Phi(t)
      return log_diff_exp(log_phi, log(-t) + log_Phi);
    }
  }

  // Computes log I(c) where
  //   I(c) = \int_{y_lb}^{y_ub} Phi( (c - y) / sigma ) dy
  // Using the identity:
  //   I(c) = sigma * ( h( (c-y_lb)/sigma ) - h( (c-y_ub)/sigma ) )
  real log_I_Phi_uniform(real c, real y_lb, real y_ub, real sigma) {
    real t_lb = (c - y_lb) / sigma;
    real t_ub = (c - y_ub) / sigma;
    return log(sigma) + log_diff_exp(log_h(t_lb), log_h(t_ub));
  }

  // Computes log of:
  //   \int_{y_lb}^{y_ub} [ Phi((c_hi - y)/sigma) - Phi((c_lo - y)/sigma) ] dy
  // = I(c_hi) - I(c_lo)
  real log_sel_interval_uniform_y(real c_lo, real c_hi,
                                  real y_lb, real y_ub,
                                  real sigma) {
    if (c_hi <= c_lo) return negative_infinity();
    return log_diff_exp(
      log_I_Phi_uniform(c_hi, y_lb, y_ub, sigma),
      log_I_Phi_uniform(c_lo, y_lb, y_ub, sigma)
    );
  }
}

data {
  // Number of redshift bins
  int<lower=1> N_bins; // for the moment N_bins = 1

  // Total number of galaxies across all bins
  int<lower=0> N_total;

  // Measured data (these are \hat x and \hat y)
  vector[N_total] x;
  vector<lower=0>[N_total] sigma_x;

  vector[N_total] y;
  vector<lower=0>[N_total] sigma_y;

  // Selection function parameters
  real haty_max;

  // Fixed half-plane cut parameters:  \hat y >= a_hp * \hat x + b_hp
  real a_hp;
  real b_hp;

  // If you later enable multiple bins, pass an array:
  // array[N_total] int<lower=1, upper=N_bins> bin_idx;
}

transformed data {
  // Standardizing predictor variable
  real mean_x = mean(x);
  real sd_x   = sd(x);
  real sd_y   = sd(y);

  vector[N_total] x_std       = (x - mean_x) / sd_x;
  vector[N_total] sigma_x_std = sigma_x / sd_x;

  // Dataset support for the "flat prior in latent y" piece
  // (keep your constraints that force y_lb < haty_max < y_ub)
  real<upper=haty_max> y_lb = -23;
  real<lower=haty_max> y_ub = -15;

  vector[N_total] sigma_x_std_sq = square(sigma_x_std);
  vector[N_total] sigma_y_sq     = square(sigma_y);

  int bin_idx = 1;

  // run configuration parameters
  int y_TF_limits = 1;
  int y_selection = 1;
  int fit_sigmas  = 1;
}

parameters {
  // Common slope across all redshift bins (in y-units per x_std unit)
  real<lower=-12 * sd_x, upper=-4 * sd_x> slope_std;

  // Intercept for each redshift bin (in y-units)
  vector<upper=0>[N_bins] intercept_std;

  // Intrinsic scatters
  real<lower=0> sigma_int_x;   // in x-units
  real<lower=0> sigma_int_y;   // in y-units
}

transformed parameters {
  real sigma_int_x_std;
  if (fit_sigmas == 0) {
    sigma_int_x_std = sigma_int_y / sd_x;
  } else {
    sigma_int_x_std = sigma_int_x / sd_x;
  }
}

model {
  // Base likelihood
  vector[N_total] yfromxstd      = intercept_std[bin_idx] + slope_std * x_std;
  vector[N_total] sigmasq1_std   = square(sigma_int_x_std) + sigma_x_std_sq;
  vector[N_total] sigmasq2       = square(sigma_int_y) + sigma_y_sq;
  vector[N_total] sigmasq_tot    = square(slope_std) .* sigmasq1_std + sigmasq2;

  y ~ normal(yfromxstd, sqrt(sigmasq_tot));
  target += log(abs(slope_std)) * N_total;

  // Term for finite support in latent y (your existing "TFR limits" block)
  if (y_TF_limits != 0) {
    vector[N_total] mu_star =
      (yfromxstd .* sigmasq2 + y .* square(slope_std) .* sigmasq1_std) ./ sigmasq_tot;

    vector[N_total] sqrt_sigmasq_star =
      abs(slope_std) * sqrt( (sigmasq1_std .* sigmasq2) ./ sigmasq_tot );

    vector[N_total] term_lb;
    vector[N_total] term_ub;

    for (n in 1:N_total) {
      term_lb[n] = normal_lcdf(y_lb | mu_star[n], sqrt_sigmasq_star[n]);
      term_ub[n] = normal_lcdf(y_ub | mu_star[n], sqrt_sigmasq_star[n]);
    }
    target += log_diff_exp(term_ub, term_lb);

    // Selection correction: \hat y <= haty_max AND \hat y >= a_hp*\hat x + b_hp
    if (y_selection != 0) {
      vector[N_total] sigma2 = sqrt(sigmasq2); // = sqrt(sigma_int_y^2 + sigma_y^2)

      for (n in 1:N_total) {
        real y_hp = a_hp * x[n] + b_hp;  // half-plane lower threshold in \hat y
        real log_sel_int =
          log_sel_interval_uniform_y(y_hp, haty_max, y_lb, y_ub, sigma2[n]);

        // divide by selection probability (up to the constant 1/(y_ub-y_lb), which cancels)
        target += -log_sel_int;
      }
    }
  }

  // Priors
  sigma_int_x ~ cauchy(0, 0.03 * 10);
  sigma_int_y ~ cauchy(0, 0.03 * 10);
}

generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
}