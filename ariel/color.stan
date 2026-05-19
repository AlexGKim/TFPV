// color.stan — Color-Correction TFR Model (Appendix C)
//
// Extends tophat.stan by adding z-band data (hat_z) and four parameters:
//   gamma    — luminosity-color slope at fixed velocity
//   delta_c  — population color-velocity slope (named delta_c to avoid clash with any
//               Stan built-in; delta is not reserved but delta_c is clearer)
//   mu_c     — conditional mean color at x = x_bar
//   tau_c    — intrinsic color scatter at fixed velocity
//
// Key structural change: the analytical 2D marginal likelihood of tophat.stan
// is replaced by a 1D numerical integral of the trivariate N_3 density over y_TF,
// using the same sinh-reparameterized Gauss-Legendre scheme as the selection function.
// Selection probability reuses integrate_binormal_strip_sinh2_gl with sigma2 -> sqrt(A11).
//
// References: paper/main.tex lines 956-1269 (Appendix C).
// Differences from tophat.stan are marked // [COLOR].

functions {
  real binormal_cdf(real z1, real z2, real rho) {
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
  // Trapezoidal integration is more efficient than Simpson's rule or Gauss-Legendre quadrature
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
    sum = binormal_cdf(-alpha_n | beta_n, -rho);

    // Evaluate integrand at interior points
    for (n in 1 : (N - 1)) {
      real y_TF = y_min + n * h;
      alpha_n = (c_plane - (y_TF - s_plane * (y_TF - c) / s)) / sigma_tot;
      beta_n = (haty_max - y_TF) / sigma2;
      sum += 2.0 * binormal_cdf(-alpha_n | beta_n, -rho);
    }

    // Evaluate integrand at right endpoint
    alpha_n = (c_plane - (y_max - s_plane * (y_max - c) / s)) / sigma_tot;
    beta_n = (haty_max - y_max) / sigma2;
    sum += binormal_cdf(-alpha_n | beta_n, -rho);

    // Trapezoidal rule with top-hat normalization
    return (h / 2.0) * sum / (y_max - y_min);
  }

  // skips one Phi that cancels out in the difference, so more accurate for large arguments
  real binormal_strip_cdf(real z1, real z2, real rho) {
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
    return 0.5 * (Phi(z1) - delta) - term1 - term2;
  }

  // skips one Phi that cancels out in the difference, so more accurate for large arguments
  real integrand(real z1, real z2, real z3, real rho) {
    real denom = sqrt((1 + rho) * (1 - rho));
    real term1 = z1 == 0 ? (z2 > 0 ? 0.25 : -0.25)
                 : owens_t(z1, (z2 / z1 - rho) / denom);
    real term2 = z2 == 0 ? (z1 > 0 ? 0.25 : -0.25)
                 : owens_t(z2, (z1 / z2 - rho) / denom);
    real z1z2 = z1 * z2;
    real delta = z1z2 < 0 || (z1z2 == 0 && (z1 + z2) < 0);
    return 0.5 * (Phi(z1) - delta) - term1 - term2;
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
    {
      int i = 1;
      real y_TF = y_min + (i - 1) * h;

      real mu = y_TF - s_plane * (y_TF - c) / s;
      real alpha1 = (c1_plane - mu) / sigma_tot;
      real alpha2 = (c2_plane - mu) / sigma_tot;
      real beta = (haty_max - y_TF) / sigma2;

      real term = binormal_strip_cdf(-alpha1 | beta, -rho)
                  - binormal_strip_cdf(-alpha2 | beta, -rho);

      sum += term;
    }
    for (i in 2 : N - 1) {
      real y_TF = y_min + (i - 1) * h;

      real mu = y_TF - s_plane * (y_TF - c) / s;
      real alpha1 = (c1_plane - mu) / sigma_tot;
      real alpha2 = (c2_plane - mu) / sigma_tot;
      real beta = (haty_max - y_TF) / sigma2;

      real term = binormal_strip_cdf(-alpha1 | beta, -rho)
                  - binormal_strip_cdf(-alpha2 | beta, -rho);

      sum += 2 * term;
    }
    {
      int i = N;
      real y_TF = y_min + (i - 1) * h;

      real mu = y_TF - s_plane * (y_TF - c) / s;
      real alpha1 = (c1_plane - mu) / sigma_tot;
      real alpha2 = (c2_plane - mu) / sigma_tot;
      real beta = (haty_max - y_TF) / sigma2;

      real term = binormal_strip_cdf(-alpha1 | beta, -rho)
                  - binormal_strip_cdf(-alpha2 | beta, -rho);

      sum += term;
    }
    return (h / 2.0) * sum / (y_max - y_min);
  }

  // Vectorized over N:
  // returns, elementwise in i,
  //   binormal_strip_cdf((z1a[i], z2[i]) | rho) - binormal_strip_cdf((z1b[i], z2[i]) | rho)
  //
  // Key speedup: owens_t is called on vectors of length N (4 calls total),
  // instead of scalar owens_t inside an i-loop.
  vector binormal_strip_cdf_diff_same_z2_vec(
           vector z1a,
           vector z1b,
           vector z2,
           real rho
         ) {
    int N = rows(z2);
    real denom = sqrt((1 + rho) * (1 - rho));

    vector[N] a1; // (z2/z1a - rho)/denom
    vector[N] a2; // (z1a/z2 - rho)/denom
    vector[N] a3; // (z2/z1b - rho)/denom
    vector[N] a4; // (z1b/z2 - rho)/denom

    // delta per your scalar definition, stored as 0/1 real
    vector[N] delta_a;
    vector[N] delta_b;

    // flags for the exact scalar corner cases
    array[N] int z1a0;
    array[N] int z1b0;
    array[N] int z20;
    array[N] int both0_a;
    array[N] int both0_b;

    // build "a" safely (no division by 0), and compute delta exactly as in scalar code
    for (i in 1 : N) {
      real z1az2 = z1a[i] * z2[i];
      real z1bz2 = z1b[i] * z2[i];

      z1a0[i] = (z1a[i] == 0);
      z1b0[i] = (z1b[i] == 0);
      z20[i] = (z2[i] == 0);

      both0_a[i] = (z1a0[i] == 1 && z20[i] == 1);
      both0_b[i] = (z1b0[i] == 1 && z20[i] == 1);

      delta_a[i] = (z1az2 < 0) || ((z1az2 == 0) && ((z1a[i] + z2[i]) < 0));
      delta_b[i] = (z1bz2 < 0) || ((z1bz2 == 0) && ((z1b[i] + z2[i]) < 0));

      // only form ratios when safe; dummy values otherwise (will be overridden)
      a1[i] = (z1a0[i] == 1) ? 0 : ((z2[i] / z1a[i] - rho) / denom);
      a3[i] = (z1b0[i] == 1) ? 0 : ((z2[i] / z1b[i] - rho) / denom);

      a2[i] = (z20[i] == 1) ? 0 : ((z1a[i] / z2[i] - rho) / denom);
      a4[i] = (z20[i] == 1) ? 0 : ((z1b[i] / z2[i] - rho) / denom);
    }

    // 4 vectorized owens_t calls (length N each)
    vector[N] t1a = owens_t(z1a, a1);
    vector[N] t2a = owens_t(z2, a2);
    vector[N] t1b = owens_t(z1b, a3);
    vector[N] t2b = owens_t(z2, a4);

    // start with the Owen's-T values, then override the exact scalar special-cases
    vector[N] term1a = t1a;
    vector[N] term2a = t2a;
    vector[N] term1b = t1b;
    vector[N] term2b = t2b;

    for (i in 1 : N) {
      if (z1a0[i] == 1)
        term1a[i] = (z2[i] > 0 ? 0.25 : -0.25);
      if (z1b0[i] == 1)
        term1b[i] = (z2[i] > 0 ? 0.25 : -0.25);

      if (z20[i] == 1)
        term2a[i] = (z1a[i] > 0 ? 0.25 : -0.25);
      if (z20[i] == 1)
        term2b[i] = (z1b[i] > 0 ? 0.25 : -0.25);
    }

    vector[N] Fa = 0.5 * (Phi(z1a) - delta_a) - term1a - term2a;
    vector[N] Fb = 0.5 * (Phi(z1b) - delta_b) - term1b - term2b;

    // exact (0,0) override
    {
      real c00 = 0.25 + asin(rho) / (2 * pi());
      for (i in 1 : N) {
        if (both0_a[i] == 1)
          Fa[i] = c00;
        if (both0_b[i] == 1)
          Fb[i] = c00;
      }
    }

    return Fa - Fb;
  }

  // Vectorized over N integration samples (trapezoid rule),
  // matching your integrate_binormal_strip_trapez() but doing the heavy work in vectors.
  real integrate_binormal_strip_trapez_vecN(
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
    if (N < 2)
      reject("integrate_binormal_strip_trapez_vecN: N must be >= 2");

    real h = (y_max - y_min) / (N - 1);

    real sigma_tot = sqrt(square(sigma2) + square(s_plane) * square(sigma1));
    real rho = sigma2 / sigma_tot;

    vector[N] y_TF;
    for (n in 1 : N)
      y_TF[n] = y_min + (n - 1) * h;

    vector[N] mu = y_TF - (s_plane / s) * (y_TF - c);
    vector[N] alpha1 = (c1_plane - mu) / sigma_tot;
    vector[N] alpha2 = (c2_plane - mu) / sigma_tot;
    vector[N] beta = (haty_max - y_TF) / sigma2;

    vector[N] term = binormal_strip_cdf_diff_same_z2_vec(-alpha1, -alpha2,
                       beta, -rho);

    // trapezoid weights
    vector[N] w = rep_vector(2.0, N);
    w[1] = 1.0;
    w[N] = 1.0;

    return (h / 2.0) * dot_product(w, term) / (y_max - y_min);
  }

  // Bracket term:
  //   Phi2(-alpha1, beta; -rho) - Phi2(-alpha2, beta; -rho)
  // for each y_TF in the input vector.
  vector strip_integrand(vector y_TF,
                         real s,
                         real c,
                         real bar_c1,
                         real bar_c2,
                         real yhat_max,
                         real sigma1_i,
                         real sigma2_i,
                         real bar_s) {
    int N = num_elements(y_TF);

    real denom = sqrt(square(sigma2_i) + square(bar_s) * square(sigma1_i));
    real rho = sigma2_i / denom;
    real sqrt1mr2 = sqrt(1.0 - square(rho));

    // y_shift = y_TF - bar_s * (y_TF - c)/s = (1 - bar_s/s)*y_TF + (bar_s*c/s)
    real k = 1.0 - bar_s / s;
    real b = bar_s * c / s;

    vector[N] y_shift = k * y_TF + b;
    vector[N] alpha1 = (bar_c1 - y_shift) / denom;
    vector[N] alpha2 = (bar_c2 - y_shift) / denom;
    vector[N] beta = (yhat_max - y_TF) / sigma2_i;

    vector[N] z1a = -alpha1;
    vector[N] z1b = -alpha2;

    // delta(-alpha1,beta) - delta(-alpha2,beta), vectorized via step()
    // step(x)=1 if x>0 else 0
    vector[N] delta_diff;
    for (n in 1 : N) {
      delta_diff[n] = 0.0;
      if (beta[n] > 0 && alpha1[n] <= 0 && alpha2[n] > 0)
        delta_diff[n] = -1.0;
      else if (beta[n] < 0 && alpha1[n] < 0 && alpha2[n] >= 0)
        delta_diff[n] = 1.0;
    }

    // Owen's-t arguments, vectorized
    vector[N] a_z1a = (beta ./ z1a + rho) / sqrt1mr2;
    vector[N] a_z1b = (beta ./ z1b + rho) / sqrt1mr2;
    vector[N] a_b1 = (z1a ./ beta + rho) / sqrt1mr2;
    vector[N] a_b2 = (z1b ./ beta + rho) / sqrt1mr2;

    // Assemble bracket
    vector[N] out = 0.5 * (Phi_approx(z1a) - Phi_approx(z1b) - delta_diff)
                    - (owens_t(z1a, a_z1a) - owens_t(z1b, a_z1b))
                    - (owens_t(beta, a_b1) - owens_t(beta, a_b2));

    return out;
  }

  // Integrand for BOTH y-cuts:
  // I(y_TF) = [Phi2(-a1,beta_max;-rho)-Phi2(-a2,beta_max;-rho)]
  //         - [Phi2(-a1,beta_min;-rho)-Phi2(-a2,beta_min;-rho)]
  vector strip_integrand_two_ycuts(vector y_TF,
                                   real s,
                                   real c,
                                   real bar_c1,
                                   real bar_c2,
                                   real yhat_min,
                                   real yhat_max,
                                   real sigma1_i,
                                   real sigma2_i,
                                   real bar_s) {
    return strip_integrand(y_TF, s, c, bar_c1, bar_c2, yhat_max, sigma1_i,
                           sigma2_i, bar_s)
           - strip_integrand(y_TF, s, c, bar_c1, bar_c2, yhat_min, sigma1_i,
                             sigma2_i, bar_s);
  }

  real integrate_binormal_strip_sinh_gl(
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
         vector gl_x,
         vector gl_w
       ) {
    int K = size(gl_x);

    if (size(gl_w) != K)
      reject("integrate_binormal_strip_sinh_gl: gl_x and gl_w must have same length");
    if (sigma2 <= 0)
      reject("integrate_binormal_strip_sinh_gl: sigma2 must be > 0");
    if (y_max <= y_min)
      reject("integrate_binormal_strip_sinh_gl: require y_max > y_min");

    real D = sqrt(square(sigma2) + square(s_plane * sigma1));
    real rho = sigma2 / D;

    rho = fmin(1 - 1e-12, fmax(-1 + 1e-12, rho));

    real u_min = asinh((haty_max - y_max) / sigma2);
    real u_max = asinh((haty_max - y_min) / sigma2);

    real mid = 0.5 * (u_min + u_max);
    real half = 0.5 * (u_max - u_min);

    real inv_s = 1.0 / s;

    vector[K] u = mid + half * gl_x;
    vector[K] t = sinh(u);
    vector[K] y_tf = haty_max - sigma2 * t;

    vector[K] diff = strip_integrand(y_tf, s, c, c1_plane, c2_plane,
                                     haty_max, sigma1, sigma2, s_plane);
    real acc = sum(gl_w .* diff .* cosh(u));
    return sigma2 * half * acc;
  }

  // ∫_{y_min}^{y_max} I(y_TF) dy_TF,
  // split into two pieces and use sinh transforms around y_TF = yhat_min and y_TF = yhat_max.
  real integrate_binormal_strip_sinh2_gl(
         real y_min,
         real y_max,
         real yhat_min,
         real yhat_max,
         real s,
         real c,
         real s_plane,
         real c1_plane,
         real c2_plane,
         real sigma1,
         real sigma2,
         vector gl_x,
         vector gl_w
       ) {
    int K = size(gl_x);

    if (size(gl_w) != K)
      reject("integrate_binormal_strip_sinh2_gl: gl_x and gl_w must have same length");
    if (sigma2 <= 0)
      reject("integrate_binormal_strip_sinh2_gl: sigma2 must be > 0");
    if (y_max <= y_min)
      reject("integrate_binormal_strip_sinh2_gl: require y_max > y_min");
    if (yhat_max <= yhat_min)
      reject("integrate_binormal_strip_sinh2_gl: require yhat_max > yhat_min");

    real y_star = fmin(y_max, fmax(y_min, 0.5 * (yhat_min + yhat_max)));

    real I1 = 0.0;
    real I2 = 0.0;

    // ---- Piece 1: y_TF in [y_min, y_star], sinh-transform around yhat_min ----
    if (y_star > y_min) {
      real umin1 = asinh((y_min - yhat_min) / sigma2);
      real umax1 = asinh((y_star - yhat_min) / sigma2);

      real mid1 = 0.5 * (umin1 + umax1);
      real half1 = 0.5 * (umax1 - umin1);

      vector[K] u1 = mid1 + half1 * gl_x;
      vector[K] ytf1 = yhat_min + sigma2 * sinh(u1);

      vector[K] f1 = strip_integrand_two_ycuts(ytf1, s, c, c1_plane,
                       c2_plane, yhat_min, yhat_max, sigma1, sigma2, s_plane);

      I1 = sigma2 * half1 * sum(gl_w .* f1 .* cosh(u1));
    }

    // ---- Piece 2: y_TF in [y_star, y_max], sinh-transform around yhat_max ----
    if (y_max > y_star) {
      real umin2 = asinh((yhat_max - y_max) / sigma2);
      real umax2 = asinh((yhat_max - y_star) / sigma2);

      real mid2 = 0.5 * (umin2 + umax2);
      real half2 = 0.5 * (umax2 - umin2);

      vector[K] u2 = mid2 + half2 * gl_x;
      vector[K] ytf2 = yhat_max - sigma2 * sinh(u2);

      vector[K] f2 = strip_integrand_two_ycuts(ytf2, s, c, c1_plane,
                       c2_plane, yhat_min, yhat_max, sigma1, sigma2, s_plane);

      I2 = sigma2 * half2 * sum(gl_w .* f2 .* cosh(u2));
    }

    return I1 + I2;
  }
}
data {
  int<lower=1> N_bins;
  int<lower=0> N_total;

  vector[N_total] y;
  vector<lower=0>[N_total] sigma_y;
  vector[N_total] x;
  vector<lower=0>[N_total] sigma_x;

  // [COLOR] z-band absolute magnitudes: hat_z = hat_y - hat_color  (Eq. cc:z_def)
  vector[N_total] z;
  vector<lower=0>[N_total] sigma_z;

  real haty_min;
  real haty_max;
  real slope_plane;
  real intercept_plane;
  real intercept_plane2;

  real<upper=haty_max> y_min;
  real<lower=haty_max> y_max;

  // [COLOR] sample-mean observed color = mean(y - z); used as prior mean for mu_c (Eq. C31)
  real c_bar_obs;

  // [KCORR] per-galaxy redshift, used by the latent k-correction term
  vector<lower=0>[N_total] z_obs;
}
transformed data {
  real mean_x = mean(x);
  real sd_x = sd(x);
  real sd_y = sd(y);
  vector[N_total] x_std = (x - mean_x) / sd_x;
  vector[N_total] sigma_x_std = sigma_x / sd_x;

  real log_lb = log(haty_max - y_min);
  real log_minus_ub = log(y_max - haty_max);
  vector[N_total] sigma_x_std_sq = square(sigma_x_std);
  vector[N_total] sigma_y_sq = square(sigma_y);
  // [COLOR]
  vector[N_total] sigma_z_sq = square(sigma_z);

  int bin_idx = 1;

  int y_TF_limits = 1;
  int y_selection = 1;
  int plane_cut = 1;
  int fit_sigmas = 1;

  real slope_plane_std = slope_plane * sd_x;
  real intercept_plane_std = intercept_plane + slope_plane_std * mean_x / sd_x;
  real intercept_plane2_std = intercept_plane2 + slope_plane_std * mean_x / sd_x;

  // [COLOR] x_bar in standardized coordinates is 0 by construction (mean of x_std = 0)
  real x_bar_std = 0.0;

  // [KCORR] mean of log1p(z_obs) — centering removes the alpha-intercept degeneracy
  real mean_log1pz = mean(log1p(z_obs));

  // GL nodes — copied verbatim from tophat.stan
  array[32] real gl_x_arr = {-0.9972638618494815635, -0.9856115115452683354,
                             -0.9647622555875064308, -0.9349060759377396892,
                             -0.8963211557660521240, -0.8493676137325699701,
                             -0.7944837959679424070, -0.7321821187402896804,
                             -0.6630442669302152010, -0.5877157572407623290,
                             -0.5068999089322293900, -0.4213512761306353454,
                             -0.3318686022821276498, -0.2392873622521370745,
                             -0.1444719615827964935, -0.0483076656877383162,
                             0.0483076656877383162, 0.1444719615827964935,
                             0.2392873622521370745, 0.3318686022821276498,
                             0.4213512761306353454, 0.5068999089322293900,
                             0.5877157572407623290, 0.6630442669302152010,
                             0.7321821187402896804, 0.7944837959679424070,
                             0.8493676137325699701, 0.8963211557660521240,
                             0.9349060759377396892, 0.9647622555875064308,
                             0.9856115115452683354, 0.9972638618494815635};
  vector[32] gl_x = to_vector(gl_x_arr);
  array[32] real gl_w_arr = {0.0070186100094700966, 0.0162743947309056706,
                             0.0253920653092620595, 0.0342738629130214331,
                             0.0428358980222266807, 0.0509980592623761762,
                             0.0586840934785355471, 0.0658222227763618468,
                             0.0723457941088485062, 0.0781938957870703065,
                             0.0833119242269467552, 0.0876520930044038111,
                             0.0911738786957638847, 0.0938443990808045656,
                             0.0956387200792748594, 0.0965400885147278006,
                             0.0965400885147278006, 0.0956387200792748594,
                             0.0938443990808045656, 0.0911738786957638847,
                             0.0876520930044038111, 0.0833119242269467552,
                             0.0781938957870703065, 0.0723457941088485062,
                             0.0658222227763618468, 0.0586840934785355471,
                             0.0509980592623761762, 0.0428358980222266807,
                             0.0342738629130214331, 0.0253920653092620595,
                             0.0162743947309056706, 0.0070186100094700966};
  vector[32] gl_w = to_vector(gl_w_arr);

  array[16] real gl_x_arr_16 = {-0.9894009349916499325,
                                -0.9445750230732325761,
                                -0.8656312023341810203,
                                -0.7554044083550030338,
                                -0.6178762444026437484,
                                -0.4580167776572273864,
                                -0.2816035507792589132,
                                -0.0950125098360222962,
                                0.0950125098360222962, 0.2816035507792589132,
                                0.4580167776572273864, 0.6178762444026437484,
                                0.7554044083550030338, 0.8656312023341810203,
                                0.9445750230732325761, 0.9894009349916499325};
  vector[16] gl_x_16 = to_vector(gl_x_arr_16);

  array[16] real gl_w_arr_16 = {0.0271524594117540949, 0.0622535239386478929,
                                0.0951585116824927848, 0.1246289512509462112,
                                0.1495959888165767320, 0.1691565193950025381,
                                0.1826034150449235888, 0.1894506104550684835,
                                0.1894506104550684835, 0.1826034150449235888,
                                0.1691565193950025381, 0.1495959888165767320,
                                0.1246289512509462112, 0.0951585116824927848,
                                0.0622535239386478929, 0.0271524594117540949};
  vector[16] gl_w_16 = to_vector(gl_w_arr_16);

  array[8] real gl_x_arr_8 = {-0.9602898564975362317, -0.7966664774136267396,
                              -0.5255324099163289858, -0.1834346424956498049,
                              0.1834346424956498049, 0.5255324099163289858,
                              0.7966664774136267396, 0.9602898564975362317};
  vector[8] gl_x_8 = to_vector(gl_x_arr_8);

  array[8] real gl_w_arr_8 = {0.1012285362903762591, 0.2223810344533744861,
                              0.3137066458778872873, 0.3626837833783619830,
                              0.3626837833783619830, 0.3137066458778872873,
                              0.2223810344533744861, 0.1012285362903762591};
  vector[8] gl_w_8 = to_vector(gl_w_arr_8);
}
parameters {
  real<lower=-9 * sd_x, upper=-4.0 * sd_x> slope_std;
  vector<lower=-24 + slope_std * mean_x / sd_x,
         upper=-14 + slope_std * mean_x / sd_x>[N_bins] intercept_std;
  real<lower=0, upper=1> sigma_int_x;
  real<lower=0, upper=1> sigma_int_y;
  real log_sigma_int_z;

  // [COLOR] color-correction parameters (Eqs. C29-C32)
  real gamma_tau_c;     // p = γ·τ_c (sampled directly; γ<0, τ_c>0 ⟹ p<0)
  real delta_c;          // population color-velocity slope                   (Eq. C30)
  real mu_c;             // mean color at x = x_bar                          (Eq. C31)
  real<lower=0> tau_c;   // intrinsic color scatter; lower=0 -> half-Cauchy  (Eq. C32)

  // [KCORR] latent k-correction parameter
  // Augments ŷ_obs by ΔM = alpha_kcorr * (log1p(z_obs[n]) - mean_log1pz)
  // (color-independent redshift trend). Untruncated to avoid boundary-prior HMC pathology
  // (see plan greedy-bubbling-pebble.md, Phase A5 decision 2026-05-18).
  real alpha_kcorr;
}
transformed parameters {
  real sigma_int_x_std;
  if (fit_sigmas == 0) {
    sigma_int_x_std = sigma_int_y / sd_x;
  } else {
    sigma_int_x_std = sigma_int_x / sd_x;
  }
  real<lower=0> sigma_int_z = exp(log_sigma_int_z);
  real gamma = gamma_tau_c / tau_c;
}
model {
  // Priors — baseline
  sigma_int_x ~ cauchy(0, 1);
  sigma_int_y ~ cauchy(0, 1);
  log_sigma_int_z ~ normal(-3, 2);

  // [COLOR] Priors for color-correction parameters (Eqs. C29-C32)
  // Reparameterized: sample gamma_tau_c = γ·τ_c directly to break the γ–τ_c banana.
  // Prior: p(γ,τ) dγ dτ → p(p/τ, τ)·|1/τ| dp dτ  (Jacobian = 1/τ_c)
  // γ<0 truncation enforced structurally (gamma_tau_c<0, tau_c>0).
  tau_c   ~ cauchy(0, 0.3);
  target += normal_lpdf(gamma | 0, 1) - log(tau_c);
  delta_c ~ std_normal();
  mu_c    ~ normal(c_bar_obs, 1);

  // [KCORR] Prior for latent k-correction parameter
  // |k-corr error| at z=0.1 is plausibly < 0.5 mag, so |alpha|*z_max ~ 0.5 -> scale 5 is wide.
  alpha_kcorr ~ normal(0, 5);

  // Per-galaxy variances in standardized x-coordinates
  vector[N_total] sigmasq1_std = square(sigma_int_x_std) + sigma_x_std_sq;
  vector[N_total] sigma1_std   = sqrt(sigmasq1_std);

  if (y_TF_limits != 0) {
    // [COLOR] A_i matrix scalar entries that don't depend on per-galaxy noise (Eq. C17)
    // γ²τ² = p², γ(γ-1)τ² = p(p-τ), (γ-1)²τ² = (p-τ)²  where p = gamma_tau_c
    real A11_base = square(gamma_tau_c) + square(sigma_int_y);
    real A12      = gamma_tau_c * (gamma_tau_c - tau_c);
    real A22_base = square(gamma_tau_c - tau_c) + square(sigma_int_z);

    // [COLOR] Closed-form numerator coefficients (paper Eq. eq:cc:linear_mean).
    // μ(y_TF) = a_n + b·y_TF with b = (1/slope_std, 1, 1 − δ_c/slope_std) — galaxy-independent.
    // In standardized x, x_bar_std = 0, so the δ_c·x̄ term in a_n vanishes.
    real inv_slope_std        = inv(slope_std);
    vector[3] b_vec           = [inv_slope_std, 1.0, 1.0 - delta_c * inv_slope_std]';
    real intercept_over_slope = intercept_std[bin_idx] * inv_slope_std;

    for (n in 1 : N_total) {
      // [KCORR] Per-galaxy k-correction mean shift, centered to remove alpha-intercept degeneracy
      real alpha_zn = alpha_kcorr * (log1p(z_obs[n]) - mean_log1pz);

      // [COLOR] Per-galaxy A_i diagonal entries (Eq. C17)
      real A11 = A11_base + sigma_y_sq[n];
      real A22 = A22_base + sigma_z_sq[n];

      // [COLOR] B_i covariance matrix (Eq. C21)
      // Off-diagonal uses σ²_{int,x} (not σ²_1) because ẑ depends on latent x, not x̂.
      real s1sq = sigmasq1_std[n];
      real sigma_intx_sq = square(sigma_int_x_std);
      matrix[3, 3] B;
      B[1, 1] = s1sq;
      B[1, 2] = 0.0;
      B[2, 1] = 0.0;
      B[1, 3] = -delta_c * sigma_intx_sq;
      B[3, 1] = -delta_c * sigma_intx_sq;
      B[2, 2] = A11;
      B[2, 3] = A12;
      B[3, 2] = A12;
      B[3, 3] = A22 + square(delta_c) * sigma_intx_sq;

      matrix[3, 3] L_B = cholesky_decompose(B);

      // [COLOR] Observed data vector for this galaxy (standardized x)
      vector[3] obs = [x_std[n], y[n], z[n]]';

      // [COLOR] Closed-form numerator integral (paper Eq. eq:cc:numerator_closed).
      // a_n = (-intercept/slope, Δ_n, Δ_n - μ_c + δ_c·intercept/slope) in standardized x.
      vector[3] a_vec = [-intercept_over_slope,
                         alpha_zn,
                         alpha_zn - mu_c + delta_c * intercept_over_slope]';
      vector[3] d_vec = obs - a_vec;

      // Triangular solves against L_B (Cholesky of B_n): v = L_B⁻¹·b, w = L_B⁻¹·d
      vector[3] v_vec = mdivide_left_tri_low(L_B, b_vec);
      vector[3] w_vec = mdivide_left_tri_low(L_B, d_vec);

      real xi_n    = dot_self(v_vec);              // ξ_n  = bᵀ B_n⁻¹ b
      real phi_n   = dot_product(v_vec, w_vec);    // φ_n  = bᵀ B_n⁻¹ d_n
      real chi_n   = dot_self(w_vec);              // χ_n  = d_nᵀ B_n⁻¹ d_n
      real mu_star = phi_n / xi_n;                 // μ*_n
      real Q_0     = chi_n - square(phi_n) / xi_n; // Q_{0,n}

      real sqrt_xi  = sqrt(xi_n);
      real u_max    = sqrt_xi * (y_max - mu_star);
      real u_min    = sqrt_xi * (y_min - mu_star);
      real log_dPhi = log_diff_exp(normal_lcdf(u_max | 0, 1),
                                    normal_lcdf(u_min | 0, 1));

      // log L_n = -log(2π) - ½·log|B_n| - ½·log(ξ_n) - ½·Q_{0,n} + log ΔΦ_n,
      // with log|B_n| = 2·∑ log(diag(L_B)).  Top-hat prior 1/(y_max - y_min).
      target += -log(2 * pi())
                - sum(log(diagonal(L_B)))
                - 0.5 * log(xi_n)
                - 0.5 * Q_0
                + log_dPhi
                - log(y_max - y_min);

      // [COLOR] Selection probability: same sinh-GL machinery as tophat.stan,
      // but sigma2 -> sqrt(A11) (Appendix C §C.3; hat_z does not enter selection cuts).
      // [KCORR] The k-correction shifts E[y_obs|y_TF] by alpha_zn, so the effective
      // selection window in y_TF-space is shifted by -alpha_zn.
      if (y_selection != 0 && plane_cut == 1) {
        real sigma_eff = sqrt(A11);  // sqrt(Var(ŷ|y_TF)) — only needed for selection
        target += -log(
          integrate_binormal_strip_sinh2_gl(
            y_min, y_max,
            haty_min - alpha_zn, haty_max - alpha_zn,
            slope_std, intercept_std[bin_idx],
            slope_plane_std,
            intercept_plane_std,
            intercept_plane2_std,
            sigma1_std[n],
            sigma_eff,       // [COLOR] sqrt(A11) replaces sqrt(sigmasq2)
            gl_x_8, gl_w_8
          )
        );
      }
    }
  }
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
}
