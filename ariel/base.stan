// ./base sample num_samples=500 num_chains=4 data file=MOCK_n10000_input.json init=MOCK_n10000_init.json output file=MOCK_n10000_base.csv
// ./base sample num_samples=500 num_chains=4 data file=DESI_input.json init=DESI_init.json output file=DESI_base.csv
// ../cmdstan/bin/stansummary output_base_?.csv -i slope -i intercept.1 -i sigma_int_x -i sigma_int_y
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
  
  // skips one Phi that cancels out in the difference, so more accurate for large arguments
  real binormal_strip_cdf(tuple(real, real) z, real rho) {
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
    return 0.5 * (Phi(z1) - delta) - term1 - term2;
  }
  
  // skips one Phi that cancels out in the difference, so more accurate for large arguments
  real integrand(tuple(real, real, real) z, real rho) {
    real z1 = z.1;
    real z2 = z.2;
    real z3 = z.3;
    // if (z1 == 0 && z2 == 0) {
    //   return 0.25 + asin(rho) / (2 * pi());
    // }
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
      
      real term = binormal_strip_cdf((-alpha1, beta) | -rho)
                  - binormal_strip_cdf((-alpha2, beta) | -rho);
      
      sum += term;
    }
    for (i in 2 : N - 1) {
      real y_TF = y_min + (i - 1) * h;
      
      real mu = y_TF - s_plane * (y_TF - c) / s;
      real alpha1 = (c1_plane - mu) / sigma_tot;
      real alpha2 = (c2_plane - mu) / sigma_tot;
      real beta = (haty_max - y_TF) / sigma2;
      
      real term = binormal_strip_cdf((-alpha1, beta) | -rho)
                  - binormal_strip_cdf((-alpha2, beta) | -rho);
      
      sum += 2 * term;
    }
    {
      int i = N;
      real y_TF = y_min + (i - 1) * h;
      
      real mu = y_TF - s_plane * (y_TF - c) / s;
      real alpha1 = (c1_plane - mu) / sigma_tot;
      real alpha2 = (c2_plane - mu) / sigma_tot;
      real beta = (haty_max - y_TF) / sigma2;
      
      real term = binormal_strip_cdf((-alpha1, beta) | -rho)
                  - binormal_strip_cdf((-alpha2, beta) | -rho);
      
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
    
    // vector[N] y_TF = linspaced_vector(N, y_min, y_max);
    vector[N] y_TF;
    for (n in 1 : N) 
      y_TF[n] = y_min + (n - 1) * h; // h = (y_max - y_min)/(N-1)
    
    vector[N] mu = y_TF - (s_plane / s) * (y_TF - c);
    vector[N] alpha1 = (c1_plane - mu) / sigma_tot;
    vector[N] alpha2 = (c2_plane - mu) / sigma_tot;
    vector[N] beta = (haty_max - y_TF) / sigma2;
    
    // term[i] = binormal_strip_cdf((-alpha1[i], beta[i]) | -rho)
    //         - binormal_strip_cdf((-alpha2[i], beta[i]) | -rho)
    vector[N] term = binormal_strip_cdf_diff_same_z2_vec(-alpha1, -alpha2,
                       beta, -rho);
    
    // trapezoid weights
    vector[N] w = rep_vector(2.0, N);
    w[1] = 1.0;
    w[N] = 1.0;
    
    return (h / 2.0) * dot_product(w, term) / (y_max - y_min);
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
    
    // Basic checks (optional but helpful)
    if (size(gl_w) != K) 
      reject("integrate_binormal_strip_sinh_gl: gl_x and gl_w must have same length");
    if (sigma2 <= 0) 
      reject("integrate_binormal_strip_sinh_gl: sigma2 must be > 0");
    if (y_max <= y_min) 
      reject("integrate_binormal_strip_sinh_gl: require y_max > y_min");
    
    // Precompute constants for this i
    real D = sqrt(square(sigma2) + square(s_plane * sigma1));
    real rho = sigma2 / D;
    
    // Clamp rho away from +/-1 for numerical safety in the CDF implementation
    rho = fmin(1 - 1e-12, fmax(-1 + 1e-12, rho));
    
    // sinh-transform bounds:
    // u = asinh( (haty_max - y_TF)/sigma2 )
    real u_min = asinh((haty_max - y_max) / sigma2);
    real u_max = asinh((haty_max - y_min) / sigma2);
    
    // Map Gauss-Legendre nodes from [-1,1] -> [u_min,u_max]
    real mid = 0.5 * (u_min + u_max);
    real half = 0.5 * (u_max - u_min);
    
    real inv_s = 1.0 / s;
    // real acc = 0;
    
    // for (k in 1 : K) {
    //   real u = mid + half * gl_x[k];
    
    //   // t = beta = (haty_max - y_TF)/sigma2
    //   real t = sinh(u);
    
    //   // Back-transform to y_TF
    //   real y_tf = haty_max - sigma2 * t;
    
    //   // m(y_tf) = y_tf - s_plane * (y_tf - c)/s
    //   real m = y_tf - s_plane * (y_tf - c) * inv_s;
    
    //   // z1 = -alpha_k = (m - c_k)/D
    //   real z1_1 = (m - c1_plane) / D;
    //   real z1_2 = (m - c2_plane) / D;
    
    //   // z2 = beta = t
    //   real z2 = t;
    
    //   // integrand in u-space includes Jacobian sigma2*cosh(u)
    //   real diff = binormal_strip_cdf((z1_1, z2) | -rho)
    //               - binormal_strip_cdf((z1_2, z2) | -rho);
    //   acc += gl_w[k] * diff * cosh(u);
    // }
    
    // for (k in 1 : K) {
    
    vector[K] u = mid + half * gl_x;
    
    // t = beta = (haty_max - y_TF)/sigma2
    vector[K] t = sinh(u);
    
    // Back-transform to y_TF
    vector[K] y_tf = haty_max - sigma2 * t;
    
    // // m(y_tf) = y_tf - s_plane * (y_tf - c)/s
    // vector[K] m = y_tf - s_plane * (y_tf - c) * inv_s;
    
    // // z1 = -alpha_k = (m - c_k)/D
    // vector[K] z1_1 = (m - c1_plane) / D;
    // vector[K] z1_2 = (m - c2_plane) / D;
    
    // // z2 = beta = t
    // vector[K] z2 = t;
    
    // integrand in u-space includes Jacobian sigma2*cosh(u)
    vector[K] diff = strip_integrand(y_tf, s, c, c1_plane, c2_plane,
                                     haty_max, sigma1, sigma2, s_plane);
    // acc += gl_w[k] * diff * cosh(u);
    real acc = sum(gl_w .* diff .* cosh(u));
    // Integral over y_TF:
    // ∫ f(y_TF) dy_TF = sigma2 * ∫ f(haty_max - sigma2*sinh u) cosh(u) du
    return sigma2 * half * acc;
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
    
    // Basic checks (optional but helpful)
    if (size(gl_w) != K) 
      reject("integrate_binormal_strip_sinh_gl: gl_x and gl_w must have same length");
    if (sigma2 <= 0) 
      reject("integrate_binormal_strip_sinh_gl: sigma2 must be > 0");
    if (y_max <= y_min) 
      reject("integrate_binormal_strip_sinh_gl: require y_max > y_min");
    
    // Precompute constants for this i
    real D = sqrt(square(sigma2) + square(s_plane * sigma1));
    real rho = sigma2 / D;
    
    // Clamp rho away from +/-1 for numerical safety in the CDF implementation
    rho = fmin(1 - 1e-12, fmax(-1 + 1e-12, rho));
    
    // sinh-transform bounds:
    // u = asinh( (haty_max - y_TF)/sigma2 )
    real u_min = asinh((haty_max - y_max) / sigma2);
    real u_max = asinh((haty_max - y_min) / sigma2);
    
    // Map Gauss-Legendre nodes from [-1,1] -> [u_min,u_max]
    real mid = 0.5 * (u_min + u_max);
    real half = 0.5 * (u_max - u_min);
    
    real inv_s = 1.0 / s;
    // real acc = 0;
    
    // for (k in 1 : K) {
    //   real u = mid + half * gl_x[k];
    
    //   // t = beta = (haty_max - y_TF)/sigma2
    //   real t = sinh(u);
    
    //   // Back-transform to y_TF
    //   real y_tf = haty_max - sigma2 * t;
    
    //   // m(y_tf) = y_tf - s_plane * (y_tf - c)/s
    //   real m = y_tf - s_plane * (y_tf - c) * inv_s;
    
    //   // z1 = -alpha_k = (m - c_k)/D
    //   real z1_1 = (m - c1_plane) / D;
    //   real z1_2 = (m - c2_plane) / D;
    
    //   // z2 = beta = t
    //   real z2 = t;
    
    //   // integrand in u-space includes Jacobian sigma2*cosh(u)
    //   real diff = binormal_strip_cdf((z1_1, z2) | -rho)
    //               - binormal_strip_cdf((z1_2, z2) | -rho);
    //   acc += gl_w[k] * diff * cosh(u);
    // }
    
    // for (k in 1 : K) {
    
    vector[K] u = mid + half * gl_x;
    
    // t = beta = (haty_max - y_TF)/sigma2
    vector[K] t = sinh(u);
    
    // Back-transform to y_TF
    vector[K] y_tf = haty_max - sigma2 * t;
    
    // // m(y_tf) = y_tf - s_plane * (y_tf - c)/s
    // vector[K] m = y_tf - s_plane * (y_tf - c) * inv_s;
    
    // // z1 = -alpha_k = (m - c_k)/D
    // vector[K] z1_1 = (m - c1_plane) / D;
    // vector[K] z1_2 = (m - c2_plane) / D;
    
    // // z2 = beta = t
    // vector[K] z2 = t;
    
    // integrand in u-space includes Jacobian sigma2*cosh(u)
    vector[K] diff = strip_integrand(y_tf, s, c, c1_plane, c2_plane,
                                     haty_max, sigma1, sigma2, s_plane);
    // acc += gl_w[k] * diff * cosh(u);
    real acc = sum(gl_w .* diff .* cosh(u));
    // Integral over y_TF:
    // ∫ f(y_TF) dy_TF = sigma2 * ∫ f(haty_max - sigma2*sinh u) cosh(u) du
    return sigma2 * half * acc;
  }

  // Bracket term:
  //   Phi2(-alpha1, beta; -rho) - Phi2(-alpha2, beta; -rho)
  // for each y_TF in the input vector.
  vector strip_integrand_2(vector y_TF,
                         real s,
                         real c,
                         real bar_c1,
                         real bar_c2,
                         real yhat_max, real yhat_min,
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
    
    vector[N] z1a = -alpha1;
    vector[N] z1b = -alpha2;
    
    for (yhat in (yhat_max, yhat_min)) {
    vector[N] beta = (yhat - y_TF) / sigma2_i;

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
    
  }

    // return out;
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
        //   target += -log(
        //                  integrate_binormal_strip_trapez(y_min, y_max,
        //                    haty_max, slope_std, intercept_std[bin_idx],
        //                    slope_plane_std, intercept_plane_std,
        //                    intercept_plane2_std, sqrt(sigmasq1_std[1]),
        //                    sqrt(sigmasq2[1]), 32));
        target += log(
                      integrate_binormal_strip_sinh_gl(y_min, y_max,
                        haty_max, slope_std, intercept_std[bin_idx],
                        slope_plane_std, intercept_plane_std,
                        intercept_plane2_std, sqrt(sigmasq1_std[1]),
                        sqrt(sigmasq2[1]), gl_x, gl_w));
      }
      // target += - N_total * log(
      //          integrate_binormal_strip_trapez(y_min, y_max,
      //            haty_max, slope_std, intercept_std[bin_idx],
      //            slope_plane_std, intercept_plane_std,
      //            intercept_plane2_std, sqrt(sigmasq1_std[1]),
      //            sqrt(sigmasq2[1]), 128));
      //                 target += - N_total * log(
      // target += - N_total *log( integrate_binormal_strip_sinh_gl(y_min, y_max,
      //    haty_max, slope_std, intercept_std[bin_idx],
      //    slope_plane_std, intercept_plane_std,
      //    intercept_plane2_std, sqrt(sigmasq1_std[1]),
      //    sqrt(sigmasq2[1]), gl_x, gl_w));
    }
  }
  
  // Priors
  sigma_int_x ~ cauchy(0, 1);
  sigma_int_y ~ cauchy(0, 1);
}
generated quantities {
  real slope = slope_std / sd_x;
  vector[N_bins] intercept = intercept_std - slope_std * mean_x / sd_x;
  // real sigma_int_x = sigma_int_x_std * sd_x;
}
