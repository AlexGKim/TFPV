functions {
  vector r_prime(vector rho, vector theta, vector x0, vector y0, vector i) {
    vector x = rho*cos(theta)+x0;
    vector y = rho*sin(theta)+y0;
    vector cosi = cos(i);
    return sqrt(x*x + y*y/cosi/cosi);
  }

  vector tantheta_prime(vector rho, vector theta, vector x0, vector y0, vector i) {
    vector x = rho*cos(theta)+x0;
    vector y = rho*sin(theta)+y0;
    vector cosi = cos(i);
    return y/x/cosi;
  }

  vector v_prime(vector Vmax, vector rprime, vector Rturn, vector alpha) {
    return Vmax*rprime/pow(pow(Rturn,alpha)+pow(rprime,alpha),1/alpha);
  }

  vector v_prime_approx(vector Vmax_Rturn, vector rprime) {
    return Vmax_Rturn * rprime;
  }

  vector v_perp(vector rho, vector theta, vector x0, vector y0, vector i, vector Vmax, vector Rturn, vector alpha) {
    vector r_p = r_prime(rho, theta, x0, y0, i);
    vector ttp = tantheta_prime(rho, theta, x0, y0, i);
    return v_prime(Vmax, r_p, Rturn, alpha) cos(atan(ttp))* sin(i);
  }  

  vector v_perp_approx(vector rho, vector theta, vector x0, vector y0, vector i, vector Vmax_Rturn) {
    vector r_p = r_prime(rho, theta, x0, y0, i);
    vector ttp = tantheta_prime(rho, theta, x0, y0, i);
    return v_prime(Vmax_Rturn, r_p) cos(atan(ttp))* sin(i);
  }    
}

data {
  int<lower=0> N;
  vector[N] i;
  vector[N] alpha;
  vector[N] rho;

  vector[N] deltaV;
  vector[N] deltaV_noise;
}

transformed data {
  // Kelly finds standard deviation between 14.2 deg between MANGA and SGA
  // real angle_dispersion_deg = 14.2;
  real angle_dispersion_deg = 4.217219770730775;
  real angle_dispersion = angle_dispersion_deg/180*pi();
}

parameters {
  real<lower=0> sigma
  vector[N] x0_raw;
  vector[N] y0_raw;
  vector[N] theta;
  vector[N]<lower=0> Vmax_Rturn;

  vector<lower=-atan(pi()/2), upper=atan(pi()/2)>[N] theta_unif;   
}

model {

  real _sig = sigma/sqrt(2);
  vector[N] x0 = x0_raw*_sig;
  vector[N] y0 = y0_raw*_sig;
  vector[N] theta = angle_dispersion * tan(theta_unif);
  vector[N] mn = v_perp(rho, theta, x0, y0, i, Vmax_Rturn) + v_perp(rho, theta+pi(), x0, y0, i, Vmax_Rturn);
  deltaV ~ normal(mn, deltaV_noise);
}