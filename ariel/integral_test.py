import numpy as np
from scipy.special import owens_t, ndtr
from scipy.stats import multivariate_normal
from scipy.stats import norm


from scipy.integrate import quad


def binormal_cdf(z, rho):
    """
    Bivariate standard normal CDF Φ2(z1, z2; rho), matching the Stan implementation
    (via Owen's T).
    """
    z1, z2 = float(z[0]), float(z[1])
    rho = float(rho)

    # avoid denom=0 at |rho|=1
    rho = np.clip(rho, -1.0 + 1e-15, 1.0 - 1e-15)

    if z1 == 0.0 and z2 == 0.0:
        return 0.25 + np.arcsin(rho) / (2.0 * np.pi)

    denom = np.sqrt((1.0 + rho) * (1.0 - rho))

    term1 = (0.25 if z2 > 0.0 else -0.25) if (z1 == 0.0) else owens_t(
        z1, (z2 / z1 - rho) / denom
    )
    term2 = (0.25 if z1 > 0.0 else -0.25) if (z2 == 0.0) else owens_t(
        z2, (z1 / z2 - rho) / denom
    )

    z1z2 = z1 * z2
    delta = (z1z2 < 0.0) or (z1z2 == 0.0 and (z1 + z2) < 0.0)

    return 0.5 * (ndtr(z1) + ndtr(z2) - float(delta)) - term1 - term2


def selection_probability_uniform_yTF(
    y_tf_min,
    y_tf_max,
    y_hat_max_i,
    sigma1_i,
    sigma2_i,
    s,
    c,
    bar_s,
    bar_c,
    quad_opts=None,
    return_error=False,
):
    """
    Compute the "limits" case:
        P(S_i=1 | θ) = (1/(y_tf_max - y_tf_min)) * ∫_{y_tf_min}^{y_tf_max} Φ2(-α, β; -ρ) dy_TF

    with
        den  = sqrt(sigma2^2 + bar_s^2 * sigma1^2)
        ρ    = sigma2 / den
        α(y) = (bar_c - ( y - bar_s*(y - c)/s )) / den
        β(y) = (y_hat_max_i - y) / sigma2

    Parameters are floats; sigma1_i,sigma2_i>0; s,bar_s!=0.
    """
    y_tf_min = float(y_tf_min)
    y_tf_max = float(y_tf_max)
    if not (y_tf_max > y_tf_min):
        raise ValueError("Require y_tf_max > y_tf_min")

    y_hat_max_i = float(y_hat_max_i)
    sigma1_i = float(sigma1_i)
    sigma2_i = float(sigma2_i)
    s = float(s)
    c = float(c)
    bar_s = float(bar_s)
    bar_c = float(bar_c)

    if sigma1_i <= 0 or sigma2_i <= 0:
        raise ValueError("sigma1_i and sigma2_i must be > 0")
    if s == 0:
        raise ValueError("s must be nonzero")
    if bar_s == 0:
        raise ValueError("bar_s must be nonzero")

    den = np.sqrt(sigma2_i**2 + (bar_s**2) * (sigma1_i**2))
    rho = sigma2_i / den  # constant in y_TF

    quad_opts = {} if quad_opts is None else dict(quad_opts)

    def integrand(y_tf):
        y_tf = float(y_tf)
        alpha = (bar_c - (y_tf - bar_s * (y_tf - c) / s)) / den
        beta = (y_hat_max_i - y_tf) / sigma2_i
        return binormal_cdf((-alpha, beta), -rho)

    integral, err = quad(integrand, y_tf_min, y_tf_max, **quad_opts)
    val = integral / (y_tf_max - y_tf_min)

    return (val, err / (y_tf_max - y_tf_min)) if return_error else val

def selection_probability_uniform_yTF_gl(
    y_tf_min,
    y_tf_max,
    y_hat_max_i,
    sigma1_i,
    sigma2_i,
    s,
    c,
    bar_s,
    bar_c,
    n_quad=15,
):
    """
    Compute:
        P(S_i=1 | θ) = (1/(y_tf_max - y_tf_min))
            * ∫_{y_tf_min}^{y_tf_max} Φ2(-α(y), β(y); -ρ) dy

    using Gauss-Legendre quadrature with n_quad nodes and binormal_cdf.

    Parameters
    ----------
    y_tf_min, y_tf_max : float
        Bounds of the uniform prior on y_TF.
    y_hat_max_i : float
        Observation-level parameter.
    sigma1_i, sigma2_i : float > 0
        Standard deviations.
    s, bar_s : float != 0
        Scale parameters.
    c, bar_c : float
        Location parameters.
    n_quad : int
        Number of Gauss-Legendre quadrature nodes (default 15).

    Returns
    -------
    float
        The selection probability, clipped to [0, 1].
    """
    y_tf_min = float(y_tf_min)
    y_tf_max = float(y_tf_max)
    if y_tf_max <= y_tf_min:
        raise ValueError("Require y_tf_max > y_tf_min")

    y_hat_max_i = float(y_hat_max_i)
    sigma1_i = float(sigma1_i)
    sigma2_i = float(sigma2_i)
    s = float(s)
    c = float(c)
    bar_s = float(bar_s)
    bar_c = float(bar_c)

    if sigma1_i <= 0 or sigma2_i <= 0:
        raise ValueError("sigma1_i and sigma2_i must be > 0")
    if s == 0:
        raise ValueError("s must be nonzero")
    if bar_s == 0:
        raise ValueError("bar_s must be nonzero")

    den = np.sqrt(sigma2_i**2 + bar_s**2 * sigma1_i**2)
    neg_rho = -sigma2_i / den

    slope = 1.0 - bar_s / s
    offset = bar_c - bar_s * c / s

    half_width = 0.5 * (y_tf_max - y_tf_min)
    midpoint = 0.5 * (y_tf_max + y_tf_min)

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)

    accum = 0.0
    for k in range(n_quad):
        y_tf = midpoint + half_width * nodes[k]
        neg_alpha = -(offset - slope * y_tf) / den
        beta = (y_hat_max_i - y_tf) / sigma2_i
        accum += weights[k] * binormal_cdf((neg_alpha, beta), neg_rho)

    val = accum / 2.0
    return float(np.clip(val, 0.0, 1.0))

# Example:
if __name__ == "__main__":
    p, perr = selection_probability_uniform_yTF(
        y_tf_min=-2.0,
        y_tf_max=2.0,
        y_hat_max_i=1.2,
        sigma1_i=0.3,
        sigma2_i=0.2,
        s=1.5,
        c=0.1,
        bar_s=1.0,
        bar_c=0.0,
        quad_opts={"epsabs": 1e-10, "epsrel": 1e-10, "limit": 200},
        return_error=True,
    )
    print("P(S=1|θ) =", p, "quad_err≈", perr)

    p = selection_probability_uniform_yTF_gl(
        y_tf_min=-2.0,
        y_tf_max=2.0,
        y_hat_max_i=1.2,
        sigma1_i=0.3,
        sigma2_i=0.2,
        s=1.5,
        c=0.1,
        bar_s=1.0,
        bar_c=0.0,
        n_quad=15,
    )
    print("P(S=1|θ) =", p)


# def binormal_cdf(z, rho):
#     z1, z2 = float(z[0]), float(z[1])

#     if z1 == 0.0 and z2 == 0.0:
#         return 0.25 + np.arcsin(rho) / (2.0 * np.pi)

#     denom = np.sqrt((1.0 + rho) * (1.0 - rho))

#     term1 = (0.25 if z2 > 0.0 else -0.25) if (z1 == 0.0) else owens_t(
#         z1, (z2 / z1 - rho) / denom
#     )
#     term2 = (0.25 if z1 > 0.0 else -0.25) if (z2 == 0.0) else owens_t(
#         z2, (z1 / z2 - rho) / denom
#     )

#     z1z2 = z1 * z2
#     delta = (z1z2 < 0.0) or (z1z2 == 0.0 and (z1 + z2) < 0.0)

#     return 0.5 * (ndtr(z1) + ndtr(z2) - float(delta)) - term1 - term2


# def compare_with_scipy_mvn_cdf(n=10, seed=0, z_scale=2.0, rho_max=0.95, print_table=True):
#     """
#     Compare binormal_cdf((z1,z2), rho) to scipy.stats.multivariate_normal.cdf
#     for n random test cases.

#     Returns a list of dicts with inputs and results.
#     """
#     rng = np.random.default_rng(seed)
#     results = []

#     for i in range(n):
#         z1, z2 = rng.normal(loc=0.0, scale=z_scale, size=2)
#         rho = rng.uniform(low=-rho_max, high=rho_max)

#         ours = float(binormal_cdf((z1, z2), rho))

#         cov = np.array([[1.0, rho],
#                         [rho, 1.0]])
#         scipy_val = float(multivariate_normal(mean=[0.0, 0.0], cov=cov).cdf([z1, z2]))

#         results.append({
#             "i": i,
#             "z1": z1,
#             "z2": z2,
#             "rho": rho,
#             "binormal_cdf": ours,
#             "scipy_mvn_cdf": scipy_val,
#             "abs_diff": abs(ours - scipy_val),
#             "rel_diff": abs(ours - scipy_val) / max(1e-15, abs(scipy_val)),
#         })

#     if print_table:
#         header = f"{'i':>2}  {'z1':>10}  {'z2':>10}  {'rho':>8}  {'ours':>14}  {'scipy':>14}  {'abs_diff':>12}"
#         print(header)
#         print("-" * len(header))
#         for r in results:
#             print(f"{r['i']:2d}  {r['z1']:10.6f}  {r['z2']:10.6f}  {r['rho']:8.4f}  "
#                   f"{r['binormal_cdf']:14.10f}  {r['scipy_mvn_cdf']:14.10f}  {r['abs_diff']:12.3e}")

#     return results


# # Example:
# compare_with_scipy_mvn_cdf(n=10, seed=123)