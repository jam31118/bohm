"""momentum distribution in Bohmian mechanics"""

from numpy import exp, sqrt, pi

def prob_density_func_momentum_bohm_gaussian_1D(p,t,t0,sigma_x,hbar=1,m=1):
    _const = sqrt(2*sigma_x**2/(pi*hbar**2))
    _alpha_t = hbar / (2*m) * (t-t0)
    _z0 = sigma_x**2 + _alpha_t * 1.0j
    return _const * abs(_z0)/abs(_alpha_t) \
            * exp(-2*sigma_x**2 / hbar**2 * (abs(_z0)/abs(_alpha_t))**2 * p**2)
