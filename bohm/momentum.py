"""momentum distribution in Bohmian mechanics"""

import numpy as np
from numpy import exp, sqrt, pi

def prob_density_func_momentum_bohm_gaussian_1D(p,t,t0,sigma_x,hbar=1,m=1):
    _const = sqrt(2*sigma_x**2/(pi*hbar**2))
    _alpha_t = hbar / (2*m) * (t-t0)
    _z0 = sigma_x**2 + _alpha_t * 1.0j
    return _const * abs(_z0)/abs(_alpha_t) \
            * exp(-2*sigma_x**2 / hbar**2 * (abs(_z0)/abs(_alpha_t))**2 * p**2)

def momentum_for_gaussian_packet(x, t, t0, mu_x, sigma_x, hbar=1.0, m=1.0):
    _alpha_t = hbar / (2*m) * (t - t0)
    _sigma_x_t = sigma_x**2 + 1.j * _alpha_t
    _abs_sigma_x_t = np.real(np.conj(_sigma_x_t) * _sigma_x_t)
    _p_x_t = hbar * _alpha_t / (2*_abs_sigma_x_t) * (x - mu_x)
    return _p_x_t

def dx_p_for_gaussian_packet(x, t, t0, mu_x, sigma_x, hbar=1.0, m=1.0):
    _alpha_t = hbar / (2*m) * (t - t0)
    _sigma_x_t = sigma_x**2 + 1.j * _alpha_t
    _abs_sigma_x_t = np.real(np.conj(_sigma_x_t) * _sigma_x_t)
    _dx_p = hbar * _alpha_t / (2*_abs_sigma_x_t)
    return _dx_p


