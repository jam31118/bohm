import numpy as np
from numbers import Real

def get_sol_linear(t0, x0, c0, c1):
    for _var in (t0, x0, c0, c1): assert isinstance(_var, Real)
    if c1 == 0: return lambda t: c0 * (t - t0) + x0
    else: return lambda t: (x0 + c0/c1) * np.exp(c1 * (t - t0)) - c0/c1

