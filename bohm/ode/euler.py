"""ODE solver by Euler method"""

from numbers import Real

"""
# notes

$$a_1 \equiv 
1 - h_i^{guess}\partial_{x}F|_{\left(t_{i},x_{i}\right)}$$

$$a_2 \equiv 
1 - h_i^{guess}\partial_{x}F|_{\left(t_{i}+h_{i}^{guess},x_{i}\right)}$$


$a_1$ and $a_2$ should be positive.

"""


def get_stepsize(ti, xi, Fx, dxFx, hi_max,
                gamma = 0.8, hi_min = 0.005, forward=True):
    """ 
    An appropreiate stepsize is estimated 
    where the implicit Euler method may work.
    
    # Arguments
    - `ti`: given time point
    - `xi`: x(ti), given value at given time point `ti`
    - `Fx`: 'F(x,t)=dx(t)/dt', time derivative of function 'x(t)'
    - `dxFx`: 'x' derivative of `Fx`
    - `hi_max`: upper bound of stepsize
    - `gamma`: 0 < float < 1
    - `hi_min`: lower bound of stepsize
    - `forward`: time direction. the timestep is positive when `forward==True`
    """
    
    if not forward: raise NotImplementedError(
        "backward time flow hasn't been dealt yet")
    assert isinstance(gamma, Real) and (0 < gamma) and (gamma < 1)
    
    _hi_guess = None
    
    _dxFx_tixi = dxFx(ti, xi)
    if _dxFx_tixi > 0: _hi_guess = min(gamma / _dxFx_tixi, hi_max)
    else: _hi_guess = hi_max
    assert isinstance(_hi_guess, Real)
    assert (1 - _hi_guess * _dxFx_tixi) > 0
    
    _a2 = None
    while _hi_guess > hi_min:
        _a2 = 1 - _hi_guess * dxFx(ti+_hi_guess, xi)
        if _a2 > 0: break
        _hi_guess *= gamma
    if (_hi_guess <= hi_min): 
        raise Exception("hi_min reached - couldn't find good stepsize")
    _hi = _hi_guess
    
    return _hi



import numpy as np
from scipy.optimize import root

def back_euler(t0, x0, t_max, Fx, dxFx, hi_max, **stepsize_kwarg):
    _t_list = [t0]
    _x_list = [x0]

    while _t_list[-1] < t_max:
        _ti, _xi = _t_list[-1], _x_list[-1]
        _hi = get_stepsize(_ti, _xi, Fx, dxFx, hi_max, **stepsize_kwarg)
        _t_next = _ti + _hi
        def _Gx(x): return (x - _xi) - _hi * Fx(_t_next, x)
        _sol = root(_Gx, _xi)  # jacobian may be supplied
        if _sol.success and _sol.x.size == 1: _x_next, = _sol.x
        else: raise Exception(
            "unique root search failed due to '{}'".format(_sol.message))
        _t_list.append(_t_next), _x_list.append(_x_next)
    
    _t_arr, _x_arr = np.array(_t_list), np.array(_x_list)
    return _t_arr, _x_arr



