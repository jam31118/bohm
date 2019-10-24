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
    if not isinstance(_hi_guess, Real): raise TypeError("`_hi_guess` should be of type `Real` but now is of type `{}`".format(type(_hi_guess)))
    assert (1 - _hi_guess * _dxFx_tixi) > 0
    
    _a2 = None
    while _hi_guess > hi_min:
        _a2 = 1 - _hi_guess * dxFx(ti+_hi_guess, xi)
        if _a2 > 0: break
        _hi_guess *= gamma
    if (_hi_guess <= hi_min): 
        raise Exception("hi_min reached - couldn't find good stepsize")
    _hi = min(_hi_guess,hi_max)  # may not be valid when forward==False
    
    return _hi




import numpy as np
from scipy.optimize import root

def back_euler(t0, x0, t_max, Fx, dxFx, hi_max, 
                    t_arr=None, **stepsize_kwarg):
    
    if t_arr is None: t_arr = np.linspace(t0, t_max, int((t_max - t0 + hi_max) / hi_max))
    else: assert isinstance(t_arr, np.ndarray) and (t_arr.ndim == 1)
    assert t0 == t_arr[0]

    _x_arr = np.empty_like(t_arr, dtype=float)
    _x_arr[0] = x0
    _t, _x = t0, x0
    
    for _i in range(1,t_arr.size):
        
        _t_target = t_arr[_i]
        
        while _t < _t_target:
            _h = get_stepsize(_t, _x, Fx, dxFx, hi_max, 
                                   **stepsize_kwarg)

            _t_next = _t + _h
            if _t_next > _t_target:
                _h = _t_target - _t
                _t_next = _t_target

            if np.abs(Fx(_t,_x)) < 1e-14:
                _x_next = _x + _h * Fx(_t,_x)
            else:
                def _Gx(x): return (x - _x) - _h * Fx(_t_next, x)
                _sol = root(_Gx, _x)  # jacobian may be supplied
                if _sol.success and _sol.x.size == 1: _x_next, = _sol.x
                else:
                    print("_i: {}, _h: {}, _t: {}, _t_next: {}, _t_target: {}".format(_i, _h, _t, _t_next, _t_target))
                    print("t0: {}, x0: {}, _x: {}".format(t0, x0, _x))
                    print("Fx: {}, dxFx: {}".format(Fx(_t,_x), dxFx(_t,_x)))
                    print("sol: ", _sol)
                    raise Exception(
                        "unique root search failed due to '{}'".format(_sol.message))

            _t, _x = _t_next, _x_next
            
        _x_arr[_i] = _x_next

    return t_arr, _x_arr




from math import sqrt
from numbers import Real

def get_stepsize_quad(ti, xi, Fx, dxFx, dx2Fx, hi_max,
                     gamma = 0.8, hi_min = 0.0001, forward=True):
    
    if not forward: raise NotImplementedError(
        "backward time flow hasn't been dealt yet")
    assert isinstance(gamma, Real) and (0 < gamma) and (gamma < 1)
    
    _hi_guess = None
    
    ## initial guess of stepsize
    _F_0, _dxFx_0, _dx2Fx_0 = Fx(ti,xi), dxFx(ti,xi), dx2Fx(ti,xi)
    _a1_0 = _dxFx_0*_dxFx_0 - 2.0 * _dx2Fx_0 * _F_0
    _a2_0 = _dxFx_0
    _D_D = _a2_0 * _a2_0 - _a1_0
    
    if _a1_0 > 0:
        if _a2_0 <= 0: _hi_guess = hi_max
        else:
            if _D_D >= 0: _hi_guess = gamma * _a2_0 / _a1_0
            else: _hi_guess = hi_max
    elif _a1_0 < 0: _hi_guess = gamma*(_a2_0+sqrt(_D_D)) / (2.0*_a1_0)
    elif _a1_0 == 0:
        if _a2_0 > 0: _hi_guess = gamma / (2.0*_a2_0)
        else: _hi_guess = hi_max
    else: raise Exception("Unexpected")
    
    assert isinstance(_hi_guess, Real)
    _D_hi_0 = _a1_0 * _hi_guess*_hi_guess - 2.0 * _a2_0 * _hi_guess + 1
    assert _D_hi_0 > 0

    
    ## fine-tune
    _a1, _a2 = _a1_0, _a2_0
    while _hi_guess > hi_min:
        _D_hi = _a1 * _hi_guess*_hi_guess - 2.0 * _a2 * _hi_guess + 1
        if _D_hi > 0: break
        print("reduced")
        _ti = ti + _hi_guess
        _F, _dxFx, _dx2Fx = Fx(_ti,xi), dxFx(_ti,xi), dx2Fx(_ti,xi)
        _a1 = _dxFx*_dxFx - 2.0 * _dx2Fx * _F
        _a2 = _dxFx
        _hi_guess *= gamma
    
    if (_hi_guess <= hi_min):
        raise Exception("hi_min reached-couldn't find good stepsize")
    _hi = min(_hi_guess, hi_max)
    
    return _hi

import numpy as np
from scipy.optimize import root

def back_euler_quad(t0, x0, t_max, Fx, dxFx, dx2Fx, hi_max, 
                    t_arr=None, **stepsize_kwarg):
    
    if t_arr is None: t_arr = np.array((t0, t_max))
    else: assert isinstance(t_arr, np.ndarray) and (t_arr.ndim == 1)
    assert t0 == t_arr[0]

    _x_arr = np.empty_like(t_arr, dtype=float)
    _x_arr[0] = x0
    _t, _x = t0, x0
    
    for _i in range(1,t_arr.size):
        
        _t_target = t_arr[_i]
        
        while _t < _t_target:
            _h = get_stepsize_quad(_t, _x, Fx, dxFx, dx2Fx, hi_max, 
                                   **stepsize_kwarg)
            _t_next = _t + _h
            if _t_next > _t_target:
                _h = _t_target - _t
                _t_next = _t_target
                
            def _Gx(x): return (x - _x) - _h * Fx(_t_next, x)
            _sol = root(_Gx, _x)  # jacobian may be supplied
            if _sol.success and _sol.x.size == 1: _x_next, = _sol.x
            else: raise Exception(
                "unique root search failed due to '{}'".format(
                    _sol.message))
            _t, _x = _t_next, _x_next
            
        _x_arr[_i] = _x_next

    return t_arr, _x_arr




#import numpy as np
#from scipy.optimize import root
#
#def back_euler(t0, x0, t_max, Fx, dxFx, hi_max, **stepsize_kwarg):
#    _t_list = [t0]
#    _x_list = [x0]
#
#    while _t_list[-1] < t_max:
#        _ti, _xi = _t_list[-1], _x_list[-1]
#        _hi = get_stepsize(_ti, _xi, Fx, dxFx, hi_max, **stepsize_kwarg)
#        _t_next = _ti + _hi
#        def _Gx(x): return (x - _xi) - _hi * Fx(_t_next, x)
#        _sol = root(_Gx, _xi)  # jacobian may be supplied
#        if _sol.success and _sol.x.size == 1: _x_next, = _sol.x
#        else: raise Exception(
#            "unique root search failed due to '{}'".format(_sol.message))
#        _t_list.append(_t_next), _x_list.append(_x_next)
#    
#    _t_arr, _x_arr = np.array(_t_list), np.array(_x_list)
#    return _t_arr, _x_arr
#


