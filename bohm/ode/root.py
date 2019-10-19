"""routine for root finding"""

def iter_newton(ti, xi, hi, Fx, dxFx, thres=1e-10, N_itermax=10):
    _x = xi
    _t_next = ti + hi
    _x_converged = False
    for _i in range(N_itermax):
        _Gx = (_x - xi) - hi * Fx(_t_next, _x)
        _dxGx = 1.0 - hi * dxFx(_t_next, _x)
        assert _dxGx != 0
        _x_next = _x - _Gx / _dxGx
        _x_converged = abs(_x_next - _x) < thres
        print("_x_next - _x: {}".format(_x_next - _x))
        if _x_converged: break
        _x = _x_next
    if not _x_converged or (_i >= N_itermax - 1): 
        raise Exception("convergence might have been failed")
    return _x_next
