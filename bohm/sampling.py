"""collection of statistical sampling"""

import numpy as np

def to_linear_cdf(x, f):
    
    assert isinstance(x, np.ndarray)
    
    _mid_arr = 0.5 * (f[1:] + f[:-1])
    _norm = np.sum(_mid_arr * np.diff(x))
    _pdf = _mid_arr / _norm

    _pdf_dx = np.empty(x.size)
    _pdf_dx[0] = 0.0
    _pdf_dx[1:] = _pdf * np.diff(x)
    _cdf = np.add.accumulate(_pdf_dx)
    
    return _cdf


def quantile(x, f, q, return_cdf=False):
    
    assert np.all((0 <= q) & (q <= 1.0))

    _cdf = to_linear_cdf(x, f)
    
    _xq_arr, _xq = np.empty_like(q), None
    for _i in range(q.size):
        _q0 = q[_i]
        if _q0 == 1.0: _xq = 1.0
        else:
            _ind = np.where(_cdf > _q0)[0][0] - 1
            _p = (_q0 - _cdf[_ind]) / (_cdf[_ind+1] - _cdf[_ind])
            _xq = _p * x[_ind+1] + (1 - _p) * x[_ind]
        _xq_arr[_i] = _xq
    
    if not return_cdf: return _xq_arr
    else: return (_xq_arr, _cdf)


