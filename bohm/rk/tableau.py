"""A toolset for Runge-Kutta method"""


import numpy as np

class Butcher_Tableau(object):
    def __init__(self, A, b, c, b_star=None):
        """Construct a Butcher tableau
        
        # Notation
        - `N_s` : the number of stage(s) in the method
        
        # Function Argument
        - `A`: 
          - if low-triangular: explicit method
          - else: implicit method, which is more generic.
        - `b_star`: [ None | numpy.ndarray with shape (N_s,) ]
          - if None:
            : fixed-step-size method
          - if numpy.ndarray:
            : adaptive-step-size method based on the error
            : .. between the approximate solution from `b` and `b_star`
            
        # Reference
        - Notation of the Butcher tableau is given on the wiki page
          for the explicit Runge-Kutta method:
          https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
        """
        for arg in (A,b,c): assert isinstance(arg, np.ndarray)
        assert (c.ndim == 1) and (b.ndim == 1) and (A.ndim == 2)
        assert (c.size == b.size) and (A.shape == (c.size, c.size))
        _N_s = c.size
        if b_star is not None:
            assert isinstance(b_star, np.ndarray)
            b_star.shape = (_N_s,)
        
        assert self.has_consistency(A, b, c, b_star)
        
        self.A, self.b, self.c, self.b_star = A, b, c, b_star
        self.N_s = _N_s
        
        
    @staticmethod
    def has_consistency(A, b, c, b_star):
        _good_A_c = np.all(c == np.sum(A, axis=1))
        _good_b = b.sum() == 1.0
        _good_b_star = True
        if b_star is not None: _good_b_star = b_star.sum() == 1.0
        _all_good = _good_A_c and _good_b and _good_b_star
        return _all_good


### Predefined tables

## Euler-Ralston
N_s = 2
A = np.zeros((N_s,N_s), dtype=float)
A[1,0] = 2.0 / 3.0
b = np.array([0.25, 0.75], dtype=float)
b_star = np.array([1.0, 0.0], dtype=float)
c = np.array([0.0, 2.0 / 3.0], dtype=float)
b_table_Euler_Ralston = Butcher_Tableau(A, b, c, b_star=b_star)

## Heun-Ralston
N_s = 2
A = np.zeros((N_s,N_s), dtype=float)
A[1,0] = 1.0
b = np.array([0.5, 0.5], dtype=float)
b_star = np.array([1.0, 0.0], dtype=float)
c = np.array([0.0, 1.0], dtype=float)
b_table_Heun_Ralston = Butcher_Tableau(A, b, c, b_star=b_star)

## Bogacki-Shampine
N_s = 4
A = np.zeros((N_s,N_s), dtype=float)
A[1,:1] = 0.5
A[2,:2] = [0.0, 0.75]
A[3,:3] = [2.0/9.0, 1.0/3.0, 4.0/9.0]
b = np.array([2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0], dtype=float)
b_star = np.array([7.0/24.0, 1.0/4.0, 1.0/3.0, 1.0/8.0], dtype=float)
c = np.array([0.0, 0.5, 0.75, 1.0], dtype=float)
b_table_Bogacki_Shampine = Butcher_Tableau(A, b, c, b_star=b_star)


