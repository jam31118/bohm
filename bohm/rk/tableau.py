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
        _good_A_c = np.all(np.isclose(c, np.sum(A, axis=1)))
        _good_b = np.isclose(b.sum(), 1.0)
        _good_b_star = True
        if b_star is not None: _good_b_star = np.isclose(b_star.sum(), 1.0)
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

## Fehlberg
N_s = 6
A = np.zeros((N_s,N_s), dtype=float)
A[1,:1] = 0.25
A[2,:2] = [3/32, 9/32]
A[3,:3] = [1932/2197,-7200/2197,7296/2197]
A[4,:4] = [439/216, -8, 3680/513, -845/4104]
A[5,:5] = [-8/27, 2, -3544/2565, 1859/4104, -11/40]
b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55], dtype=float)
b_star = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0], dtype=float)
c = np.array([0.0, 0.25, 0.375, 12/13, 1.0, 0.5], dtype=float)
b_table_Fehlberg = Butcher_Tableau(A, b, c, b_star=b_star)

## Cash-Karp
N_s = 6
A = np.zeros((N_s,N_s), dtype=float)
A[1,:1] = 0.2
A[2,:2] = [3/40, 9/40]
A[3,:3] = [3/10, -9/10, 6/5]
A[4,:4] = [-11/54, 5/2, -70/27, 35/27]
A[5,:5] = [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]
b = np.array([37/378, 0, 250/621, 125/594, 0, 512/1771], dtype=float)
b_star = np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4], dtype=float)
c = np.array([0.0, 0.2, 0.3, 0.6, 1.0, 0.875], dtype=float)
b_table_Cash_Karp = Butcher_Tableau(A, b, c, b_star=b_star)

## Dormand-Prince
N_s = 7
A = np.zeros((N_s,N_s), dtype=float)
A[1,:1] = 0.2
A[2,:2] = [3/40, 9/40]
A[3,:3] = [44/45, -56/15, 32/9]
A[4,:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
A[5,:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
A[6,:6] = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)
b_star = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)
c = np.array([0.0, 0.2, 0.3, 0.8, 8/9, 1.0, 1.0], dtype=float)
b_table_Dormand_Prince = Butcher_Tableau(A, b, c, b_star=b_star)
