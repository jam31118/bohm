"""Trajectory-related objects are defined"""

from os import path

from scipy.integrate import ode
import numpy as np

from ntype import is_real_number, is_integer_valued_real


#### TO IMPLEMENT ####
## consider when all keyword arguments havn't been provided.

## Does ode solver of scipy works for complex number also?
## .. If so, let the initial_value be complex also.

class Trajectory(ode):
    def __init__(self, ndim=None, dydt=None, initial_value=None, t0=None, t_max=None,
                 integrator='dopri5', **integrator_kwargs):
        """
        ## Arguments
        #
        # ndim
        : the number of independent variables
        - 'ndim' is same as the number of elements in 'initial_value'
        - 'ndim' is same as the number of elements 
        .. in the returned array(or list) from 'dydt'
        - Although the default value of this argument is None,
        .. it should not be None and given by the caller.
        .. The default value is set to None in order to
        .. allow pure keyword argument syntax
        #
        # initial_value
        : initial values of independent variables (i.e. except time)
        - Type: array-like, scalar
        """

        ## Check input arguments and assign it to member variables if needed.
        assert ndim is not None
        assert is_integer_valued_real(ndim)
        self.ndim = int(ndim)

        if dydt is not None: assert callable(dydt)
        self.dydt = dydt

        assert type(integrator) is str
        self.initial_value = None
        if initial_value is not None:
            if np.iterable(initial_value):
                self.initial_value = initial_value
            elif is_real_number(initial_value):
                self.initial_value = [initial_value]  # Make a scalar into a list
            else: raise TypeError("'initial_value' should be of type either iterable or scalar numerics")

        for arg in [t0, t_max]:
            if arg is not None: assert is_real_number(arg)

        self.t0 = t0
        self.t_max = t_max

        self.integrator_name = integrator

        if self.dydt is not None:
            super().__init__(self.dydt)
            self.set_integrator(self.integrator_name, **integrator_kwargs)
            self.sol = []
            self.set_solout(lambda t, y: self.sol.append([t, *y]))

            self.initial_value_is_set = False
            if (self.initial_value is not None) and (self.t0 is not None):
                self.set_initial_value(self.initial_value, t=self.t0)
                self.initial_value_is_set = True

        self.array = None

        self.data_file_path = None

    def set_initial_value(self, *args, **kwargs):
        assert self.dydt is not None
        super().set_initial_value(*args, **kwargs)
        self.initial_value_is_set = True

    def propagate(self, t_max=None, keep_solution=False):
        ## Check prerequisites
        assert self.dydt is not None
        assert self.initial_value_is_set

        assert type(keep_solution) is bool

        if t_max is None:
            assert self.t_max is not None
            t_max = self.t_max
        else: assert is_real_number(t_max)

        super().integrate(t_max)

        ## [TODO] Implement splited-step intergation
        ## self.max_step * num_of_timepoints_per_intergation ..


        if super().successful():
            self.array = np.array(self.sol)
            if not keep_solution:
                del self.sol
                self.sol = None
        else: raise Exception("Solving ODE failed.")

    def __getitem__(self, index_exp):
        #if not isinstance(self.array, np.ndarray):
        #    raise Exception("'self.array' should be array-like. Check whether the ODE had been solved.")
        self._check_if_trajectory_is_ready()
        return self.array[index_exp]

    def get_t(self):
        assert isinstance(self.array, np.ndarray)
        t = self.array[:,0]
        return t

    def _check_if_trajectory_is_ready(self):
        if not isinstance(self.array, np.ndarray):
            raise TypeError("'self.array' should be array-like. Check whether the ODE had been solved.")
    
    @classmethod
    def from_file(cls, file_path):
        assert path.isfile(file_path)
        loaded_array = None
        try: loaded_array = np.loadtxt(file_path, dtype=np.double)
        except: 
            print("Attempt to load trajectory data from '%s'" % file_path)
            raise IOError("Loading trajectory data from file have been failed.")
        
        ## Check properties of 'loaded_array'
        #expected_num_of_columns = self.ndim + 1  # number of independent variables + time varible
        expected_dimension_of_array = 2   # rows for time, columns for each variables
        assert isinstance(loaded_array, np.ndarray)
        assert loaded_array.ndim == expected_dimension_of_array
        #assert loaded_array.shape[1] == expected_num_of_columns


        ## Extract required member variables
        ndim = loaded_array.shape[1] - 1
        initial_value = loaded_array[0,1:]
        t0, t_max = loaded_array[0,0], loaded_array[-1,0]
        traj_obj = cls(ndim=ndim, initial_value=initial_value, t0=t0, t_max=t_max)
        traj_obj.array = loaded_array
        traj_obj.data_file_path = file_path
        return traj_obj


class Trajectory_Polar_Box(Trajectory):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

    def get_x(self):
        #assert isinstance(self.array, np.ndarray)
        self._check_if_trajectory_is_ready()
        rho = self.array[:,1]
        phi = self.array[:,2]
        x = rho * np.cos(phi)
        return x

    def get_y(self):
        #assert isinstance(self.array, np.ndarray)
        self._check_if_trajectory_is_ready()
        rho = self.array[:,1]
        phi = self.array[:,2]
        y = rho * np.sin(phi)
        return y


class Trajectory_Spherical_Box(Trajectory):
    """signature: (dydt=None, initial_value=None, t0=None, t_max=None, 
    integrator='dopri5', **integrator_kwargs) -> object"""
    def __init__(self, **kwargs):
        ndim = None
        expected_ndim = 3
        if 'ndim' in kwargs.keys():
            ndim = kwargs.pop('ndim')
            assert ndim == expected_ndim
        else: ndim = expected_ndim
        super().__init__(ndim=ndim, **kwargs)

    def get_x(self):
        self._check_if_trajectory_is_ready()
        rho = self.array[:,1]
        theta = self.array[:,2]
        phi = self.array[:,3]
        x = rho * np.sin(theta) * np.cos(phi)
        return x

    def get_y(self):
        self._check_if_trajectory_is_ready()
        rho = self.array[:,1]
        theta = self.array[:,2]
        phi = self.array[:,3]
        y = rho * np.sin(theta) * np.sin(phi)
        return y
    
    def get_z(self):
        self._check_if_trajectory_is_ready()
        rho = self.array[:,1]
        theta = self.array[:,2]
        z = rho * np.cos(theta)
        return z
