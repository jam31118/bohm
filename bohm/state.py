"""Several types of State Function objects are defined."""

from os import path
from struct import unpack

import numpy as np
from scipy.special import sph_harm

from tdse.grid import Grid, Grid_Polar, Grid_Spherical
from tdse.grid import process_index_exp, squeeze_as_possible

from qprop.core import Qprop20


class State_Function(object):
    def __init__(self):
        pass

    def get_norm(self):
        pass

    def normalize(self):
        pass

    def __getitem__(self):
        pass


class State_Function_In_Polar_Box(State_Function):
    def __init__(self, grid_polar):
        """Initialize self with respect to input 'grid_polar'"""
        ## Check and assign input arguments
        assert type(grid_polar) is Grid_Polar
        self.grid = grid_polar

    def get_norm(self, psi):
        r"""Calculate squared norm of this state function in polar coordinate representation.

        # Normalization
        - For polar box boundary condition, the formula is like the following:

        $|A|^2 \approx \Delta{\rho}\Delta{\phi}\sum^{N_{\rho}-1}_{i=0}\sum^{N_{\phi}-1}_{j=0}{\left|\psi(\rho_{i},\phi_{j},t)\right|^2\rho_{i}}$

        where $A$ is a normalization constant

        ## TODO
        - The notion of 'norm' is different from 'squared norm'. Fix it.
        """
        psi_sq = (psi * psi.conj()).real
        norm = np.einsum(psi_sq, [0,1], self.grid.rho.array, [0]).sum()
        norm *= self.grid.rho.delta * self.grid.phi.delta
        return norm

    def normalize(self, psi):
        norm = self.get_norm(psi)
        norm_const = 1.0 / np.sqrt(norm)
        psi_normed = psi * norm_const
        return psi_normed

    def __getitem__(self, index_exp):
        pass


class Analytic_State_Function(State_Function):
    def __init__(self, grid, func, f_kwargs={}, normalize=True):
        """
        ## NOTES
        # Assumption of constant norm
        The norm is assumed to be constant for all time
        based on the fact that this object is intended
        to deal with analytical solution for the TDSE.
        Since the solution of the TDSE preserves its norm,
        the constant norm assumption seems realiable.
        """
        ## Check and assign input arguments
        # 'grid_polar' is checked in super().__init__()
        #super().__init__(grid_polar)
        assert Grid in type(grid).mro()
        if not hasattr(self, 'grid'):
            self.grid = grid

        assert type(f_kwargs) is dict
        self.f_kwargs = f_kwargs

        assert type(normalize) is bool
        self.normalize = normalize

        assert callable(func)
        self.given_func = func

        ## Calculate normalization constant if 'self.normalize' is True
        self.normalization_constant = None
        if self.normalize:
            self.normalization_constant = self._get_normalization_constant(self.given_func, **self.f_kwargs)
        else: self.normalization_constant = 1.0

        self.shape = self.grid.shape

    def _get_normalization_constant(self, func, f_kwargs={}):
        raise NotImplementedError("Please implement this method")

    def get_state_func(self, *args, **kwargs):
        """Add a normalization constant to a given function."""
        return self.normalization_constant * self.given_func(*args, **kwargs)

    def get_value(self, indices):
        coord = self.grid.get_value(indices)
        return self.get_state_func(*coord)

    def __getitem__(self, index_exp):
        coord = self.grid[index_exp]
        return self.get_state_func(*coord)

    def _get_normalization_constant(self, func, f_kwargs={}):
        # [180304 NOTE] Replace [:,:,0] to [...,0]
        # .. modify Grid_Polar code to support '...' expression

        assert type(f_kwargs) is dict
        coord = self.grid[...,0]
        psi_x_t0 = func(*coord, **f_kwargs)
        norm = self.get_norm(psi_x_t0)
        normalization_constant = 1.0 / np.sqrt(norm)
        return normalization_constant



class Analytic_State_Function_In_Polar_Box(Analytic_State_Function, State_Function_In_Polar_Box):
    def __init__(self, grid_polar, func, f_kwargs={}, normalize=True):
        State_Function_In_Polar_Box.__init__(self, grid_polar)
        Analytic_State_Function.__init__(self, grid_polar, func, f_kwargs=f_kwargs, normalize=normalize)



class State_Function_In_Spherical_Box(State_Function):
    def __init__(self, grid_spherical):
        """Initialize self with respect to input 'grid_polar'"""
        ## Check and assign input arguments
        assert type(grid_spherical) is Grid_Spherical
        self.grid = grid_spherical
        self.shape = self.grid.shape

    def get_norm(self, psi):
        psi_sq = (psi * psi.conj()).real
        sin_theta_array = np.sin(self.grid.theta.array)
        rho_array = self.grid.rho.array
        sum_on_rho = self.grid.rho.delta * np.einsum(psi_sq, [0,1,2], rho_array, [0])
        sum_on_rho_theta = self.grid.theta.delta * np.einsum(sum_on_rho, [1,2], sin_theta_array, [1])
        norm_sq = self.grid.phi.delta \
            * (sum_on_rho_theta.sum() - 0.5 * (sum_on_rho_theta[0] + sum_on_rho_theta[-1]))
        return norm_sq


class Analytic_State_Function_In_Spherical_Box(Analytic_State_Function, State_Function_In_Spherical_Box):
    """

    ## Development Notes ##
    The inheritence order is crucial: 'Analytic_State_Function, State_Function_In_Spherical_Box'
    is the proper order. It is because some attributes should be inherited
    not from State_Function_In_Spherical_Box but from Analytic_State_Function.
    However, the picture is not so clear. Thus, we need to identify which attributes should be inherited
    from which parent classes.
    """
    def __init__(self, grid_spherical, func, f_kwargs={}, normalize=True):
        State_Function_In_Spherical_Box.__init__(self, grid_spherical)
        Analytic_State_Function.__init__(self, grid_spherical, func, f_kwargs=f_kwargs, normalize=normalize)




class Gradient_State_Function(object):
    """Base class for gradient of state function"""
    def __init__(self, state_function):
        assert isinstance(state_function, State_Function)
        #assert State_Function in type(state_function).mro()
        for attr in ['__getitem__', 'get_value']:
            assert hasattr(state_function, attr)

        ## Assign member variables
        self.sf = state_function
        self.grid = self.sf.grid

    def __getitem__(self, index_exp):
        pass



class Gradient_State_Function_In_Polar_Box(Gradient_State_Function):
    def __init__(self, state_function):
        ## Check input arguments and assign to member arguments if necessary
        assert State_Function_In_Polar_Box in type(state_function).mro()
        #assert hasattr(state_function, '__getitem__')
        #self.sf = state_function
        #self.grid = self.sf.grid
        Gradient_State_Function.__init__(self, state_function)

        self._gradient_rho_vec = np.vectorize(self._gradient_rho)
        self._gradient_phi_vec = np.vectorize(self._gradient_phi)

    def _gradient_rho(self, i, j, k):
        """Calculate radial component of the gradient of state function ('self.sf')
        at coordinate corresponding to given indices ('i','j','k')

        ## NOTE ##
        # Index correspondence:
        - 'i': rho index
        - 'j': phi index
        - 'k': time index
        """
        last_index_rho = self.grid.rho.N - 1
        grad_rho = complex(1.0 / (2.0 * self.grid.rho.delta))
        if not (i in [0, last_index_rho]):
            grad_rho *= self.sf[i+1,j,k] - self.sf[i-1,j,k]
        elif i == 0:
            grad_rho *= - 3.0 * self.sf[i,j,k] + 4.0 * self.sf[i+1,j,k] - self.sf[i+2,j,k]
        elif i == last_index_rho:
            grad_rho *= - self.sf[i-1,j,k]   # self.sf[i+1,j,k] == self.sf[self.grid.rho.N,j,k] == 0
        else: raise IndexError("Index is out of range: 'i'")

        return grad_rho


    def _gradient_phi(self, i, j, k):
        """Calculate azimuthal component of the gradient of state function ('self.sf')
        at coordinate corresponding to given indices ('i','j','k')

        [NOTE] Index correspondence:
        - 'i': rho index
        - 'j': phi index
        - 'k': time index
        """
        last_index_phi = self.grid.phi.N - 1
        grad_phi = complex(1.0 / (2.0 * self.grid.phi.delta))
        if i != 0:
            grad_phi /= self.grid.rho[i]
            if not (j in [0, last_index_phi]):
                grad_phi *= self.sf[i,j+1,k] - self.sf[i,j-1,k]
            elif j == 0:
                grad_phi *= self.sf[i,j+1,k] - self.sf[i,last_index_phi,k]
            elif j == last_index_phi:
                grad_phi *= self.sf[i,0,k] - self.sf[i,j-1,k]
            else: raise IndexError("Index is out of range: 'j'")
        else:
            grad_phi *= complex(1.0 / (2.0 * self.grid.rho.delta))
            grad_phi_1, grad_phi_2 = None, None
            if not (j in [0, last_index_phi]):
                grad_phi_1 = (self.sf[1,j+1,k] - self.sf[1,j-1,k])
                grad_phi_2 = (self.sf[2,j+1,k] - self.sf[2,j-1,k])
            elif j == 0:
                grad_phi_1 = (self.sf[1,j+1,k] - self.sf[1,last_index_phi,k])
                grad_phi_2 = (self.sf[2,j+1,k] - self.sf[2,last_index_phi,k])
            elif j == last_index_phi:
                grad_phi_1 = (self.sf[1,0,k] - self.sf[1,j-1,k])
                grad_phi_2 = (self.sf[2,0,k] - self.sf[2,j-1,k])
            else: raise IndexError("Index is out of range: 'j'")
            grad_phi *= 4.0 * grad_phi_1 - grad_phi_2

        return grad_phi


    def __getitem__(self, index_exp):
        """[NOTE] It may be a source of error/bug because of the 'index_exp'. See again.
        'index_exp' should support also 'Ellipsis' etc. More general method should be used.
        """

        ndim = 3   # two spatial and one temporal dimension
        assert len(index_exp) == ndim
        slices = tuple(np.arange(self.sf.shape[idx])[index_exp[idx]] for idx in range(ndim))
        coordinates = np.meshgrid(*slices, indexing='ij')

        grad_rho = self._gradient_rho_vec(*coordinates)
        grad_phi = self._gradient_phi_vec(*coordinates)

        result = []
        for arr in [grad_rho, grad_phi]:
            arr = np.squeeze(arr)
            if arr.ndim == 0: arr = complex(arr)
            result.append(arr)

        return tuple(result)




class Gradient_State_Function_In_Spherical_Box(Gradient_State_Function):
    def __init__(self, state_function):
        ## Check input arguments and assign them into member variables
        #assert State_Function_In_Spherical_Box in type(state_function).mro()
        assert isinstance(state_function, State_Function_In_Spherical_Box)
        # Variable such as 'grid' and 'sf'(abbrev. of state function) are assigned into
        # .. member variable of 'self' at the inside of 'Gradient_State_Function's __init__().
        #Gradient_State_Function.__init__(self, state_function)
        super().__init__(state_function)

        ## Define some meta information.
        self.shape = self.grid.shape
        self.ndim = self.grid.ndim

        ## Make member methods as vectorized functions for array-like indexing of gradient values.
        self._gradient_rho_vec = np.vectorize(self._gradient_rho)
        self._gradient_theta_vec = np.vectorize(self._gradient_theta)
        self._gradient_phi_vec = np.vectorize(self._gradient_phi)


    def _gradient_rho(self, i, j, k, l):
        """Calculate radial component (denoted by 'rho') of the gradient of state function ('self.sf')
        at coordinate corresponding to given indices ('i','j','k', 'l')

        Second-order finite difference approximation was used to evaluate gradient value.

        ## NOTE ##
        # Index correspondence:
        - 'i': rho index
        - 'j': theta index
        - 'k': phi index
        - 'l': time index
        """
        last_index_rho = self.grid.rho.N - 1
        grad_rho = complex(1.0 / (2.0 * self.grid.rho.delta))
        if not (i in [0, last_index_rho]):
            # two-sided 2nd order finite difference approximation.
            grad_rho *= self.sf[i+1,j,k,l] - self.sf[i-1,j,k,l]
        elif i == 0:
            # one-sided 2nd order finite difference approximation.
            grad_rho *= - 3.0 * self.sf[i,j,k,l] + 4.0 * self.sf[i+1,j,k,l] - self.sf[i+2,j,k,l]
        elif i == last_index_rho:
            # two-sided 2nd order finite difference approximation.
            grad_rho *= - self.sf[i-1,j,k,l]   # self.sf[i+1,j,k] == self.sf[self.grid.rho.N,j,k] == 0
        else: raise IndexError("Index '%s' is out of range: [%d,%d]" % ('i',0, last_index_rho))

        return grad_rho


    def _partial_diff_theta(self, i, j, k, l):
        """Calculate partial differentiated state function with repsect to 
        polar angle coordniate (denoted by 'theta')

        Second-order finite difference approximation was used to evaluate partial differential value.

        This method takes the boundary effect into account
        by applying one-sided finite difference formula
        """

        last_index_theta = self.grid.theta.N - 1
        result = complex(1.0 / (2.0 * self.grid.theta.delta))
        if j not in [0, last_index_theta]:
            result *= self.sf[i, j+1, k, l] - self.sf[i, j-1, k, l]
        elif j == 0:
            result *= - 3.0 * self.sf[i,j,k,l] + 4.0 * self.sf[i,j+1,k,l] - self.sf[i,j+2,k,l]
        elif j == last_index_theta:
            result *= 3.0 * self.sf[i,j,k,l] - 4.0 * self.sf[i,j-1,k,l] + self.sf[i,j-2,k,l]
        else: raise IndexError("Index '%s' is out of range: [%d,%d]" % ('j',0, last_index_theta))

        return result


    def _gradient_theta(self, i, j, k, l):
        """Calculate theta component of the gradient of state function ('self.sf')
        at coordinate corresponding to given indices ('i','j','k', 'l')

        ## NOTE ##
        # Index correspondence:
        - 'i': rho index
        - 'j': theta index
        - 'k': phi index
        - 'l': time index
        """

        result = complex(1.0)
        if i != 0:
            result /= self.grid.rho[i]
            result *= self._partial_diff_theta(i,j,k,l)
        else:
            result /= (2.0 * self.grid.rho.delta)
            result *= - 3.0 * self._partial_diff_theta(i, j, k, l) \
                + 4.0 * self._partial_diff_theta(i+1, j, k, l) \
                - self._partial_diff_theta(i+2, j, k, l)

        return result


    def _partial_diff_phi(self, i, j, k, l):
        """Calculate partial differentiated state function with repsect to phi

        This method takes the boundary effect into account
        such as applying periodic boundary condition
        """
        last_index_phi = self.grid.phi.N - 1
        result = complex(1.0 / (2.0 * self.grid.phi.delta))
        if k not in [0, last_index_phi]:
            result *= self.sf[i,j,k+1,l] - self.sf[i,j,k-1,l]
        elif k == 0:
            result *= self.sf[i,j,k+1,l] - self.sf[i,j,last_index_phi,l]
        elif k == last_index_phi:
            result *= self.sf[i,j,0,l] - self.sf[i,j,k-1,l]
        else: raise IndexError("Index '%s' is out of range: [%d,%d]" % ('k',0, last_index_phi))

        return result


    def _partial_diff_phi_over_sin_theta(self, i, j, k, l):
        """Calculate partial differentiated state function with repsect to phi, divided by sin theta

        This method handles singularities at theta = 0 or \pi
        by using L'Hopital's rule at the singlar points.

        ## Assumptions:
        - The first and last theta grid value are 0 and \pi respectively.
        """

        last_index_theta = self.grid.theta.N - 1
        result = complex(1.0)
        if j not in [0, last_index_theta]:
            result /= np.sin(self.grid.theta[j])
            result *= self._partial_diff_phi(i,j,k,l)
        elif j == 0:
            result /= 2.0 * self.grid.theta.delta  # cos(0) = 1.0 is omitted
            result *= - 3.0 * self._partial_diff_phi(i,j,k,l) \
                + 4.0 * self._partial_diff_phi(i,j+1,k,l) \
                - self._partial_diff_phi(i,j+2,k,l)
        elif j == last_index_theta:
            result *= - 1.0  # = cos(pi)
            result /= 2.0 * self.grid.theta.delta
            result *= 3.0 * self._partial_diff_phi(i,j,k,l) \
                - 4.0 * self._partial_diff_phi(i,j-1,k,l) \
                + self._partial_diff_phi(i,j-2,k,l)
        else: raise IndexError("Index '%s' is out of range: [%d,%d]" % ('j',0, last_index_theta))

        return result


    def _gradient_phi(self, i, j, k, l):
        """Calculate phi component of the gradient of state function ('self.sf')
        at coordinate corresponding to given indices ('i','j','k', 'l')

        ## NOTE ##
        # Index correspondence:
        - 'i': rho index
        - 'j': theta index
        - 'k': phi index
        - 'l': time index
        """

        #last_index_phi = self.grid.phi.N - 1
        result = complex(1.0)
        #if i != 0:  # this may becoma a problem when rho[0] != 0, which is the case in Qprop etc.
        if self.grid.rho[i] != 0.0:
            result /= self.grid.rho[i]
            result *= self._partial_diff_phi_over_sin_theta(i,j,k,l)
        else:
            assert i == 0  # if not, then, grid.rho[i] should be smaller than 0, which shouldn't be the case.
            result /= 2.0 * self.grid.rho.delta
            result *= - 3.0 * self._partial_diff_phi_over_sin_theta(i,j,k,l) \
                + 4.0 * self._partial_diff_phi_over_sin_theta(i+1,j,k,l) \
                - self._partial_diff_phi_over_sin_theta(i+2,j,k,l)

        return result


    def __getitem__(self, index_exp):

        index_exp = process_index_exp(index_exp, self.ndim)
        slices = [np.arange(self.sf.shape[idx])[index_exp[idx]] for idx in range(self.ndim)]
        indices = np.meshgrid(*slices, indexing='ij')

        grad_rho = self._gradient_rho_vec(*indices)
        grad_theta = self._gradient_theta_vec(*indices)
        grad_phi = self._gradient_phi_vec(*indices)

        result = squeeze_as_possible([grad_rho, grad_theta, grad_phi], complex)

        return tuple(result)


    def get_value(self, indices):
        """Return all components of gradient vector of given state function.
        
        This method is dedicated for single set of indices,
        not the array of set of indices.
        """
        assert len(indices) == self.ndim
        return self._gradient_rho(*indices), self._gradient_theta(*indices), self._gradient_phi(*indices)
        


class Numerical_State_Function(State_Function):
    pass


from qprop.default import default_config
class Numerical_State_Function_In_Spherical_Box_Qprop(Numerical_State_Function, State_Function_In_Spherical_Box):
    def __init__(self, q, delta_theta, delta_phi,
                 time_file_name = '', wf_file_name = '',
                 size_of_complex_double = 16):
        
        assert isinstance(q, Qprop20)
        self.q = q
        
        for arg in [delta_phi, delta_theta]: assert arg > 0
        for arg in [time_file_name, wf_file_name]: assert type(arg) is str
        assert int(size_of_complex_double) == size_of_complex_double
        self.size_of_complex_double = int(size_of_complex_double)
        
        ## Construct absolute paths for required files
        if time_file_name == '': time_file_name = default_config['time_file_name']
        if wf_file_name == '': wf_file_name = default_config['timed_wf_file_name']
        calc_home = path.abspath(q.home)
        self.wf_file_path = path.join(calc_home, wf_file_name)
        assert path.isfile(self.wf_file_path)
        self.time_file_path = path.join(calc_home, time_file_name)
        assert path.isfile(self.time_file_path)
        
        grid_spherical = self.get_spherical_grid_from_qprop_object(self.q, delta_theta, delta_phi, self.time_file_path)
        
        ## This initializer do some initialization such as assigning 'grid_spherical' to member variable etc.
        State_Function_In_Spherical_Box.__init__(self, grid_spherical)
        #State_Function_In_Spherical_Box.__init__(self, grid_spherical)
        
        # Check whether the size of state function file is consistent with the given Qprop object.
        self.check_sf_binary_file_size()
        
        
        self.get_value_from_indices_vec = np.vectorize(self.get_value_from_indices)
        
    
    def _evaluate_at_origin(self):
        
        pass

    
    @staticmethod
    def get_spherical_grid_from_qprop_object(q, delta_theta, delta_phi, time_file_path):
        """Construct 'Grid_Spherical' object from given arguments including qprop object"""
        assert isinstance(q, Qprop20)

        time_array = np.loadtxt(time_file_path)
        assert isinstance(time_array, np.ndarray)
        assert time_array.ndim == 1

        ## Discard the last timepoint to ensure the uniformness
        time_grid = time_array[:-1]

        ## Ensuring the uniformness of the temporal grid
        ## Also, assure the time_grid is in acsending order
        time_grid_intervals = np.diff(time_grid)
        if (time_grid_intervals.std() < 1e-12) and np.all(time_grid_intervals > 0):
            # If the uniformness is ensured, the grid interval can be safely obtained.
            delta_t = time_grid[1] - time_grid[0]
        else: raise Exception("The time grid should have uniform intervals in ascending order")
        
        ## Define necessary range(s) of each space-time coordinate
        # Define temporal range
        t_min = time_grid[0]
        t_max = time_grid[-1]
        # Define radial range
        rho_min = 0.0
        rho_max = (q.grid.numOfRadialGrid+1) * q.grid.delta_r

#        grid = Grid_Spherical(q.grid.delta_r, delta_theta, delta_phi, delta_t, 
#                              rho_max=rho_max+1e-10, rho_min=q.grid.delta_r, t_max=t_max+1e-10)

        ## Generate Grid object
        # [NOTE] A small amount (1e-10) is added to 'rho_max' and 't_max'
        # .. to ensure that last element isn't cut off by when constructing
        # .. radial grid and temporal grid respectively.
        # .. It is because if the length of given range (e.g. 'rho_max - rho_min')
        # .. is not an integer multiple of given coordinate grid interval (e.g. 'delta_rho'),
        # .. the remainder is cutted to make the length of range to be an integer multiple
        # .. of the given grid interval. The side being cut off is determined by 
        # .. 'cut_min' and 'cut_max' argument of the grid object's __init__() method.
        grid = Grid_Spherical(q.grid.delta_r, delta_theta, delta_phi, delta_t, 
                              rho_max=rho_max+1e-10, rho_min=rho_min, t_max=t_max+1e-10,
                              fix_delta_rho=True, fix_delta_theta=False, fix_delta_phi=False, fix_delta_t=True)
        
        return grid
        
        
    def check_sf_binary_file_size(self):
        """Return True if the size of state function file is consistent with the given Qprop object."""
        time_array = np.genfromtxt(self.time_file_path)
        assert isinstance(time_array, np.ndarray)
        assert time_array.ndim == 1
        
        size_per_element = self.size_of_complex_double

        num_of_elements_per_time = self.q.grid.numOfRadialGrid * self.q.grid.sizeOf_ell_m_unified_grid()
        num_of_elements = num_of_elements_per_time * time_array.size
        expected_size_of_wf_file = num_of_elements * size_per_element

        size_of_wf_file = path.getsize(self.wf_file_path)

        if size_of_wf_file != expected_size_of_wf_file:
            raise Exception("The size of size_of_wf_file (=%d) and expected_size_of_wf_file (=%d) are inconsistent"
                           % (size_of_wf_file, expected_size_of_wf_file))
    

    def read_sf_partial_array_for_ell_m(self, l,i):
        """

        ## Parameters:
        # l, integer, an index of time grid
        # i, integer, an index of radial grid
        """
        ## Check index range
        assert (l < self.grid.t.N)
        assert (i < self.q.grid.numOfRadialGrid)
        
        N_ell_m = self.q.grid.sizeOf_ell_m_unified_grid()
        sf_array_for_ell_m = np.empty((N_ell_m,), dtype=np.complex)

        size_per_element = self.size_of_complex_double
        num_of_elements_per_time = self.q.grid.numOfRadialGrid * self.q.grid.sizeOf_ell_m_unified_grid()

        with open(self.wf_file_path, 'rb') as f:
            for u in range(N_ell_m):
                offset_num_of_element = l * num_of_elements_per_time + u * self.q.grid.numOfRadialGrid + i
                offset = offset_num_of_element * size_per_element
                f.seek(offset,0)
                bytes_data = f.read(size_per_element)
#                 if bytes_data:
                sf_array_for_ell_m[u] = complex(*unpack('dd',bytes_data))
#                 else:
#                     print("Indices (rho, ell-m, t) == (%d, %d, %d)" % (i,u,l))
#                     print("Accessed offset == %d" % (offset))
#                     raise IOError("No data could be read from the wavefunction file")

        return sf_array_for_ell_m
    

    def get_sph_harm_array(self, theta_val, phi_val):
        ## Construct an array for containing result
        N_ell_m = self.q.grid.sizeOf_ell_m_unified_grid()
        sph_harm_array = np.empty((N_ell_m,), dtype=np.complex)
                
        for idx, ell_m_tuple in enumerate(self.q.grid.get_l_m_iterator()):
            ell, m = ell_m_tuple
            sph_harm_array[idx] = sph_harm(m,ell,phi_val,theta_val)

        return sph_harm_array
    
    def get_value_from_indices(self, i, j, k, l):
        #assert isinstance(grid, Grid_Spherical)
        sf_partial_array = self.read_sf_partial_array_for_ell_m(l, i)
        sph_harm_array = self.get_sph_harm_array(self.grid.theta[j], self.grid.phi[k])
        sf_value = 1.0 / self.grid.rho[i] * (sf_partial_array * sph_harm_array).sum()
        return sf_value

    def get_value(self, indices):
        return self.get_value_from_indices(*indices)
    
    def __getitem__(self, index_exp):
        ## [180322 NOTE] For a moment, only Grid_Spherical has 'get_partial_indices()' method.
        partial_indices = self.grid.get_partial_indices(index_exp)
        #print(partial_indices)
        return self.get_value_from_indices_vec(*partial_indices)



from mpi4py import MPI
class Numerical_State_Function_In_Spherical_Box_Qprop_MPI(Numerical_State_Function_In_Spherical_Box_Qprop):
    def __init__(self, *args, mpi_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(mpi_file, MPI.File)
        self.mpi_file = mpi_file

    def read_sf_partial_array_for_ell_m(self, l,i):
        """
        ## Parameters:
        # l, integer, an index of time grid
        # i, integer, an index of radial grid
        """
        ## Check index range
        assert (l < self.grid.t.N)
        assert (i < self.q.grid.numOfRadialGrid)
        
        N_ell_m = self.q.grid.sizeOf_ell_m_unified_grid()
        sf_array_for_ell_m = np.empty((N_ell_m,), dtype=np.complex)

        size_per_element = self.size_of_complex_double
        num_of_elements_per_time = self.q.grid.numOfRadialGrid * self.q.grid.sizeOf_ell_m_unified_grid()

        offset_num_of_element = l * num_of_elements_per_time + i * self.q.grid.numOfRadialGrid
        offset = offset_num_of_element * size_per_element

        buf = np.empty((self.q.grid.sizeOf_ell_m_unified_grid(),), dtype=complex)
        self.mpi_file.Read_at(offset, buf)
        return buf

##### State function for numerical propagation ####
### it should support some method such as __getitem__ etc.
### to be used in Bohmian calculation etc.
#
#def is_writable_file(file):
#    is_writable = False
#    if hasattr(file, 'writable'):
#        if file.writable(): is_writable = True
#    return is_writable
#
#def is_readable_file(file):
#    is_readable = False
#    if hasattr(file, 'readable'):
#        if file.readable(): is_readable = True
#    return is_readable
#
#def is_binary_file(file):
#    is_binary = False
#    if hasattr(file, 'mode'):
#        file_mode_lowercase = file.mode.lower()
#        is_binary = 'b' in file_mode_lowercase
#    return is_binary
#
#def is_writable_binary_file(file):
#    #file_mode_lowercase = file.mode.lower()
#    #is_binary = 'b' in file_mode_lowercase
#    is_writable_and_binary = is_binary_file(file) and is_writable_file(file)
#    return is_writable_and_binary
#
#
#text_fontdict_template = {'backgroundcolor':(0,0,0,0.4), 
#                 'fontsize':'x-large', 'family':'monospace', 'color':'white', 'weight':'bold'}
#
#si_time_unit_to_factor = {
#    's':1e1, 'ms':1e3, 'us':1e6, 'ns':1e9, 'ps':1e12, 'fs':1e15, 'as':1e18
#}
#

