"""Several types of State Function objects are defined."""

from os import path
from struct import unpack

import numpy as np
from scipy.special import sph_harm

from tdse.grid import Grid, Grid_Polar, Grid_Spherical
from tdse.grid import process_index_exp, squeeze_as_possible

from qprop import Qprop20


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


class Numerical_State_Function_In_Spherical_Box_Qprop(Numerical_State_Function, State_Function_In_Spherical_Box):
    def __init__(self, q, delta_theta, delta_phi,
                 time_file_name = 'hydrogen_re-time.dat', wf_file_name = 'hydrogen_re-wf.bin',
                 size_of_complex_double = 16):
        
        assert isinstance(q, Qprop20)
        self.q = q
        
        for arg in [delta_phi, delta_theta]: assert arg > 0
        for arg in [time_file_name, wf_file_name]: assert type(arg) is str
        assert int(size_of_complex_double) == size_of_complex_double
        self.size_of_complex_double = int(size_of_complex_double)
        
        ## Construct absolute paths for required files
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

        time_array = np.genfromtxt(time_file_path)
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




#### State function for numerical propagation ####
## it should support some method such as __getitem__ etc.
## to be used in Bohmian calculation etc.

def is_writable_file(file):
    is_writable = False
    if hasattr(file, 'writable'):
        if file.writable(): is_writable = True
    return is_writable

def is_readable_file(file):
    is_readable = False
    if hasattr(file, 'readable'):
        if file.readable(): is_readable = True
    return is_readable

def is_binary_file(file):
    is_binary = False
    if hasattr(file, 'mode'):
        file_mode_lowercase = file.mode.lower()
        is_binary = 'b' in file_mode_lowercase
    return is_binary

def is_writable_binary_file(file):
    #file_mode_lowercase = file.mode.lower()
    #is_binary = 'b' in file_mode_lowercase
    is_writable_and_binary = is_binary_file(file) and is_writable_file(file)
    return is_writable_and_binary


text_fontdict_template = {'backgroundcolor':(0,0,0,0.4), 
                 'fontsize':'x-large', 'family':'monospace', 'color':'white', 'weight':'bold'}

si_time_unit_to_factor = {
    's':1e1, 'ms':1e3, 'us':1e6, 'ns':1e9, 'ps':1e12, 'fs':1e15, 'as':1e18
}


#import unit
#from vis.ntype import check_type_ndarray
#from vis.plot import get_ylim
#from vis.layout import get_text_position_and_inner_alignment
#from matplotlib import rcParams
#
#from grid import Grid_Cartesian_1D
#
#import matplotlib.pyplot as plt
#
#
#rcParams['mathtext.fontset'] = 'stix'
#
#class State_function_in_Box_1D(object):
#    """
#    Handle state function of a system with particle(s)
#    
#    [NOTE] wavefunction is also known as wave function
#    """
#    def __init__(self, grid, psi_x_t0, t_0, normalize=True, save_copy=True):
#        """
#        'normalize': boolean
#            True : default, normalize the input initial state function 'psi_x_t0' 
#                when it is assigned to the member variable of self.
#            False : don't normalize 'psi_x_t0' 
#                when it is assigned to the member variable of self.
#        """
#        ## Check input arguments and assign them into member varables if needed
#        # For flags
#        assert type(normalize) is bool
#        assert type(save_copy) is bool
#        #
#        # For 'grid'
#        assert type(grid) is Grid_Cartesian_1D
#        self.grid = grid
#        #
#        # For 'psi_x_t0'
#        self._check_valid_state_function_array(psi_x_t0)
#        if save_copy:
#            self.psi_x_t = psi_x_t0.astype(np.complex, copy=True)
#        else:
#            self.psi_x_t = psi_x_t0.astype(np.complex, copy=False)
#        #
#        # For initial time
#        try: t_0 = float(t_0)
#        except: raise TypeError("'t_0' should be of type 'float'")
#        self.time = t_0
#        
#        ## Normalize the state function if specified to do so.
#        if normalize:
#            self.normalize()
#    
#    def _check_valid_state_function_array(self, sf_array):
#        """Check whether an input array for a state function is valid.
#        
#        Check the type, shape, dimension of the input array 'sf_array'
#        to be compliant with the 'grid' of self.
#        
#        [NOTE] 'sf_array' abbreviates 'state function array'
#        """
#        assert type(sf_array) is np.ndarray
#        assert sf_array.ndim == 1
#        assert sf_array.shape[0] == self.grid.x.N
#        assert check_type_ndarray(sf_array, 'float') or check_type_ndarray(sf_array, 'complex')
#        
#            
#    def get_norm(self):
#        """Calculate and return norm of the current state function"""
#        norm = ((self.psi_x_t * self.psi_x_t.conj()).sum() * self.grid.x.delta).real
#        return norm
#    
#    
#    def normalize(self):
#        """
#        Normalize the current state function
#        
#        [NOTE] It returns nothing.
#        """
#        norm = self.get_norm()
#        normalization_constant = 1.0 / np.sqrt(norm)
#        self.psi_x_t *= normalization_constant
#        
#        
#    def get_squared(self):
#        """Return norm square of the state function in position representation."""
#        return (self.psi_x_t * self.psi_x_t.conj()).real
#    
#    def _calculate_squared(self, arr):
#        self._check_valid_state_function_array(arr)
#        return (arr * arr.conj()).real
#    
#    
#    def save(self, file):
#        """
#        Save state function to disk.
#        
#        [To Do]
#        - Support saving into a binary file.
#        """
#        ## Check prerequisite
#        #assert is_writable_binary_file(file)
#        assert is_writable_file(file)
#        is_binary = is_binary_file(file)
#        
#        if is_binary: self.psi_x_t.tofile(file)
#        else:
#            for idx0 in range(self.grid.x.N):
#                c = self.psi_x_t[idx0]
#                st = '%.18e %.18e\n' % (c.real, c.imag)
#                file.write(st)
#    
#    def load(self, file, index, save_to_self=False):
#        assert is_readable_file(file)
#        is_binary = is_binary_file(file)
#        
#        num_of_element_to_read = self.grid.x.N
#        num_of_element_to_ignore = index * num_of_element_to_read
#        
#        loaded_state_function = np.empty((self.grid.x.N,), dtype=np.complex)
#        if is_binary:
#            file.seek(num_of_element_to_ignore * np.dtype(np.complex).itemsize, 0)
#            loaded_state_function[:] = np.fromfile(file, dtype=np.complex, count=num_of_element_to_read)[:]
#        else:
#            real_imag_2D = np.empty((num_of_element_to_read,2), dtype=float)
#            for idx in range(num_of_element_to_ignore):
#                file.readline()
#            for idx in range(num_of_element_to_read):
#                line = file.readline()
#                st_list = line.split(' ')
#                real_imag_2D[idx,:] = [float(st_list[0]), float(st_list[1])]
#            loaded_state_function[:] = np.apply_along_axis(lambda x: complex(*x), axis=1, arr=real_imag_2D)[:]
#        
#        if save_to_self:
#            self.psi_x_t[:] = loaded_state_function[:]
#        else:
#            return loaded_state_function
#    
#    def plot(self, which_part='real', fig=None, ax=None, show_time=True, time_unit='fs', 
#             other_state=None, other_time=None):
#        """Plot current state function.
#        
#        ## Argument(s):
#        # 'show_time'
#        : Determine whether the time of the state function to be shown on the plot
#        - if True:
#            the time is shown on the plot
#        - if False:
#            the time is not shown on the plot
#        - [NOTE] This parameter is turned off when 'other_state' is specified
#        .. and the corresponding 'other_time' parameter isn't specified
#        .. It is because the time of the 'other_state' is unknown 
#        .. and would generally be different from internal time 'self.time'.
#        
#        # 'other_time'
#        - [NOTE] If 'other_time' is not given, this argument will be ignored.
#        """
#        
#        ## Check and pre-process input arguments
#        # Process 'which_part'
#        assert type(which_part) is str
#        assert which_part.lower() in ['real','re','imag','imaginary','im','square','sq']
#        which_part = which_part.lower()
#        #
#        # Process 'ax' and 'fig'
#        if ax is None:
#            assert fig is None
#            fig = plt.figure()
#            ax = fig.gca()
#        #
#        # Process 'other_time'
#        if other_time is not None:
#            assert is_real_number(other_time)
#            if other_state is None:
#                other_time = None
#                print("'other_time' will be ignored since there's no given 'other_state'")
#        #
#        # Process 'show_time'
#        # - This parameter is turned off when 'other_state' is specified
#        # .. and the corresponding 'other_time' parameter isn't specified
#        assert type(show_time) is bool
#        if (other_state is not None) and (other_time is None):
#            show_time = False
#        #
#        # Process 'time_unit'
#        assert type(time_unit) is str
#        assert time_unit in si_time_unit_to_factor.keys()
#        #
#        # Process 'other_state'
#        if other_state is not None:
#            self._check_valid_state_function_array(other_state)
#            if check_type_ndarray(other_state, 'float'):
#                other_state = other_state.astype(np.complex)
#        
#        ## Set appropriate fontsize system (e.g. 'medium', 'large', 'x-large' etc.)
#        fig_size_in_inch = fig.get_size_inches()
#        fig_size_len_geom_mean = (fig_size_in_inch[0] * fig_size_in_inch[1]) ** 0.5
#        rcParams['font.size'] = fig_size_len_geom_mean * 2.0
#        
#        ## Set an array to be plotted following the 'which_part' argument
#        ylabel=''
#        y_unit = 'a.u.'
#        
#        sf_array = None
#        if other_state is not None:
#            sf_array = other_state
#        else: 
#            # Set current state function of self as default
#            sf_array = self.psi_x_t
#        
#        if which_part in ['real','re']:
#            to_be_plotted = sf_array.real
#            ylabel = r'$Re\,\left[\psi\,\left(x,t\right)\right]\,\,/\,\,%s$' % y_unit
#        elif which_part in ['imag','imaginary','im']:
#            to_be_plotted = sf_array.imag
#            ylabel = r'$Im\,\left[\psi\,\left(x,t\right)\right]\,\,/\,\,%s$' % y_unit
#        elif which_part in ['square','sq']:
#            #to_be_plotted = self.get_squared()
#            to_be_plotted = self._calculate_squared(sf_array)
#            ylabel = r'$\left|\psi\,\left(x,t\right)\right|^2\,\,/\,\,%s$' % y_unit
#        else: raise Exception("Unsupported plotting data type: %s" % which_part)
#
#        # Determine ylim
#        try: 
#            ylim_abs_max = max(map(abs,get_ylim(to_be_plotted)))
#            ylim = [-ylim_abs_max, ylim_abs_max]
#        except:
#            ylim = [None,None]
#        #else: raise Exception("Unexpected error")
#                
#        if which_part in ['square','sq']:
#            ylim[0] = 0
#        
#        ax_color = (0.4,0,0,1)
#        ax_plot_kwargs = {'color' : ax_color}
#        ax.tick_params(axis='y', labelsize='x-large', colors=ax_color)
#        
#        x_n = self.grid.x.array
#        ax.plot(x_n, to_be_plotted, **ax_plot_kwargs)
#        
#        ax.set_ylim(*ylim)
#        
#        x_unit = 'a.u.'
#        ax.set_ylabel(ylabel, fontsize='xx-large', color=ax_color)
#        ax.set_xlabel(r'$x\,\,/\,%s$' % (x_unit), fontsize='xx-large')
#        
#        
#        ax.tick_params(axis='x', labelsize='x-large')
#        
#        
#        ## Add text for representing time, if told to do so.
#        if show_time:
#            # Determine time to show
#            time_to_show = None
#            if other_time is None:
#                time_to_show = self.time
#            else: time_to_show = other_time
#            
#            # Configure text appearence
#            text_fontdict = text_fontdict_template
#            
#            text_xy, text_align_dict = get_text_position_and_inner_alignment(ax, 'nw',scale=0.05)
#            
#            pos_x, pos_y = text_xy
#            text_fontdict = {**text_fontdict, **text_align_dict}
#            #text_fontdict['va'] = 'top'
#            #text_fontdict['ha'] = 'left'
#            #pos_x, pos_y = get_text_position(fig, ax, ha=text_fontdict['ha'], va=text_fontdict['va'])
#
#            # Construct time text string
#            time_unit = time_unit.lower()
#            unit_factor = si_time_unit_to_factor[time_unit]
#            text_content = r'time = %.3f %s' % (time_to_show * unit.au2si['time'] * unit_factor, time_unit)
#
#            # Add text to the axis
#            text = ax.text(pos_x, pos_y, text_content, fontdict=text_fontdict, transform=ax.transAxes)
#        
#        # Return plot related objects for futher modulation
#        return fig, ax
#    
#    
#    def plot_real(self, **kwargs):
#        """Plot real part of the current state function."""
#        return self.plot(which_part='real', **kwargs)
#
#



