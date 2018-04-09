from sys import stdout
from itertools import product

import numpy as np
from scipy.interpolate import griddata

from ntype import is_integer_valued_real
from tdse.coordinate import transform_to_canonical_range_of_spherical_coordinate

from .state import State_Function, State_Function_In_Polar_Box, State_Function_In_Spherical_Box
from .state import Gradient_State_Function_In_Polar_Box, Gradient_State_Function_In_Spherical_Box


value_instead_zero = 1e-50


class Velocity_Field(object):
    def __init__(self, state_function):
        assert State_Function in type(state_function).mro()

        ## Set member variables
        self.sf = state_function
        self.grid = self.sf.grid
        self.ndim = self.grid.ndim


class Velocity_Field_Polar_Box(Velocity_Field):
    def __init__(self, state_function):
        ## Process input arguments
        assert State_Function_In_Polar_Box in type(state_function).mro()

        ## Set variables (such as 'sf', 'grid', 'ndim' etc.) into member variables with basic type checks
        Velocity_Field.__init__(self, state_function)

        self.grad = Gradient_State_Function_In_Polar_Box(self.sf)

        ## Caching related part
        # Initialize varaibles for caching.
        self.cached_indices = ()
        for i in range(self.grid.ndim): self.cached_indices += (None,)
        #
        # Initialize container varibles
        self.grid_values_2D = None
        self.grad_phi_values_1D = None
        self.grad_rho_values_1D = None
        self.psi_values_1D = None


    def is_out_of_range(self, i, j, k):
        """Check whether the input indices correponds to any unit cell in the given grid ('grid_polar')

        This function is for polar grid.
        It checks whether (i == grid.rho.N) or (k == grid.t.N - 1).
        This is a case for rho grid with boundary condition
        where the value at the end in positive direction of rho is always zero.

        ## Notes ##
        # Concept of unit cell of a grid
        For a given grid, it is possible to define unit cell that corresponds to each index set.
        e.g. Let 'rho', 'phi', 't' is an array of grid points in each dimension,
        Then, a unit cell, which is in 3D vector space, defined by 8 vertices:
        [(rho[i],phi[i],t[i]), (rho[i],phi[i],t[i+]), (rho[i],phi[i+1],t[i]), ..., (rho[i+1],phi[i+1],t[i+1])]
        exists for each index set: (i,j,k)

        # Handling edges
        For example, for a grid with index i that runs from 0 to N-1, for index i,
        the unit cell is ill-defined since grid point that corresponds to i+1 index doesn't exist.
        Therefore, indices that corresponds to edges of a grid should be handled as a special case,
        possibly by using if-else statements etc.
        """

        self.check_whether_index_is_in_supported_range(i,j,k)
        outside_of_range = (i == self.grid.rho.N) or (k == self.grid.t.N - 1)
        return outside_of_range


    def check_whether_index_is_in_supported_range(self, i, j, k):
        """Check whether an index is in the supported range"""

        grid_polar = self.grid

        for arg in [i,j,k]: assert is_integer_valued_real(arg)
        # Because of the boundary condition,
        # .. when i == grid_polar.rho.N, it means out of spaital region
        # .. and velocity vector is set to zero at this region.
        assert (0 < (i + 1)) and (i < (grid_polar.rho.N + 1))  # i = 0, 1, ..., grid_polar.rho.N

        # When j == grid_polar.phi.N - 1, then the unit_cell's vertices' 'j'
        # .. ranges from (grid_polar.phi.N - 1 to grid_polar.phi.N)
        # .. where 'grid_polar.phi.N' is turned into '0' accordingin to the periodic boundary condition.
        assert (0 < (j + 1)) and (j < grid_polar.phi.N)  # j = 0, 1, ..., grid_polar.phi.N - 1

        assert (0 < (k + 1)) and (k < grid_polar.t.N)


    def move_to_principle_region_for_polar_coord(self, rho, phi, small_amount = 1e-10):
        """

        ## Development Note (180317):
        - this method should be migrated to 'coordinate' module
        """

        grid_polar = self.grid

        if rho < grid_polar.rho.min:
            if rho < 0:
                rho = - rho
                phi += np.pi
            elif 0 <= rho:
                raise Exception("'rho' should be bigger than 'grid_polar.rho.min'")
            else: raise Exception("Unexpected case: neither 'rho < 0' nor '0 <= rho'")
        phi = phi % (2.0 * np.pi - small_amount)
        return rho, phi


    def interpolate(self, rho, phi, t):
        """
        Interpolate 'psi', 'gradient_rho' and 'gradient_phi'
        from given psi ('sf', which means state function) and gradient values ('grad')
        """

        # Aliasing for convenience
        grid = self.grid
        sf = self.sf
        grad = self.grad

        # e.g. (rho,phi) == (-0.1, 0.0) -> (0.1, \pi)
        # e.g. (rho,phi) == (0.25, 3.0 * \pi) -> (0.25, \pi)
        rho, phi = self.move_to_principle_region_for_polar_coord(rho, phi)

        # Calculate corresponding set of indices
        i, j, k = grid.get_index(rho, phi,t)


        # Check whether the current index set is identical to cached indices
        current_indices = (i,j,k)
        use_cached_data = False
        if self.cached_indices == current_indices: use_cached_data = True
        else:
            # Cache indices for performance
            self.cached_indices = current_indices  # it should be of type tuple.


        if not use_cached_data:
            self.check_whether_index_is_in_supported_range(i, j, k)

            # 'is_out_of_range' filters out the case for 'i == grid.rho.N' and 'k == grid.t.N - 1'
            # .. where the velocity vector is set to zero.
            assert not self.is_out_of_range(i,j,k)

            ## Initialize unit cells to be filled.
            ## .. These unit cells will be used as a bounding box
            ## ..within which a point is interpolated based on the values at unit cell's vertices
            unit_cell_shape = (2,2,2)
            psi_values = np.empty(unit_cell_shape, dtype=np.complex)
            grad_rho_values = np.empty(unit_cell_shape, dtype=np.complex)
            grad_phi_values = np.empty(unit_cell_shape, dtype=np.complex)
            grid_values = np.empty(unit_cell_shape + (3,), dtype=np.float)

            for idx0 in range(unit_cell_shape[0]):
                for idx1 in range(unit_cell_shape[1]):
                    for idx2 in range(unit_cell_shape[2]):
                        ## Determine index coordinate within unit cell array
                        sl = np.index_exp[idx0,idx1,idx2]
                        ## Determine index coordinate within grid array
                        # .. for phi index 'j', periodic condition is applied so that
                        # .. if (j+idx0) is equal to grid.phi.N, then it is converted to 0.
                        coord_indice = np.index_exp[idx0 + i, ((idx1 + j) % grid.phi.N), idx2 + k]

                        ## Assign each unit cell
                        grid_values[sl+np.index_exp[:]] = grid[coord_indice]
                        psi_values[sl] = sf[coord_indice]
                        grad_rho, grad_phi = grad[coord_indice]
                        grad_rho_values[sl] = grad_rho
                        grad_phi_values[sl] = grad_phi

                if i == (grid.rho.N - 1):
                    ## [180308 NOTE] it only holds for equidistanced rho grid
                    grid_values[1,:,:,0] = grid_values[0,:,:,0] + grid.rho.delta
                    grid_values[1,:,:,1:] = grid_values[0,:,:,1:]
                    psi_values[1,:,:] = 0

                    # Approximate grad_rho by 2-point finite, one-side finite difference formula
                    # .. by the boundrary conditon in which psi values are zero at grid.rho.max
                    # .. which corresponds to rho index of grid.rho.N,
                    # .. 'psi_values[1,:,:]' is omitted from the original formula:
                    # .. (psi_values[1,:,:] - psi_values[0,:,:]) / grid.rho.delta
                    grad_rho_values[1,:,:] = - 1.0 / grid.rho.delta * psi_values[0,:,:]

                    # By the boundrary conditon in which psi values are zero at grid.rho.max
                    # .. which corresponds to rho index of grid.rho.N,
                    # .. the gradient component along phi direction is also zero
                    grad_phi_values[1,:,:] = 0

                    # if i == (grid.rho.N - 1), then, don't loop for idx0 = 1 but just break
                    # .. since 'i + idx0' == '(grid.rho.N - 1) + idx0' == 'grid.rho.N'
                    # .. where 'grid.rho.N' is out of index of grid array so that
                    # .. the values at the corresponding positions should be set manually.
                    break

            ## If this part doesn't exist, interpolation may not work well.
            ## .. It is because of the discontinuity caused by the periodic condition
            ## .. of azimuthal ('phi') coordinate. As an (possibly unique) example,
            ## .. if 'j == (grid.phi.N - 1)', then 'j+1 == grid.phi.N', which corresponds to 'j+1 == 0'
            ## .. thus, phi value at j is around 2.0 * np.pi but the value at j+1 is 0.0
            ## .. raising a big discontinuity which causes the interpolation doesn't work.
            if j == (grid.phi.N - 1):
                grid_values[:,1,:,:] = grid_values[:,0,:,:]
                grid_values[:,1,:,1] += grid.phi.delta

            ## Reshape the grid values (coordinate values) to make a shape:
            ## .. (number of coordinate points, coordinate space dimension)
            ## .. where 'number of coordinate points' is 'equal to number_of_vertice_per_unit_cell'
            num_of_vertice_per_unit_cell = 1  # '1' is just for an initialization for following multiplications.
            for num_of_vertice_per_dimension in unit_cell_shape:
                num_of_vertice_per_unit_cell *= num_of_vertice_per_dimension
            unit_cell_dimension = len(unit_cell_shape)
            self.grid_values_2D = grid_values.flatten().reshape(num_of_vertice_per_unit_cell, unit_cell_dimension)

            self.grad_phi_values_1D = grad_phi_values.flatten()
            self.grad_rho_values_1D = grad_rho_values.flatten()
            self.psi_values_1D = psi_values.flatten()

        else:
            for var in [self.grid_values_2D, self.grad_phi_values_1D, self.grad_rho_values_1D, self.psi_values_1D]:
                assert isinstance(var, np.ndarray)

        # Consider using more grid points since sometimes the points go out of the unit cell
        # consider using neaest algorithm. it would be okay if the grid point is dense enough ..
        target_point = [rho, phi, t]
        interpol_algo = 'linear'
        grad_phi = np.squeeze(griddata(self.grid_values_2D, self.grad_phi_values_1D, [target_point], method=interpol_algo))
        grad_rho = np.squeeze(griddata(self.grid_values_2D, self.grad_rho_values_1D, [target_point], method=interpol_algo))
        psi = np.squeeze(griddata(self.grid_values_2D, self.psi_values_1D, [target_point], method=interpol_algo))

        for arg in [grad_rho,grad_rho,psi]:
            if np.isnan(arg):
                #for debugging
                print("grid_values")
                print(grid_values_2D)
                print("")
                print("rho,phi,t = ",(rho,phi,t))
                print("i,j,k = ",(i,j,k))
                raise ValueError("the interpolated number is nan!")

        return psi, grad_rho, grad_phi


    def get_velocity(self, t, y):
        """Calculate velocity vector of Bohmian particle"""

        rho, phi = y
        stdout.write("\rt = %.3e, y = [%.3e, %.3e]" % (t,rho,phi))

        # Aliasing of 'grid_polar' to 'grid' for convenience etc.
        grid = self.grid
        grad = self.grad

        i, j, k = grid.get_index(rho, phi,t)

        self.check_whether_index_is_in_supported_range(i,j,k)

        drho_dt, dphi_dt = None, None
        outside_of_range = self.is_out_of_range(i,j,k)
        if outside_of_range:
            # If the particle goes outside of the range .. set it zero.
            # [NOTE] It should be figured out why, if the particle really can go outside of the range
            # .. and try to reduce that case since it is one of inaccurate phenomena
            drho_dt, dphi_dt = 0, 0
        else:
            ## Interpolate
            psi, grad_rho, grad_phi = self.interpolate(rho, phi, t)

            ## To prevent singularity
            ## [NOTE] There should be more elegant and accurte way to do this prevention
            #value_instead_zero = 1e-50  # [180319, NOTE] this is defined as module variable
            if psi == 0.0:
                print("[WARNING] psi become zero. Continuing with replacing it to %.2e" % value_instead_zero)
                psi = value_instead_zero

            ## Calculate velocity vector components in atomic unit
            drho_dt = (1.0 / psi * grad_rho).imag
            dphi_dt = (1.0 / psi * grad_phi).imag

        return [drho_dt, dphi_dt]




class Velocity_Field_In_Spherical_Box(Velocity_Field):
    def __init__(self, state_function):
        assert State_Function_In_Spherical_Box in type(state_function).mro()

        ## Perform basic initializeation such as
        ## .. assigning 'sf', 'grid', 'ndim' etc. to member variables
        Velocity_Field.__init__(self, state_function)

        self.grad = Gradient_State_Function_In_Spherical_Box(self.sf)

        ## Caching related part
        # Initialize varaibles for caching.
        self.cached_indices = ()
        for i in range(self.grid.ndim): self.cached_indices += (None,)
        #
        # Initialize container varibles
        self.grid_values_2D = None
        self.grad_rho_values_1D = None
        self.grad_theta_values_1D = None
        self.grad_phi_values_1D = None
        self.psi_values_1D = None


    def check_whether_index_is_in_supported_range(self, i, j, k, l):
        in_supported_range = False
        in_supported_range |= self.grid.indices_are_in_supported_range(i,j,k,l)
        in_supported_range |= (i == self.grid.rho.N)
        assert in_supported_range

    def is_out_of_range(self, i, j, k, l):
        self.check_whether_index_is_in_supported_range(i,j,k,l)
        outside_of_range = (i == self.grid.rho.N) or (l == self.grid.t.N - 1)
        return outside_of_range

    def interpolate(self, rho, theta, phi, t):
        """Interpolate psi and gradient of psi values for given coordinates,
        specified by 'rho', 'theta', 'phi' and 't'.
        """

        # Calculate corresponding set of indices
        i, j, k, l = self.grid.get_index(rho, theta, phi,t)


        # Check whether the current index set is identical to cached indices
        current_indices = (i,j,k,l)
        use_cached_data = False
        if self.cached_indices == current_indices: use_cached_data = True
        else:
            # Cache indices for performance
            self.cached_indices = current_indices  # it should be of type tuple.


        if not use_cached_data:


            self.check_whether_index_is_in_supported_range(i, j, k, l)

            # 'is_out_of_range' filters out the case for 'i == grid.rho.N' and 'k == grid.t.N - 1'
            # .. where the velocity vector is set to zero.
            assert not self.is_out_of_range(i,j,k,l)

            ## [180317 NOTE] Consider implementing in dimension-generalized scheme
            ## Initialize unit cells to be filled.
            ## .. These unit cells will be used as a bounding box
            ## .. within which some quantities at a point in the box
            ## .. are interpolated based on the values at unit cell's vertices
            #
            # Construct a tuple for representing shape of unit_cell
            num_of_vertices_per_dimension = 2
            unit_cell_shape = ()
            for idx in range(self.grid.ndim):
                unit_cell_shape += (num_of_vertices_per_dimension,)
            #
            # Construct containers for each quantities at vertices of the unit_cell
            psi_values = np.empty(unit_cell_shape, dtype=np.complex)
            grad_rho_values = np.empty(unit_cell_shape, dtype=np.complex)
            grad_theta_values = np.empty(unit_cell_shape, dtype=np.complex)
            grad_phi_values = np.empty(unit_cell_shape, dtype=np.complex)
            grid_values = np.empty(unit_cell_shape + (self.grid.ndim,), dtype=np.float)


            ## [180317 NOTE] Consider implementing using a generator
            ## .. for looping on all sets of indices of each vertex

            # flattening may be the solution

            is_at_radial_boundary = i == (self.grid.rho.N - 1)
            cell_indices = None
            if is_at_radial_boundary:
                # 'partial_indices' runs for 'theta', 'phi', 'time' axis of 'unit_cell'
                partial_indices = product(range(num_of_vertices_per_dimension), repeat=self.grid.ndim-1)
                #cell_indices = ((0,i,j,k) for i,j,k in partial_indices)
                cell_indices = ((0,) + indices for indices in partial_indices)
            else: cell_indices = product(range(num_of_vertices_per_dimension), repeat=self.grid.ndim)

            for sl in cell_indices:
                ## (OBSOLATE) Determine index coordinate within unit cell array
                #sl = np.index_exp[idx0,idx1,idx2]

                ## Determine index coordinate within grid array
                # .. for phi index 'j', periodic condition is applied so that
                # .. if (j+idx0) is equal to grid.phi.N, then it is converted to 0.
                coord_indice = np.index_exp[sl[0] + i, sl[1] + j, ((sl[2] + k) % self.grid.phi.N), sl[3] + l]

                ## Assign each unit cell
                grid_values[sl+np.index_exp[:]] = self.grid.get_value(coord_indice)  # apply for single set of indices
                #grid_values[sl+np.index_exp[:]] = self.grid[coord_indice]
                psi_values[sl] = self.sf.get_value(coord_indice)  # apply for single set of indices
                #psi_values[sl] = self.sf[coord_indice]
                grad_rho_values[sl], grad_theta_values[sl], grad_phi_values[sl] = self.grad.get_value(coord_indice) # apply for single set of indices
                #grad_rho_values[sl], grad_theta_values[sl], grad_phi_values[sl] = self.grad[coord_indice]
                #grad_rho, grad_theta, grad_phi = self.grad[coord_indice]
                #grad_rho_values[sl] = grad_rho
                #grad_theta_values[sl] = grad_theta
                #grad_phi_values[sl] = grad_phi

            if is_at_radial_boundary:
                ## [180308 NOTE] it only holds for equidistanced rho grid where 'grid.rho.delta' is constant
                grid_values[1,:,:,:,0] = grid_values[0,:,:,:,0] + self.grid.rho.delta
                grid_values[1,:,:,:,1:] = grid_values[0,:,:,:,1:]
                psi_values[1,:,:,:] = 0

                # Approximate grad_rho by 2-point finite, one-side finite difference formula
                # .. by the boundrary conditon in which psi values are zero at grid.rho.max
                # .. which corresponds to rho index of grid.rho.N,
                # .. 'psi_values[1,:,:,:]' is omitted from the original formula:
                # .. (psi_values[1,:,:,:] - psi_values[0,:,:,:]) / grid.rho.delta
                grad_rho_values[1,:,:,:] = - 1.0 / self.grid.rho.delta * psi_values[0,:,:,:]

                # By the boundrary conditon in which psi values are zero at self.grid.rho.max
                # .. which corresponds to rho index of grid.rho.N,
                # .. the gradient component along theta and phi direction is also zero
                grad_theta_values[1,:,:,:] = 0
                grad_phi_values[1,:,:,:] = 0

            ## If this part doesn't exist, interpolation may not work well.
            ## .. It is because of the discontinuity caused by the periodic condition
            ## .. of azimuthal ('phi') coordinate. As an (possibly unique) example,
            ## .. if 'k == (self.grid.phi.N - 1)', then 'k+1 == self.grid.phi.N', which corresponds to 'k+1 == 0'
            ## .. thus, phi value at k is around 2.0 * np.pi but the value at k+1 is 0.0
            ## .. raising a big discontinuity which causes the interpolation doesn't work.
            ## [180327 NOTE] This holds only for equidistanced 'phi' array.
            if k == (self.grid.phi.N - 1):
                # 'phi_index_order' is an index corresponding to 'phi' is 'k' which is 3-rd 
                # .. (corresponding to 2 in terms of counting index)
                phi_index_order = 2
                # overwriting (as a re-initialization) for the next line.
                grid_values[:,:,1,:,:] = grid_values[:,:,0,:,:]  
                # add phi.delta to get next bigger phi value to construct unit cell.
                grid_values[:,:,1,:,phi_index_order] += self.grid.phi.delta  

            ## Reshape the grid values (coordinate values) to make a shape:
            ## .. (number of coordinate points, coordinate space dimension)
            ## .. where 'number of coordinate points' is 'equal to number_of_vertice_per_unit_cell'
            num_of_vertice_per_unit_cell = 1  # '1' is just for an initialization for following multiplications.
            for num_of_vertice_per_dimension in unit_cell_shape:
                num_of_vertice_per_unit_cell *= num_of_vertice_per_dimension
            unit_cell_dimension = self.grid.ndim  # Same as len(unit_cell_shape)
            self.grid_values_2D = grid_values.flatten().reshape(num_of_vertice_per_unit_cell, unit_cell_dimension)

            self.grad_rho_values_1D = grad_rho_values.flatten()
            self.grad_theta_values_1D = grad_theta_values.flatten()
            self.grad_phi_values_1D = grad_phi_values.flatten()
            self.psi_values_1D = psi_values.flatten()

        else:
            # If the cached data are going to be used, check if they are of proper type.
            for var in [self.grid_values_2D, self.grad_phi_values_1D, self.grad_rho_values_1D, self.grad_theta_values_1D, self.psi_values_1D]:
                assert isinstance(var, np.ndarray)

        # [180328 OBSOLATE] Consider using more grid points since sometimes the points go out of the unit cell
        # [180328 OBSOLATE] consider using neaest algorithm. it would be okay if the grid point is dense enough ..
        target_point = [rho, theta, phi, t]
        interpol_algo = 'linear'

        
        try:
            grad_rho = np.squeeze(griddata(self.grid_values_2D, self.grad_rho_values_1D,
                                           [target_point], method=interpol_algo))
            grad_theta = np.squeeze(griddata(self.grid_values_2D, self.grad_theta_values_1D,
                                             [target_point], method=interpol_algo))
            grad_phi = np.squeeze(griddata(self.grid_values_2D, self.grad_phi_values_1D,
                                           [target_point], method=interpol_algo))
            psi = np.squeeze(griddata(self.grid_values_2D, self.psi_values_1D,
                                      [target_point], method=interpol_algo))
        except:
            print("grid_values: ")
            print(self.grid_values_2D)
            print("")
            print(str(self.grid.names) + " = ",(rho, theta, phi, t))
            print("i,j,k,l = ",(i,j,k,l))
            raise ValueError("the interpolation falied.")

        for arg in [grad_rho, grad_theta, grad_rho, psi]:
            if np.isnan(arg):
                #for debugging
                print("grid_values: ")
                print(self.grid_values_2D)
                print("")
                print(str(self.grid.names) + " = ",(rho, theta, phi, t))
                print("i,j,k,l = ",(i,j,k,l))
                raise ValueError("the interpolated number is nan!")

        return psi, grad_rho, grad_theta, grad_phi


    def get_velocity(self, t, y, logging=False):
        """Calculate velocity vector of Bohmian particle

        ## NOTE: order of coordinates in 'y'
        For spherical coordinate system,
        the coordinate vector 'y' is ordered by the following:
        ('rho','theta','phi')
        where each component means radial, polar, azimulthal coordinate
        in spherical coordinate system.
        """

        assert len(y) == self.grid.ndim - 1  # '-1' excludes one dimension for time
        rho, theta, phi = transform_to_canonical_range_of_spherical_coordinate(*y)
        if logging: stdout.write("\rt = %.3e, y = [%.3e, %.3e, %.3e]" % (t,rho,theta,phi))

        # [180319 NOTE: needless] Aliasing of 'grid_polar' to 'grid' for convenience etc.
        #grid = self.grid
        #grad = self.grad

        i, j, k, l = self.grid.get_index(rho, theta, phi, t)

        ## [180317 NOTE] This part is somewhat redundant.
        ## .. Consider removal, while remaining the concreteness.
        self.check_whether_index_is_in_supported_range(i,j,k,l)

        drho_dt, dtheta_dt, dphi_dt = None, None, None
        outside_of_range = self.is_out_of_range(i,j,k,l)
        velocity_vector = None
        if outside_of_range:
            # If the particle goes outside of the range .. set it zero.
            # [NOTE] It should be figured out why, if the particle really can go outside of the range
            # .. and try to reduce that case since it is one of inaccurate phenomena
            velocity_vector = [0 for idx in range(self.ndim - 1)]  # excluding one dimension for time.
            #drho_dt, dtheta_dt, dphi_dt = 0, 0, 0
        else:
            ## Interpolate
            #psi, grad_rho, grad_theta, grad_phi = self.interpolate(rho, theta, phi, t)
            interpolated_values = self.interpolate(rho, theta, phi, t)
            psi = interpolated_values[0]
            grad_vector = interpolated_values[1:]

            ## To prevent singularity
            ## [NOTE] There should be more elegant and accurte way to do this prevention
            # 'value_instead_zero' is declared as a global variable
            if psi == 0.0:
                print("[WARNING] psi become zero. Continuing by replacing it with %.2e" % value_instead_zero)
                psi = value_instead_zero

            ## Calculate velocity vector components in atomic unit
            velocity_vector = [(1.0 / psi * grad_component).imag for grad_component in grad_vector]
            #drho_dt = (1.0 / psi * grad_rho).imag
            #dtheta_dt = (1.0 / psi * grad_theta).imag
            #dphi_dt = (1.0 / psi * grad_phi).imag
            #drho_dt, dtheta_dt, dphi_dt = velocity_vector
    
        ## The 'velocity_vector' should be of type tuple.
        return velocity_vector 
        #return [drho_dt, dtheta_dt, dphi_dt]

