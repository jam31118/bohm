from os.path import abspath, isdir, join

import numpy as np

from ntype import is_real_number, is_integer_valued_real

from .traj import Trajectory_Spherical_Box



class Ensemble_In_Spherical_Box(object):
    def __init__(self, velocity_func, t_min, t_max, num_of_traj=None, initial_values=None, **integrator_kwargs):
        
        self.spatial_dimension = 3
        # Let's generate this kind of format automatically from grid etc. for the sake of generalization
        self.traj_data_file_name_format = 'rho-%.3f-theta-%.3f-phi-%.3f.dat'
        
        assert callable(velocity_func)
        self.velocity_func = velocity_func
        
        for arg in [t_min, t_max]: assert is_real_number(arg)
        assert t_min < t_max
        self.t_min, self.t_max = t_min, t_max
        
        self.num_of_traj = None
        if num_of_traj is not None:
            assert is_integer_valued_real(num_of_traj)
            self.num_of_traj = int(num_of_traj)
        
        self.initial_values = None
        self.initial_values_are_set = False
        if initial_values is not None: self.set_initial_values(initial_values)
        
        self.integrator_kwargs = integrator_kwargs
        
        self.shape = None
        self.traj_array = None
        if self.initial_values_are_set: self.construct_array_of_trajectory()
        
    def construct_array_of_trajectory(self):
        assert self.initial_values_are_set
        assert (self.num_of_traj is not None) and (self.initial_values is not None)
        assert self.num_of_traj == self.initial_values.shape[0]
        
        traj_array = np.empty((self.num_of_traj,), dtype=Trajectory_Spherical_Box)
        traj_init_kwargs = {'dydt':self.velocity_func, 't0':self.t_min, 't_max':self.t_max, 
                               'initial_value': None, **self.integrator_kwargs}
        for traj_index, initial_value in enumerate(self.initial_values):
            traj_init_kwargs['initial_value'] = initial_value
            traj_array[traj_index] = Trajectory_Spherical_Box(**traj_init_kwargs)
        
        self.traj_array = traj_array
        self.shape = self.traj_array.shape
    
    def set_initial_values(self, initial_values):
        initial_values = np.array(initial_values)
        assert initial_values.ndim == 2
        assert initial_values.shape[1] == self.spatial_dimension
        self.initial_values = initial_values
        self.initial_values_are_set = True
        inferred_num_of_traj = self.initial_values.shape[0]
        if self.num_of_traj is None: # <=> if num_of_traj had already been set by input argument
            self.num_of_traj = inferred_num_of_traj
        else: assert self.num_of_traj == inferred_num_of_traj
    
    def propagate(self, savedir=None):
        save_traj = False
        file_name_format = None
        if savedir is not None:
            assert type(savedir) is str and isdir(savedir)
            savedir = abspath(savedir)
            save_traj = True
            file_name_format = join(savedir,self.traj_data_file_name_format)
        
        for traj in self.traj_array:
            try:
                traj.propagate()
                if traj.successful(): print(" Propagation Succeeded.")
            except Exception as e:
                print("")
                print("Propagation Failed for a trajectory with initial value: %s" % str(traj.initial_value))
                print("Error Contents: ",e, end='\n\n')
                traj.array = np.array(traj.sol)
            
            if save_traj:
                assert type(file_name_format) is str
                save_file_name = file_name_format % (tuple(traj.initial_value))
                if traj.array.ndim == 2: # check if traj.array is not empty
                    np.savetxt(save_file_name, traj.array)
        
    def __getitem__(self, index_exp):
        return self.traj_array[index_exp]

