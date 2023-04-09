import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configuration.config as config
import data.data_load as data_load
import utility

class GSSP(gym.Env):

    def __init__(self, num_jobs, num_machines, operations_data, due_date):
        
        self.N = num_jobs
        self.M = num_machines
        self.K = len(operations_data)
        self.T = 1000 # 임시
        self._due_date = due_date
        
        # Observations are dictionaries with the timetable (num_machine * max_t) and the job waiting list.
        self.observation_space = spaces.Dict(
            {
                # we are creating an observation space with self.num_machines*self.max_t variables,
                # each taking on self.num_jobs different values (job idxs).
                "job_schedule_matrix": spaces.MultiDiscrete([self.N+1]*(self.M)*(self.T)),
                "operation_allocation_status": spaces.MultiBinary(self.K),
                "operation_job_idxs": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(self.K,), dtype=int),
                "operation_processing_times": spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(self.K,), dtype=int)
            }
        )

        # We have (operation idx, machine idx) pair which is N * M combinations
        # a (operation idx, machine idx) pair means that the job corresponding to 
        # operation idx is assigned to the machine corresponding to machine idx.
        self.action_space = spaces.Tuple([
                    spaces.Discrete(self.K),
                    spaces.Discrete(self.M),
                ])

    def _get_obs(self):
        return {
            "job_schedule_matrix": self._job_schedule_matrix, # 2D (num_machines * num_time_indexs) array containing job index
            "operation_allocation_status": self._operation_allocation_status, # 1D(num_operations) array having allocation status
            "operation_job_idxs": self._operation_job_idxs,
            "operation_processing_times": self._operation_processing_times,
        }
    
    def _get_info(self):
        return {
            "tardiness": 0
        }

    def reset(self, operations_data=None, seed=None, options=None):
        # Initialize the value of unallocated cell as -1 (num machine * num time index with all -1 values)
        self._job_schedule_matrix = np.full((self.M, self.T), -1, dtype=int)            
        # Reset all operations in waiting list to an unassigned status (=false)
        self._operation_allocation_status = np.full(self.K, False, dtype=bool)
        self._operation_processing_times = np.full(self.K, -1, dtype=int)
        self._operation_job_idxs = np.full(self.K, -1, dtype=int)
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # An action is an operation-machine (k,m) pair that determines
        # which machine to assign the operation to. 
        
        operation_idx = action[0]
        machine_idx = action[1]
        job_idx = self._operation_job_idxs[operation_idx]
        processing_time = self._operation_processing_times[operation_idx]
        
        # action
        # The assignment is made at the earliest possible time index that is available for allocation
        target_t = self.find_smallest_available_t(machine_idx, job_idx, processing_time)
        if target_t != -1:
            self._job_schedule_matrix[machine_idx, target_t:target_t + processing_time] = job_idx
            self._operation_allocation_status[operation_idx] = True
        else:
            pass # nothing change
        
        # An episode is done iff the all operations are allocated
        terminated = np.all(self._operation_allocation_status)


        tardiness = utility.calculate_tardiness(self._job_schedule_matrix, self._due_date, self._operation_processing_times)
        reward = (-tardiness) if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    # Return the allocation_matrix initialized in flatten form as an easy-to-read (M by T) matrix form
    def flatten(self, allocation_matrix_flat):
        return allocation_matrix_flat.reshape(self.M, self.T)    
    
    def is_job_scheduled_at_same_time(self, machine_idx, job_idx, start_time, processing_time):
        for m in range(self.M):
            if m != machine_idx:
                for t in range(start_time, start_time + processing_time):
                    if self._job_schedule_matrix[m, t] == job_idx:
                        return True
        return False

    def find_smallest_available_t(self, machine_idx, job_idx, processing_time):
        for t in range(self.T - processing_time + 1):
            is_valid_slot = True
            for i in range(processing_time):
                if self._job_schedule_matrix[machine_idx, t + i] != -1:
                    is_valid_slot = False
                    break
                for m in range(self.M):
                    if m != machine_idx and self._job_schedule_matrix[m, t + i] == job_idx:
                        is_valid_slot = False
                        break
                if not is_valid_slot:
                    break
            if is_valid_slot:
                return t
        return -1  # No valid time slot found

# code to test the env works properly


input_file_path = r'C:\Users\JS\Desktop\코드\01_ORScheduler\MYJSSP\data\taillard\open_shop_scheduling\tai4_4.txt'
if __name__ == '__main__':
    
    instances = data_load.read_input_data(input_file_path)
    for idx, (num_jobs, num_machines, processing_times, machines, operations_data, due_date) in enumerate(instances):
        
        env = GSSP(num_jobs, num_machines, operations_data, due_date)
        
        # Test the reset method
        initial_observation = env.reset()
        # assert initial_observation['allocation_matrix'].shape == (num_machines * max_t,)
        # assert initial_observation['waiting_list'].shape == (num_operations,)
        
        # Test the step method with a sample action
        action = (1, 3)  # Allocate job 3 to machine 1
        observation, reward, terminated, _, info = env.step(action)
        
        # Check if the output matches the expected result
        # You need to define the expected result based on your problem requirements
        expected_observation = ...
        expected_reward = ...
        expected_done = ...
        expected_info = ...

        # assert np.array_equal(observation['allocation_matrix'], expected_observation['allocation_matrix'])
        # assert np.array_equal(observation['waiting_list'], expected_observation['waiting_list'])
        # assert reward == expected_reward
        # assert done == expected_done
        # assert info == expected_info