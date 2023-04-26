import numpy as np
import gymnasium as gym
from gymnasium import spaces
import utils.utility as utility
from configuration.config import config

n_envs = config['envs']['num_envs']

class GSSP(gym.Env):

    def __init__(self, num_jobs, num_machines, operations_data, due_dates, max_T):
        
        self.N = num_jobs
        self.M = num_machines
        self.K = len(operations_data)
        self.T = max_T
        self._due_dates = due_dates
        self.operations_data = operations_data
        (max_job_idx, max_processing_time, max_machine_idx) = max(operations_data)
        self._norm_operations_data = utility.normalize_operations_data(
            operations_data=operations_data,
            max_processing_time = max_processing_time,
            max_job_idx = max_job_idx,
            max_machine_idx = max_machine_idx
        )
        
        self._operation_processing_times = [data[1] for data in self.operations_data]
        self._operation_job_idxs = [data[0] for data in  self.operations_data]
        
        self._norm_operation_processing_times = [data[1] for data in self._norm_operations_data]
        self._norm_operation_job_idxs = [data[0] for data in  self._norm_operations_data]

        
        self._cumulative_reward = 0
        self._cumulative_tardiness = 0
        self._additional_tardiness = 0

        # Observations are dictionaries with the timetable (num_machine * max_t) and the job waiting list.
        self.observation_space = spaces.Dict(
            {
                # we are creating an observation space with self.num_machines*self.max_t variables,
                # each taking on self.num_jobs different values (job idxs).
                # "job_schedule_matrix": spaces.Box(low=-1, high=self.N, shape=(self.M, self.T,), dtype=int),
                "job_schedule_matrix": spaces.Box(low=-1, high=1, shape=(self.M, self.T,), dtype=float),
                "operation_allocation_status": spaces.MultiBinary(self.K),
                # "operation_job_idxs": spaces.Box(low=0, high=max_job_idx, shape=(self.K,), dtype=int),
                "operation_job_idxs": spaces.Box(low=0, high=1, shape=(self.K,), dtype=float),
                # "operation_processing_times": spaces.Box(low=0, high=max_processing_times, shape=(self.K,), dtype=int)
                "operation_processing_times": spaces.Box(low=0, high=1, shape=(self.K,), dtype=float)
            }
        )

        # We have (operation idx, machine idx) pair which is N * M combinations
        # a (operation idx, machine idx) pair means that the job corresponding to 
        # operation idx is assigned to the machine corresponding to machine idx.
        self.action_space = spaces.Discrete(self.K * self.M)
        
    def _get_obs(self):
        return {
            "job_schedule_matrix": self._norm_job_schedule_matrix.ravel(), # 2D (num_machines * num_time_indexs) array containing job index
            "operation_allocation_status": self._operation_allocation_status, # 1D(num_operations) array having allocation status
            "operation_job_idxs": self._norm_operation_job_idxs,
            "operation_processing_times": self._norm_operation_processing_times,
        }
    
    def _get_info(self):
        return {
            "addtional tardiness": self._additional_tardiness,
            "op_schedule_matrix": self._op_schedule_matrix
        }

    def reset(self, seed=None, options=None):
        # Initialize the value of unallocated cell as -1 (num machine * num time index with all -1 values)
        self._job_schedule_matrix = np.full((self.M, self.T), -1, dtype=int)            
        self._op_schedule_matrix = np.full((self.M, self.T), -1, dtype=int)            
        self._norm_job_schedule_matrix = np.full((self.M, self.T), -1, dtype=int)            
        
        # Reset all operations in waiting list to an unassigned status (=false)
        self._operation_allocation_status = np.full(self.K, False, dtype=bool)
        self._cumulative_reward = 0
        self._cumulative_tardiness = 0
        self._additional_tardiness = 0
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # An action is an operation-machine (k,m) pair that determines
        # which machine to assign the operation to. 
        
        (operation_idx, machine_idx) = self._get_action_to_assignment(action)
        
        job_idx = self._operation_job_idxs[operation_idx]
        norm_job_idx = self._norm_operation_job_idxs[operation_idx]
        processing_time = self._operation_processing_times[operation_idx]
        
        # action
        # The assignment is made at the earliest possible time index that is available for allocation
        additional_tardiness = 0.0
        reward = 0.0
        if not self._operation_allocation_status[operation_idx]:
            reward += 0.1 # reward shaping
            target_t = self.find_smallest_available_t(machine_idx, job_idx, processing_time)
            if target_t != -1:
                # additional_tardiness = utility.additional_tardiness(self._job_schedule_matrix, machine_idx, target_t, self._due_dates, processing_time)
                self._job_schedule_matrix[machine_idx, target_t:target_t + processing_time] = job_idx
                self._operation_allocation_status[operation_idx] = True
                self._op_schedule_matrix[machine_idx, target_t:target_t + processing_time] = operation_idx
                self._norm_job_schedule_matrix[machine_idx, target_t:target_t + processing_time] = norm_job_idx
                # reward += (additional_tardiness / 1000.0)
            else:
                pass # nothing change
        else:
            reward -= 0.1 # reward shaping
            
            
        # An episode is done iff the all operations are allocated
        terminated = np.all(self._operation_allocation_status)
        if terminated:
            reward -= utility.calculate_tardiness(self._job_schedule_matrix, self._due_dates) / 1000
            self._cumulative_tardiness += utility.calculate_tardiness(self._job_schedule_matrix, self._due_dates) / 1000
        #     # additional_reward = 3
        #     # if self._cumulative_reward < 0:
        #     #     reward += (-self._cumulative_reward)
        #     # reward += additional_reward
        #     pass
        # else:
        #     reward -= (additional_tardiness / 1000.0)
    
        # self._additional_tardiness = additional_tardiness
        observation = self._get_obs()
        info = self._get_info()
        self._cumulative_reward += reward
        return observation, reward, terminated, False, info
    
    def _get_action_to_assignment(self, action):
        k = action % self.K
        m = action // self.K
        
        return (k, m)

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
