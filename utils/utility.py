import numpy as np
import numpy as np

from configuration.config import config
n_envs = config['envs']['num_envs']

def flatten_observations(states):
    flattened_states = np.concatenate((states['job_schedule_matrix'].reshape(n_envs,-1),
                             states['operation_allocation_status'], 
                             states['operation_job_idxs'],
                             states['operation_processing_times']),
                             axis=1)
    return flattened_states


def calculate_tardiness(job_schedule_matrix, due_dates, processing_times):
    num_machines = len(job_schedule_matrix)
    max_T = len(job_schedule_matrix[0])
    
    completion_times = [0] * num_machines
    tardiness = [0] * num_machines

    for machine_idx in range(num_machines):
        current_time = 0

        for time_idx in range(max_T):
            job_idx = job_schedule_matrix[machine_idx][time_idx]

            if job_idx != -1:  # If a job is scheduled at this time index
                processing_time = processing_times[job_idx]
                completion_time = current_time + processing_time
                completion_times[machine_idx] = completion_time

                tardiness[machine_idx] += max(0, completion_time - due_dates[job_idx])

                current_time = completion_time

    return tardiness

