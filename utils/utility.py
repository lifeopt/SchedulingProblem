import numpy as np
import numpy as np
 
from configuration.config import config
n_envs = config['envs']['num_envs']
# n_envs = 1

def calculate_average_tardienss(avg_tardiness, total_terminated_epi, new_tardiness):
    
    for i in range(n_envs):
        if new_tardiness[i] > 0:
            total_terminated_epi += 1
            avg_tardiness = (avg_tardiness * total_terminated_epi + new_tardiness) / total_terminated_epi
    return avg_tardiness, total_terminated_epi

def normalize_operations_data(operations_data, max_processing_time, max_job_idx, max_machine_idx):
    normalized_data = []
    for jobidx, processing_time, machine in operations_data:
        # 각 속성을 최대값으로 나누어 정규화합니다
        normalized_jobidx = jobidx / max_job_idx
        normalized_processing_times = processing_time / max_processing_time
        normalized_machine_index = machine / max_machine_idx
        # 정규화된 값을 튜플로 묶어서 normalized_data 리스트에 추가합니다
        normalized_data.append((normalized_jobidx, normalized_processing_times, normalized_machine_index))
    return normalized_data

def flatten_observations(states):
    flattened_states = np.concatenate((states['job_schedule_matrix'].reshape(n_envs,-1),
                             states['operation_allocation_status'], 
                             states['operation_job_idxs'],
                             states['operation_processing_times']),
                             axis=1)
    return flattened_states

def additional_tardiness(job_schedule_matrix, machine_idx, start_time_idx, due_dates, processing_time):
    end_time_idx = start_time_idx + processing_time - 1
    completion_time = 0
    for time_idx in range(len(job_schedule_matrix[machine_idx])):
        if job_schedule_matrix[machine_idx][time_idx] != -1:
            completion_time = max(completion_time, time_idx)
    original_tardiness = max(0, completion_time - due_dates[machine_idx])
    new_tardiness = max(0, end_time_idx - due_dates[machine_idx])
    addional_tardiness = max(0, new_tardiness - original_tardiness)
    return addional_tardiness

def calculate_tardiness(job_schedule_matrix, due_dates):
    n_machines, max_T = job_schedule_matrix.shape
    sum_tardiness = 0
    for machine_idx in range(n_machines):
        completion_time = 0
        for time_idx in range(len(job_schedule_matrix[machine_idx])):
            if job_schedule_matrix[machine_idx][time_idx] != -1:
                completion_time = max(completion_time, time_idx)
        sum_tardiness += max(0, completion_time - due_dates[machine_idx])
    return sum_tardiness




if __name__ == '__main__':
    
    job_schedule_matrix = np.array([
        [-1, -1,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3, -1, -1, -1],
        [-1,  0,  0,  0, -1,  1,  1, -1, -1, -1, -1,  4, -1, -1,  2,  2,  2,  2, -1, -1, -1, -1, -1, -1, -1],
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    ])

    operation_processing_times = [3, 5, 5, 5, 5, 5, 7, 3, 3, 3, 3]
    operation_allocation_status = [True, True, True, True, True, True, True, True, False, True, False]
    operation_job_idxs = [0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 4]
    due_dates = [10, 10, 10]
    calculate_tardiness(job_schedule_matrix, due_dates)
                
