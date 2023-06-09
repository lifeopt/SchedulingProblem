# input_file_path = r'C:/Users/JS/Desktop/코드/01_ORScheduler/MYJSSP/data/taillard/open_shop_scheduling/tai4_4.txt'


from math import ceil
from configuration.config import config
import numpy as np

input_file_path = config['paths']['input_file_path']

time_unit = 1   # 1분 단위
max_hours_per_schedule = 24

def convert_processing_times(processing_times, time_unit):
    converted_times = []

    for row in processing_times:
        converted_times.append([ceil(element / time_unit) for element in row])
        
    return converted_times

def pre_processing(num_jobs, num_machines, num_operations, processing_times, machines):
    converted_processing_times = convert_processing_times(processing_times, time_unit)
    
    operations_data = make_operation_data(num_jobs, num_machines, converted_processing_times, machines)
    due_date = make_due_data(converted_processing_times)
    num_operations = len([elem for row in processing_times for elem in row])
    
    # max_T = ceil(max_hours_per_schedule * 60 / time_unit) # max time index for job schedule matrix
    max_T = sum([sum(row) for row in converted_processing_times])
    job_schedule_matrix_dim = num_machines * max_T
    operation_allocation_status_dim = len(operations_data)
    operation_job_idxs_dim =  len(operations_data)
    operation_processing_times_dim =  len(operations_data)
    num_features = (
        job_schedule_matrix_dim
        + operation_allocation_status_dim
        + operation_job_idxs_dim
        + operation_processing_times_dim
    )
    
    num_actions = num_machines * num_operations
    return operations_data, due_date, num_features, converted_processing_times, max_T, num_actions

def make_operation_data(num_jobs, num_machines, processing_times, machines):
    # this function maps the operations index to (job idx, processing time, machine)
    operation_data = []
    for i in range(num_jobs):
        for j in range(num_machines):
            operation_idx = i * num_machines + j
            operation_data.append((i, processing_times[i][j], machines[i][j]))
            # operation_data[operation_idx] = (i, processing_times[i][j], machines[i][j])

    return operation_data

def make_due_data(processing_times):
    due_dates = [sum(row) for row in processing_times]
    return due_dates

def read_input_data(input_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    instances = []
    start_indexes = [i for i, line in enumerate(lines) if line.startswith('number')]
    start_indexes.append(len(lines))  # add end index for last instance
    
    for i in range(len(start_indexes) - 1):
        start_idx = start_indexes[i]
        end_idx = start_indexes[i+1]
        
        data_lines = [line for line in lines[start_idx:end_idx] # skip the lines starts with these words
                      if not line.startswith('number') 
                      and not line.startswith('processing') 
                      and not line.startswith('machines')]
        
        num_jobs, num_machines, *_ = map(int, data_lines[0].split())

        
        processing_times = []
        machines = []
        for i in range(num_jobs):
            processing_time = data_lines[1 + i].split()
            processing_times.append(list(map(int, processing_time)))
            machines.append(list(map(int, data_lines[1 + num_jobs + i].split())))
        
        
        num_operations = len([elem for row in processing_times for elem in row])
        operations_data, due_date, num_features, converted_processing_times, max_T, num_actions \
        = pre_processing(num_jobs, num_machines, num_operations, processing_times, machines)
        instances.append((num_jobs,
                          num_machines,
                          processing_times,
                          machines,
                          operations_data,
                          due_date,
                          num_features,
                          converted_processing_times,
                          max_T,
                          num_actions))
    return instances

if __name__ == '__main__':
    instances = read_input_data(input_file_path)
    for idx, (num_jobs, num_machines, processing_times, machines,
              operations_data, due_date, num_features, converted_processing_times, max, num_actions_T) in enumerate(instances):
        print(f"Instance {idx + 1}:")
        print(f"Number of jobs: {num_jobs}")
        print(f"Number of machines: {num_machines}")
        print("Processing times:")
        for row in processing_times:
            print(row)
        print("Machine order:")
        for row in machines:
            print(row)
        print("/n")    
