input_file_path = r'C:\Users\JS\Desktop\코드\01_ORScheduler\MYJSSP\data\taillard\open_shop_scheduling\tai4_4.txt'
filepath = r'C:\Users\JS\Desktop\코드\01_ORScheduler\MYJSSP\data\taillard\open_shop_scheduling\tai4_4.txt'


def make_operation_data(num_jobs, num_machines, processing_times, machines):
    # this function maps the operations index to (job idx, processing time, machine)
    operation_data = {}
    for i in range(num_jobs):
        for j in range(num_machines):
            operation_idx = i * num_machines + j
            operation_data[operation_idx] = (i, processing_times[i][j], machines[i][j])

    return operation_data

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
            processing_times.append(list(map(int, data_lines[1 + i].split())))
            machines.append(list(map(int, data_lines[1 + num_jobs + i].split())))
        
        operations_data = make_operation_data(num_jobs, num_machines, processing_times, machines)
        instances.append((num_jobs, num_machines, processing_times, machines, operations_data))

    return instances



if __name__ == '__main__':
    instances = read_input_data(input_file_path)
    for idx, (num_jobs, num_machines, processing_times, machines, operations_data) in enumerate(instances):
        print(f"Instance {idx + 1}:")
        print(f"Number of jobs: {num_jobs}")
        print(f"Number of machines: {num_machines}")
        print("Processing times:")
        for row in processing_times:
            print(row)
        print("Machine order:")
        for row in machines:
            print(row)
        print("\n")