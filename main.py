import os
import numpy as np
import data.data_load as data_load
import gymnasium as gym
import env.register
import env.gssp_env


input_file_path = r'C:\Users\JS\Desktop\코드\01_ORScheduler\MYJSSP\data\taillard\open_shop_scheduling\tai4_4.txt'
   
def main():  
    
    instances = data_load.read_input_data(input_file_path)

    for i, instance in enumerate(instances):
        # num_jobs, num_machines, processing_times, machines
        (num_jobs, num_machines, processing_times, machines, operations_data, due_date) = instance
        env = gym.make('GSSP-v0', num_jobs = num_jobs, num_machines =  num_machines,
                       operations_data = operations_data, due_date = due_date)
        obs, info = env.reset(operations_data = operations_data)
        print("hi")

    env.close()

if __name__ == '__main__':
    main()