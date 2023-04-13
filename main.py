import os
import numpy as np
import data.data_load as data_load
import gymnasium as gym
import env.register
import env.gssp_env
import torch
from networks.A2C import A2C
from configuration.config import config

# input_file_path = r'C:\Users\JS\Desktop\코드\01_ORScheduler\MYJSSP\data\taillard\open_shop_scheduling\tai4_4.txt'

import numpy as np


# environment hyperparams
n_envs = 10
n_updates = 1000
n_steps_per_update = 128
randomize_domain = False
   
def main():  
    
    instances = data_load.read_input_data(input_file_path)

    for i, instance in enumerate(instances):
        # num_jobs, num_machines, processing_times, machines
        (num_jobs, num_machines, processing_times, machines, operations_data, due_date) = instance
        
        envs = gym.vector.make('GSSP-v0', num_jobs = num_jobs, num_machines =  num_machines,
                        operations_data = operations_data, due_date = due_date, num_envs=config['n_envs'], max_episode_steps=600)
        
        # set the device
        use_cuda = config['use_cuda']
        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
           
        # init the agent
        # agent = A2C(obs_shape, action_shape, device, config['critic_lr'], config['critic_lr'], n_envs)
        
        # env = gym.make('GSSP-v0', num_jobs = num_jobs, num_machines =  num_machines,
        #                operations_data = operations_data, due_date = due_date)
        # obs, info = env.reset(operations_data = operations_data)
        print("hi")

if __name__ == '__main__':  
    main()
