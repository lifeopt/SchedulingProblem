import data.data_load as data_load
import gymnasium as gym
import env.register
import env.gssp_env
import torch
from networks.A2C import A2C
from configuration.config import config
from train import save_load_weights
from train import train
import sys
from pathlib import Path
from utils import plot_results
import os
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent / "configuration"))

input_file_path = config['paths']['input_file_path']
n_envs = config['envs']['num_envs']
n_updates = config['envs']['num_updates']
use_cuda = config['envs']['use_cuda']
randomize_domain = config['envs']['randomize_domain']
actor_lr = config['agent']['actor_lr']
critic_lr = config['agent']['critic_lr']
is_save_weights = config['model_utils']['save_weights']
is_load_weights = config['model_utils']['load_weights']
num_instance_limit = config['train']['num_instance_limit']

def main():  
    
    instances = data_load.read_input_data(input_file_path)
    writer = SummaryWriter(log_dir='runs/GSSP_training')
    for i, instance in enumerate(instances):
        if num_instance_limit >= i:
            continue
        (num_jobs, num_machines, processing_times, machines, operations_data,
         due_date, num_features, converted_processing_times, max_T, num_actions) = instance
        
        if randomize_domain:
            envs = gym.vector.AsyncVectorEnv(
                [
                    lambda: gym.make('GSSP-v0', num_jobs = num_jobs, num_machines =  num_machines,
                        operations_data = operations_data, due_date = due_date, max_T = max_T, max_episode_steps=600)
                    for i in range(n_envs)
                ]
            )
        else:
            envs = gym.vector.make('GSSP-v0', num_jobs = num_jobs, num_machines =  num_machines,
                operations_data = operations_data, due_date = due_date, max_T = max_T, num_envs = n_envs, max_episode_steps=600)
            
        
        # set the device
        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
       
        # init the agent
        agent = A2C(num_features, num_actions, device, critic_lr, actor_lr, n_envs)
        
        # create a wrapper environment to save episode returns and episode lengths
        envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

        # training the agent
        (critic_losses, actor_losses, entropies) = train.train(agent=agent, envs_wrapper=envs_wrapper,
                                                               device=device, writer=writer, instance_id=i)
            
        if is_save_weights:
            save_load_weights.save_weights(agent)

        if is_load_weights:
            agent = A2C(num_features, num_actions, device, critic_lr, actor_lr)
            save_load_weights.load_weights(agent)
        plot_results.plotting(agent, envs_wrapper, critic_losses, actor_losses, entropies)
        

if __name__ == '__main__':  
    main()
