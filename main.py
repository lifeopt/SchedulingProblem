import os
import numpy as np
import data.data_load as data_load
import gymnasium as gym
import env.register
import env.gssp_env
import torch
from networks.A2C import A2C
from configuration.config import config
from tqdm import tqdm
import utility

# input_file_path = r'C:\Users\JS\Desktop\코드\01_ORScheduler\MYJSSP\data\taillard\open_shop_scheduling\tai4_4.txt'

import numpy as np

# environment hyperparams
n_envs = 3
n_updates = 1000
n_steps_per_update = 128
randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
actor_lr = 0.001
critic_lr = 0.005

def custom_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task(data)
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'custom':  # Handle custom messages
            result = env._handle_custom(data)
            remote.send(result)
        else:
            raise NotImplementedError
        

   
def main():  
    
    instances = data_load.read_input_data(input_file_path)

    for _, instance in enumerate(instances):
        # num_jobs, num_machines, processing_times, machines
        (num_jobs, num_machines, processing_times, machines, operations_data,
         due_date, num_features, converted_processing_times, max_T, num_actions) = instance
        
        # Replace the default worker function with your custom worker function
        gym.vector.async_vector_env._worker = custom_worker
        
        
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make('GSSP-v0', num_jobs = num_jobs, num_machines =  num_machines,
                            operations_data = operations_data, due_date = due_date, max_T = max_T, max_episode_steps=600)
                for i in range(n_envs)
            ]
        )
        
        # set the device
        use_cuda = config['use_cuda']
        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        
       
        # init the agent
        agent = A2C(num_features, num_actions, device, config['critic_lr'], config['actor_lr'], n_envs)
        
        
        # create a wrapper environment to save episode returns and episode lengths
        envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

        critic_losses = []
        actor_losses = []
        entropies = []
        
        # use tqdm to get a progress bar for training
        for sample_phase in tqdm(range(n_updates)):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
            ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
            ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
            masks = torch.zeros(n_steps_per_update, n_envs, device=device)

            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = envs_wrapper.reset(seed=42)
                states = utility.preprocess_observation(states)

            # play n steps in our parallel environments to collect data
            for step in range(n_steps_per_update):
                # select an action A_{t} using S_{t} as input for the agent
                actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                    states
                )

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                states, rewards, terminated, truncated, infos = envs_wrapper.step(
                    actions.cpu().numpy()
                )

                ep_value_preds[step] = torch.squeeze(state_value_preds)
                ep_rewards[step] = torch.tensor(rewards, device=device)
                ep_action_log_probs[step] = action_log_probs

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor([not term for term in terminated])

                # calculate the losses for actor and critic
                critic_loss, actor_loss = agent.get_losses(
                    ep_rewards,
                    ep_action_log_probs,
                    ep_value_preds,
                    entropy,
                    masks,
                    gamma,
                    lam,
                    ent_coef,
                    device,
                )

                # update the actor and critic networks
                agent.update_parameters(critic_loss, actor_loss)

                # log the losses and entropy
                critic_losses.append(critic_loss.detach().cpu().numpy())
                actor_losses.append(actor_loss.detach().cpu().numpy())
                entropies.append(entropy.detach().mean().cpu().numpy())
                
                print("hi")

if __name__ == '__main__':  
    main()
