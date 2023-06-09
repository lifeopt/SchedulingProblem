from tqdm import tqdm
import torch
from configuration.config import config
# from utils.visualize_job_schedule import display_gantt_chart
from utils import visualize_job_schedule
from utils import utility
import numpy as np

n_envs = config['envs']['num_envs']
n_updates = config['envs']['num_updates']
n_steps_per_update = config['envs']['num_steps_per_update']
gamma = config['agent']['gamma']
lam = config['agent']['lam']
ent_coef = config['agent']['entropy_coef']

global_n_completes_eps = 0

episode_lengths = np.zeros(n_envs)

def train(agent, envs_wrapper, device, writer, due_dates,instance_id):
    critic_losses = []
    actor_losses = []
    entropies = []
    epi_count = 0
    avg_tardiness = 0.0
    
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
            
            avg_tardiness, epi_count = utility.calculate_average_tardienss(avg_tardiness, epi_count, infos['tardiness'])
            
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
        critic_loss_value = critic_loss.detach().cpu().numpy()
        actor_loss_value = actor_loss.detach().cpu().numpy()
        entropy_value = entropy.detach().mean().cpu().numpy()

        critic_losses.append(critic_loss_value)
        actor_losses.append(actor_loss_value)
        entropies.append(entropy_value)

        writer.add_scalar(f'Instance_{instance_id}/Critic_loss', critic_loss_value, sample_phase)
        writer.add_scalar(f'Instance_{instance_id}/Actor_loss', actor_loss_value, sample_phase)
        writer.add_scalar(f'Instance_{instance_id}/Entropy', entropy_value, sample_phase)
    
    writer.close()
        
        
        
    return (critic_losses, actor_losses, entropies)
