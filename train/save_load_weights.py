import os
import torch
import torch
from configuration.config import config


actor_weights_path = config['model_utils']['actor_weights_path']
critic_weights_path =  config['model_utils']['critic_weights_path']

def save_weights(agent):
    if not os.path.exists("weights"):
        os.mkdir("weights")

    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)

def load_weights(agent):
    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.actor.eval()
    agent.critic.eval()
