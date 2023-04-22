import os
import torch
import torch
from configuration.config import config



def save_weights(agent, critic_weights_path, actor_weights_path):
    if not os.path.exists("weights"):
        os.mkdir("weights")

    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)

def load_weights(agent, critic_weights_path, actor_weights_path):
    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.actor.eval()
    agent.critic.eval()
