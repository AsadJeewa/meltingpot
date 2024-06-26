import numpy as np
import torch 
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot

def batchify_obs(obs, device):#stack agents
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)#store only values and ignore keys
    # transpose to be (batch, channel, height, width) from  (batch, height, width, channel) where channel can be stacks 
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)
    return obs

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)
    return x

def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(env_id, frame_size, stack_size):
    env = load_meltingpot(env_id)
    env = MeltingPotCompatibilityV0(env, render_mode="None")
    env = color_reduction_v0(env, 'full') # grayscale
    env = resize_v1(env, frame_size, frame_size) # resize and stack images
    env = frame_stack_v1(env, stack_size)
    return env