import numpy as np
import torch 
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1, gym_vec_env_v0, concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1
# import envpool
import gymnasium as gym
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

def make_env(env_id, frame_size, stack_size, shimmy=False, capture_video=False, run_name=""):
    # if capture_video:
    #     env = MeltingPotCompatibilityV0(env, render_mode="rgb_array")
    #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # else:
    if shimmy:
        env = load_meltingpot(env_id)
        env = MeltingPotCompatibilityV0(env, render_mode="None")
        env = color_reduction_v0(env, 'full') # grayscale
        env = resize_v1(env, frame_size, frame_size) # resize and stack images
        env = frame_stack_v1(env, stack_size)
        # env = gym_vec_env_v0(env,2)
        # env = pettingzoo_env_to_vec_env_v1(env)

    # print(envpool.list_all_envs())
    # exit()
    else:

    #     envs = envpool.make(
    #         "MiniGrid-Empty-5x5-v0",
    #         env_type="gymnasium",
    #         num_envs=2,
    #     )
                
        env = gym.make(env_id)#, render_mode="human")
    #     # env = color_reduction_v0(env, 'full') # grayscale
    #     # env = resize_v1(env, frame_size, frame_size) # resize and stack images
    #     # env = frame_stack_v1(env, stack_size)
    #     # print(env.observation_space)
    #     env = gym_vec_env_v0(env,2)

    # env = concat_vec_envs_v1(
    #   env,
    #   num_vec_envs=3,
    #   num_cpus=0,
    #   base_class="gymnasium")
    return env