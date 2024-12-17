import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from meltingpot.shim_cleanrl.utils import batchify_obs, batchify, unbatchify

class PettingZooToMoGymWrapper(gym.Env):
    def __init__(self, pettingzoo_env, env_id):
        super().__init__()
        self.pz_env = pettingzoo_env

        # Focus on the first agent only
        self.agent = self.pz_env.possible_agents[0]
        num_objectives = 2
        obs_space = self.pz_env.observation_space(self.agent)

        # Set the observation and action spaces for the selected agent
        self.observation_space = spaces.Box(low=obs_space.low[0][0][0], high=obs_space.high[0][0][0], shape=np.array([obs_space.shape[2], obs_space.shape[0],obs_space.shape[1]]), dtype=np.float32)
        self.action_space = self.pz_env.action_space(self.agent)
        self.reward_space = spaces.Box(low=0.0, high=1.0, shape=(num_objectives,), dtype=np.float32)#TODO get from env
        self.spec = EnvSpec(id=env_id) 
    
    def reset(self, seed=None, options=None):
        # Reset the environment and get initial observations
        obs = self.pz_env.reset(seed=seed, options=options)
        return obs[0][self.agent], {}  # Return the observation for the first agent and info

    def step(self, action):
        # Prepare the action dictionary for the environment
        actions = {self.agent: action}

        # Step the environment with the given action
        obs, scalar_rewards, terminations, truncations, infos = self.pz_env.step(actions)
        # Extract the values for the first agent
        observation = obs[self.agent]
        vector_reward = infos[self.agent]
        termination = terminations[self.agent]
        truncation = truncations[self.agent]
        # done = terminations[self.agent] or truncations[self.agent]
        info = infos.get(self.agent, {})
        return observation, vector_reward, termination, truncation, info#TO DO FIX VEC REWARDS
