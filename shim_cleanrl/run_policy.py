import torch
import numpy as np
from mappo_shared import MAPPO
from ppo import PPO
from utils import batchify_obs
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

#DEBUG START
ma = False
load_model = True
pz  = False
frame_size = (64, 64)
stack_size = 4
#DEBUG END

if pz:
    from pettingzoo.butterfly import pistonball_v6
    env = pistonball_v6.parallel_env(render_mode="None")
else:
    from shimmy import MeltingPotCompatibilityV0
    from shimmy.utils.meltingpot import load_meltingpot
    env = load_meltingpot("clean_up_simple")#prisoners_dilemma_in_the_matrix__arena
    env = MeltingPotCompatibilityV0(env, render_mode="human")#human
    env = color_reduction_v0(env, 'full') # grayscale
    env = resize_v1(env, frame_size[0], frame_size[1]) # resize 
    env = frame_stack_v1(env, stack_size=stack_size) # stack

observations = env.reset()[0]#wrapped in tuple
steps = 0
r = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if load_model:
        # exp_name = "gpu_ctde_state_4x4__clean_up_simple__1__1695335842"
        # exp_name = "gpu_ctde_state_4x4_divergent__clean_up_simple__1__1695350009"

        exp_name = "fixed_ppo_single_hard__clean_up_simple__86427174__1700575418"
        # exp_name = "save_ppo_single_easy__clean_up_simple__1933059180__1700078728"
        # exp_name = "save_full_ppo_single_hard__clean_up_simple__121193687__1700102084"
        # exp_name = "noturn_gpu_ctde_state_multicoord__clean_up_simple__1__1697719103"

        modeldir = "model/TO PRESENT/"
        if ma:
            ppo = MAPPO(env.action_space(env.possible_agents[0]).n)
        else:
            ppo = PPO(env.action_space(env.possible_agents[0]).n)#MAPPO 
        print("****",env.action_space(env.possible_agents[0]).n)
        # exit()
        print("INIT")
        for param in ppo.actor.parameters():
            print(param.data)
            print(param.data.shape)
        print("LOAD WEIGHTS")
        # ppo.feature_extractor.load_state_dict(torch.load(modeldir+"feat_"+exp_name,  map_location=torch.device('cpu')))
        ppo.actor.load_state_dict(torch.load(modeldir+"actor_"+exp_name,  map_location=torch.device('cpu')))
        for param in ppo.actor.parameters():
            print(param.data)
        # exit()
        ppo.actor.eval()

while env.agents:
    if load_model:
        obs = batchify_obs(observations,"cpu")
        if ma:
            actions, logprobs, _, values = ppo.get_actions_and_values(obs,len(env.possible_agents))
        else:
            actions, logprobs, _, values = ppo.get_action_and_value(obs)
        actions = dict(zip(env.agents, actions))
    else: 
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("REWARDS: ",rewards)
    i = 0
    for agent in env.agents:
        if (i>len(r)-1):
            r.append(rewards[agent])
        else:
             r[i]+= rewards[agent]
        i+=1
    # r[0]+= rewards["player_0"]
    # r[1]+= rewards["player_1"]
    steps+=1
    print("Step: ",steps," | Cumulative Rewards: ",r)    
    # print("ACTIONS: ",actions)
    if True in terminations:
        print("DONE")
        break
print("Epsodic Return: ",r)
print("Mean Episodic Returns: ",np.mean(r))
env.close()