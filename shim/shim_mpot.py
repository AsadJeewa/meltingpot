import torch
import numpy as np
from shim.cleanrl_mappo_shared import MAPPO, batchify_obs
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

#DEBUG START
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

        exp_name = "sanity_single__clean_up_simple__3__1698623132"
        # exp_name = "noturn_gpu_ctde_state_multicoord__clean_up_simple__1__1697719103"

        modeldir = "model/TO PRESENT/"+exp_name
        mappo = MAPPO(env.action_space(env.possible_agents[0]).n)
        print(env.action_space(env.possible_agents[0]).n)
        # exit()
        # print("INIT")
        # for param in mappo.actor.parameters():
        #     print(param.data)
        # print("LOAD WEIGHTS")
        mappo.actor.load_state_dict(torch.load(modeldir,  map_location=torch.device('cpu')))
        for param in mappo.actor.parameters():
            print(param.data)
        # exit()
        # mappo.actor.eval()

while env.agents:
    if load_model:
        obs = batchify_obs(observations,"cpu")
        actions, logprobs, _, values = mappo.get_actions_and_values(obs,len(env.possible_agents))
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