import torch
import numpy as np
from shim.cleanrl_mappo_shared import Agent, batchify_obs
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
r = [0,0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while env.agents:
    
    if load_model:
        # exp_name = "ctde_state_4x4_divergent__clean_up_simple__1__20-09-2023_14-51-32"
        exp_name = "ctde_state_4x4__clean_up_simple__1__20-09-2023_14-51-30"
        # exp_name = "gpu_ctde_state_4x4__clean_up_simple__1__20-09-2023_21-46-14"
        # exp_name = "ctde_pistonball__clean_up_simple__1__20-09-2023_16-04-10"
        modeldir = "model/current/"+exp_name
        agent = Agent(env.action_space(env.possible_agents[0]).n)
        model = agent.actor
        model.load_state_dict(torch.load(modeldir,  map_location=torch.device('cpu')))
        model.eval()
        obs = batchify_obs(observations,"cpu")
        actions, logprobs, _, values = agent.get_actions_and_values(obs,len(env.possible_agents))
        actions = dict(zip(env.agents, actions))
    else: 
        obs = batchify_obs(observations,"cpu")
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    r[0]+= rewards["player_0"]
    r[1]+= rewards["player_1"]
    steps+=1
    print("Step: ",steps," | Rewards: ",r)
    if True in terminations:
        print("DONE")
        break
print("Epsodic Return: ",r)
print("Mean Episodic Returns: ",np.mean(r))
env.close()