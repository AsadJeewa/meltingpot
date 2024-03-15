import torch
import numpy as np
from mappo_shared import MAPPO
from ppo import PPO
from utils import batchify_obs
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

#DEBUG START
load_model = True

ma = False
pz  = False
pomdp = True

frame_size = (64, 64)
stack_size = 4
#DEBUG END

if pz:
    from pettingzoo.butterfly import pistonball_v6
    env = pistonball_v6.parallel_env(
            render_mode="human", continuous=False
        )#TODO Add num_steps
else:
    from shimmy import MeltingPotCompatibilityV0
    from shimmy.utils.meltingpot import load_meltingpot
    env = load_meltingpot("clean_up_simple")#prisoners_dilemma_in_the_matrix__arena
    env = MeltingPotCompatibilityV0(env, render_mode="human")#human #TODO Add num_steps
env = color_reduction_v0(env, 'full') # grayscale
env = resize_v1(env, frame_size[0], frame_size[1]) # resize 
env = frame_stack_v1(env, stack_size=stack_size) # stack

observations = env.reset()[0]#wrapped in tuple
steps = 0
r = []
numCleanAttempts = 0
numCleanSuccesses = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if load_model:
        # exp_name = "gpu_ctde_state_4x4__clean_up_simple__1__1695335842"
        # exp_name = "gpu_ctde_state_4x4_divergent__clean_up_simple__1__1695350009"

        # exp_name = "mappo_shared__pistonball__1475661751__1702916831"

        exp_name = "ppo__clean_up_simple__1743393291__1702852185"
        # exp_name = "fixed_ppo_single_hard__clean_up_simple__86427174__1700575418"
        # exp_name = "fixed_ppo_single_easy_cpu__clean_up_simple__726506732__1700571979"
        # exp_name = "ppo_single_2__clean_up_simple__582671304__1700611799"
        # exp_name = "save_ppo_single_easy__clean_up_simple__1933059180__1700078728"
        # exp_name = "save_full_ppo_single_hard__clean_up_simple__121193687__1700102084"
        # exp_name = "noturn_gpu_ctde_state_multicoord__clean_up_simple__1__1697719103"

        modeldir = "model/TO PRESENT/"
        num_actions = env.action_space(env.possible_agents[0]).n
        if ma:
            ppo = MAPPO(num_actions)
        else:
            ppo = PPO(num_actions)#MAPPO 
        print("INIT")
        for param in ppo.feature_extractor.parameters():
            print(param.data)
            break
        print("LOAD WEIGHTS")
        ppo.feature_extractor.load_state_dict(torch.load(modeldir+"feat_"+exp_name,  map_location=torch.device('cpu')))
        ppo.actor.load_state_dict(torch.load(modeldir+"actor_"+exp_name,  map_location=torch.device('cpu')))
        for param in ppo.feature_extractor.parameters():
            print(param.data)
            break
        ppo.actor.eval()

while env.agents:
    if load_model:
        obs = batchify_obs(observations,"cpu")
        if ma:
            actions, logprobs, _, values = ppo.get_actions_and_values(obs, pomdp, len(env.possible_agents))
        else:
            actions, logprobs, _, values = ppo.get_action_and_value(obs)
        actions = dict(zip(env.agents, actions.tolist()))
    else: 
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    if(actions["player_0"]==5):
        numCleanAttempts+=1
    if(infos["player_0"][1]==1):    
        numCleanSuccesses+=1
    i = 0
    for agent in env.agents:
        if (i>len(r)-1):
            r.append(rewards[agent])
        else:
             r[i]+= rewards[agent]
        i+=1
    steps+=1
    print("Step: ",steps," | Cumulative Rewards: ",r)    
    if True in terminations:
        print("DONE")
        break
print("Episodic Return: ",r)
print("Mean Episodic Returns: ",np.mean(r))

cleanWasteRatio =  numCleanSuccesses/ numCleanAttempts
appleRatio = np.mean(r) / steps
cleanRatio =  numCleanSuccesses/ steps
print("Clean Attempts: ",numCleanAttempts, "Clean Successes: ", numCleanSuccesses, " =", cleanWasteRatio)
print("Mean Episodic Returns: ",np.mean(r))
print("Apple Ratio: ", appleRatio)
print("Clean Ratio: ", cleanRatio)
env.close()