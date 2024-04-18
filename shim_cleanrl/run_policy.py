import torch
import numpy as np
from mappo_shared import MAPPO
from ppo import PPO
from utils import batchify_obs
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from matplotlib import pyplot as plt
import seaborn as sb
import os
import csv

#DEBUG START
load_model = True

ma = False
pz  = False
pomdp = False

frame_size = (64, 64)
stack_size = 4

plot_actions = True
#DEBUG END

def plot(actionTrack):
    sb.set_theme()
    # exp_name = "ppo"
    y_ticks = [-1, 0, 1]
    # y_ticks = [0, 1, 2]
    y_labels = ['clean', 'move', 'eat']
    # y_ticks = [-1, 0]
    # y_labels = ['clean', 'move']
    fig, ax = plt.subplots(1,1)
    plt.figure().set_figwidth(15)
    plt.yticks(y_ticks, y_labels)
    # ax.set_yticklabels(y_labels)
    plt.ylim([-1,1])
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.title(exp_name)
    # actionTrack = [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]
    plt.plot(actionTrack,linewidth=1)
    # plt.scatter(range(len(actionTrack)),actionTrack,s=0.5)
    plt.savefig(os.path.join(os.getcwd(),"fig/"+exp_name+"_actionTrack.png"))
    print("SAVED AT: ",os.path.join(os.getcwd(),"fig/"+exp_name+"_actionTrack.png"))
    print(actionTrack)
    # plt.show()

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
actionTrack = []
cleanOrEat = False

if load_model:
        # exp_name = "gpu_ctde_state_4x4__clean_up_simple__1__1695335842"
        # exp_name = "gpu_ctde_state_4x4_divergent__clean_up_simple__1__1695350009"
        # exp_name = "mappo_shared__pistonball__1475661751__1702916831"
        # exp_name = "ppo__clean_up_simple__1743393291__1702852185"
        # exp_name = "fixed_ppo_single_hard__clean_up_simple__86427174__1700575418"
        # exp_name = "fixed_ppo_single_easy_cpu__clean_up_simple__726506732__1700571979"
        # exp_name = "ppo_single_2__clean_up_simple__582671304__1700611799"
        # exp_name = "save_ppo_single_easy__clean_up_simple__1933059180__1700078728"
        # exp_name = "save_full_ppo_single_hard__clean_up_simple__121193687__1700102084"
        # exp_name = "noturn_gpu_ctde_state_multicoord__clean_up_simple__1__1697719103"

        exp_name = "ppo_single_small__clean_up_simple__489263223__1711101428"
        # exp_name = "ppo_single_4x4_divergent_2eat__clean_up_simple__49690449__1712575972"
        # exp_name = "ppo_single_4x4_divergent__clean_up_simple__1317678540__1712571018"
        # exp_name = "ppo_single_large__clean_up_simple__1045332753__1711356834"
        # exp_name = "ppo_single_8x8_divergent_2eat__clean_up_simple__1487285556__1712678194"
        # exp_name = "ppo_single_8x8_divergent__clean_up_simple__1436946905__1712167722"
        # exp_name = "ppo_single_16x16__clean_up_simple__566093860__1712134889"

        modeldir = "model/LATEST/"
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
else: 
    exp_name = "random"
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
    i = 0
    for agent in env.agents:
        print(infos[agent])
        if (i>len(r)-1):
            r.append(rewards[agent])
        else:
             r[i]+= rewards[agent]
        if(actions[agent]==5):
            numCleanAttempts+=1
            print("ATTEMPT CLEAN")
        if(infos[agent][1]==1):#Vector Reward    
            numCleanSuccesses+=1
            cleanOrEat = True
            actionTrack.append(-1)
            print("SUCCESS CLEAN")
        if(infos[agent][0]==1):
            cleanOrEat = True
            actionTrack.append(1)
            print("EAT")
        i+=1
    steps+=1
    print("Step: ",steps," | Cumulative Rewards: ",r)    
    if(not cleanOrEat):
        actionTrack.append(0)
    cleanOrEat = False
    if True in terminations:
        print("DONE")
        break
print("Episodic Return: ",r)
print("Mean Episodic Returns: ",np.mean(r))

cleanEfficiencyRatio =  numCleanSuccesses/ numCleanAttempts if numCleanAttempts != 0 else 0
appleRatio = np.mean(r) / steps
cleanRatio =  numCleanSuccesses/ steps
print("Clean Attempts: ",numCleanAttempts, "Clean Successes: ", numCleanSuccesses, " =", cleanEfficiencyRatio)
print("Mean Episodic Returns: ",np.mean(r))
print("Apple Ratio: ", appleRatio)
print("Clean Ratio: ", cleanRatio)
print(exp_name)
if plot_actions:
    plot(actionTrack)
env.close()

fields = ['Exp', 'Num Steps', 'Clean Attempts', 'Clean Successes', 'Clean Efficiency Ratio', 'Apple Ratio', 'Clean Ratio', 'Mean Episodic Return']
filename = "exp_log.csv"
with open(filename, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    # writer.writeheader()
    myDict = [{'Exp': exp_name, 'Num Steps': steps, 'Clean Attempts': numCleanAttempts, 'Clean Successes': numCleanSuccesses, 'Clean Efficiency Ratio': cleanEfficiencyRatio, 'Apple Ratio': appleRatio, 'Clean Ratio': cleanRatio, 'Mean Episodic Return': np.mean(r)}]
    writer.writerows(myDict)