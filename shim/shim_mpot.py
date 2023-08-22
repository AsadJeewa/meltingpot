import torch
from shim.cleanrl_ppo_pz import Agent, batchify_obs

mpot = True
if mpot:
    from shimmy import MeltingPotCompatibilityV0
    from shimmy.utils.meltingpot import load_meltingpot
    env = load_meltingpot("clean_up")#prisoners_dilemma_in_the_matrix__arena
    env = MeltingPotCompatibilityV0(env, render_mode="human")#human
else:
    from pettingzoo.butterfly import pistonball_v6
    env = pistonball_v6.parallel_env(render_mode="None")
observations = env.reset()[0]#wrapped in tuple
steps = 0
while env.agents:
    # actions_ = {agent: env.action_space(agent).sample() for agent in env.agents}
    exp_name = "divergent_long_0_2023-08-10 17:32:57.855943"
    agent = Agent(env.action_space(env.possible_agents[0]).n)
    model = agent.actor
    # model.load_state_dict(torch.load("model/"+exp_name))
    model.eval()
    obs = batchify_obs(observations,"cpu", True)
    actions, logprobs, _, values = agent.get_action_and_value(obs)
    actions = dict(zip(env.agents, actions))
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for player in observations.keys():
        print(player,": ",observations[player]["VECTOR_REWARD"])
    print("*************")
    steps+=1
env.close()