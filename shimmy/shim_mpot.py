mpot = True
if mpot:
    from shimmy import MeltingPotCompatibilityV0
    from shimmy.utils.meltingpot import load_meltingpot
    env = load_meltingpot("clean_up")
    env = MeltingPotCompatibilityV0(env, render_mode="human")#human
else:
    from pettingzoo.butterfly import pistonball_v6
    env = pistonball_v6.parallel_env(render_mode="None")

observations = env.reset()
steps = 0
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(steps)
    steps+=1
    if(any(trunc for trunc in truncations.values())):
        print("ANY TRUNC",steps)
    if(all(trunc for trunc in truncations.values())):
        print("ALL TRUNC",steps)
    if(any(term for term in terminations.values())):
        print("ANY TERM",steps)    
    if(all(term for term in terminations.values())):
        print("ALL TERM",steps)    
    #print("TERM: ",terminations,"****")
    #print("TRUNC: ",truncations,"$$$$")
env.close()