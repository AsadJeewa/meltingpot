import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.butterfly import pistonball_v6
import argparse
import os
import random
from distutils.util import strtobool
import time

from utils import batchify_obs, batchify, unbatchify
from shimmy import MeltingPotCompatibilityV0 #venv
from shimmy.utils.meltingpot import load_meltingpot

#BEGIN DEBUG

#END DEBUG

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=random.randint(0,np.iinfo(np.int32).max),
        help="seed of the experiment")
    # Algorithm specific arguments
    parser.add_argument("--env_id", type=str, default="clean_up_simple",
        help="the id of the environment")
    parser.add_argument("--timesteps", type=int, default=2e6,
        help="total timesteps of the experiments")
    parser.add_argument("--batch_size", type=int, default=32,
        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
        help="the learning rate of the optimizer")
    parser.add_argument("--epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--num_steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num_stacked", type=int, default=4,
        help="the number of stacked framed")
    parser.add_argument("--frame_size", type=int, default=64,#32
        help="the frame size of observations")
    parser.add_argument("--pz", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, pz environment is used instead of mpot")
    parser.add_argument("--prosocial", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, prosocial training")
    
    args = parser.parse_args()
    return args

class MAPPO(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),#pixel observations, out channels 32
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)#predict actions 0,1,2 for one agent (policy)
        self.critic = self._layer_init(nn.Linear(512, 1))#predict value (given state)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_values(self, x):
        return self.critic(self.feature_extractor(x / 255.0))# get value of state for each agent

    def get_actions_and_values(self, x, num_agents, device="cpu", actions=None):
        # x is combined observations
        upper_bound = x.shape[0]#batch size
        genActions = False
        if actions is None:
            actions = torch.zeros(num_agents).to(device)
            genActions = True
        hidden_all = self.feature_extractor(x / 255.0)
        #TODO Add param for parameter sharing
        for i in range(upper_bound):
            # NAN ISSUE
            hidden = hidden_all[i]
            logits = self.actor(hidden) #probabilities
            probs = Categorical(logits=logits) # create a distribution
            if genActions:
                action = probs.sample()
                actions[i] = action
        actions = actions.int()
        return actions, probs.log_prob(actions), probs.entropy(), self.critic(hidden_all) #actions, logprobs, _, values 

if __name__ == "__main__":
    args = parse_args()
    pz = args.pz
    exp_name = f"{args.exp_name}__{args.env_id}__{args.seed}__{int(time.time())}"
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = args.batch_size
    total_timesteps = args.timesteps
    num_epochs = args.epochs

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
    if pz: 
        num_steps = 125
    else:
        num_steps = args.num_steps #default 1000
        # frame_size = (88, 88)
        # stack_size = 3
    
    frame_size = (args.frame_size, args.frame_size)
    stack_size = args.num_stacked

    """ ENV SETUP """
    if pz:
        env = pistonball_v6.parallel_env(
            render_mode="None", continuous=False, max_cycles=num_steps
        )
    else:
        env = load_meltingpot(args.env_id)
        env = MeltingPotCompatibilityV0(env, render_mode="None")
    

    env = color_reduction_v0(env, 'full') # grayscale
    env = resize_v1(env, frame_size[0], frame_size[1]) # resize 
    env = frame_stack_v1(env, stack_size=stack_size) # stack

    if pz:
        num_agents = 1
        num_actions = env.action_space.n
        observation_size = env.observation_space.shape
    else:
        num_agents = len(env.possible_agents)
        num_actions = env.action_space(env.possible_agents[0]).n
        observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    mappo = MAPPO(num_actions=num_actions).to(device)
    optimizer = optim.Adam(mappo.parameters(), lr=args.learning_rate, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    # replay buffer
    '''
    if mpot:
        rb_obs = torch.zeros((num_agents, stack_size, *frame_size)).to(device)
        rb_actions = torch.zeros((num_agents)).to(device)
        rb_logprobs = torch.zeros((num_agents)).to(device)
        rb_rewards = torch.zeros((num_agents)).to(device)
        rb_terms = torch.zeros((num_agents)).to(device)
        rb_values = torch.zeros((num_agents)).to(device)#for each agent 
    else:
    '''
    rb_obs = torch.zeros((num_steps, num_agents, stack_size, *frame_size)).to(device)#EPISODE
    rb_actions = torch.zeros((num_steps, num_agents)).to(device)
    rb_logprobs = torch.zeros((num_steps, num_agents)).to(device)
    rb_rewards = torch.zeros((num_steps, num_agents)).to(device)
    rb_terms = torch.zeros((num_steps, num_agents)).to(device)
    rb_values = torch.zeros((num_steps, num_agents)).to(device)#for each agent 

    """ TRAINING LOGIC """
    # train for n number of episodes
    tb = SummaryWriter(log_dir="runs/"+exp_name)
    num_updates = total_timesteps // num_steps
    step_count = 0
    for update in range(1, int(num_updates) + 1):
        print("COLLECTING EXPERIENCE")
        # collect an episode
        mappo.feature_extractor.eval()
        mappo.actor.eval()
        mappo.critic.eval()
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=args.seed)
            # reset the episodic return
            total_episodic_return = 0
            
            # each episode has num_steps
            terminated = False
            for step in range(0, num_steps):
                step_count+=1
                # rollover the observation
                obs = batchify_obs(next_obs, device) # for torch 
                # get action from the agent
                actions, logprobs, _, values = mappo.get_actions_and_values(obs, num_agents, device)
                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env) # info has vector reward
                )
                # add to episode storage
                rb_obs[step] = obs #BUFFER STORES EACH TRANSITION
                rb_rewards[step] = batchify(rewards, device) #each agent separate
                #ONE EPISODE AT AT TIME
                if args.prosocial:
                    rb_rewards[step]  = torch.mean(rb_rewards[step])
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    terminated = True
                    break
            #if not terminated:
            #    end_step = num_steps-1 #CHECK

        # GENERATE EXPERIENCE FOR REPLAY BUFFER (Actor)

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            for t in reversed(range(end_step)):
                #GAE empirical returns minus the value function baseline
                delta = (
                    rb_rewards[t]
                    + gamma * rb_values[t + 1] * rb_terms[t + 1]
                    - rb_values[t]
                )
                rb_advantages[t] = delta + gamma * gamma * rb_advantages[t + 1]
            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)#single transition (ignore agent)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        #SAMPLE FROM BUFFER AND UPDATE NETWORKS (Learner)        
        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []

        mappo.feature_extractor.train()
        mappo.actor.train()
        mappo.critic.train()
        print("TRAINING")#ON COLLECTED EXPERIENCE
        for repeat in range(num_epochs):#pass over batch n times
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            #SHUFFLE FOR TRAINING
            #PASS THROUGH ENTIRE EPISODE IN SHUFFLED BATCHES
            for start in range(0, len(b_obs), batch_size):#minibatch
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]#shuffled
                _, newlogprob, entropy, values = mappo.get_actions_and_values(
                    b_obs[batch_index], num_agents, device, b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp() #divergence

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() #clipping

                # Value loss
                v_loss_unclipped = (values - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    values - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        torch.save(mappo.feature_extractor.state_dict(), "model/feat_"+exp_name)
        torch.save(mappo.actor.state_dict(), "model/actor_"+exp_name)
        print("save")

        print(f"Mean Episodic Return: {np.mean(total_episodic_return)}")
        tb.add_scalar("mean_episodic_return",np.mean(total_episodic_return), step_count)
        print(f"Step Count: {step_count}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        tb.add_scalar("v_loss", v_loss.item(), step_count)
        print(f"Policy Loss: {pg_loss.item()}")
        tb.add_scalar("pg_loss", pg_loss.item(), step_count)
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

    tb.close()
'''
    """ RENDER THE POLICY """
    env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
    env = color_reduction_v0(env)
    env = resize_v1(env, 64, 64)
    env = frame_stack_v1(env, stack_size=4)

    mappo.actor.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_actions_and_values(obs, num_agents, device)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
'''