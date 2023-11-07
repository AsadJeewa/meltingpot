# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from shimmy import MeltingPotCompatibilityV0
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
    parser.add_argument("--total_timesteps", type=int, default=1e7,
        help="total timesteps of the experiments")
    parser.add_argument("--batch_size", type=int, default=32,
        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
        help="the learning rate of the optimizer")
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--epochs", type=int, default=3,
        help="the K epochs to update the policy")
    parser.add_argument("--num_steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num_stacked", type=int, default=4,
        help="the number of stacked framed")
    parser.add_argument("--frame_size", type=int, default=64,#32
        help="the frame size of observations")
    parser.add_argument("--prosocial", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, prosocial training")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--ent_coef", type=float, default=0.1,#default 0.01
        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.1,#default=0.5
        help="coefficient of the value function")
    parser.add_argument("--clip_coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            # layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            # #no maxpool
            # nn.ReLU(),
            # layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            # nn.ReLU(),
            # layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            # nn.ReLU(),
            # nn.Flatten(),
            # layer_init(nn.Linear(16, 512)),#from 84x84 to 64x64
            # nn.ReLU(),

            layer_init(nn.Conv2d(4, 32, 3, padding=1)),#pixel observations, out channels 32
            nn.MaxPool2d(2),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Flatten(start_dim=0,end_dim=2),
            layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
def batchify_obs(obs, device):#stack agents
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)#store only values and ignore keys
    # transpose to be (batch, channel, height, width)
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
    
if __name__ == "__main__":
    args = parse_args()
    exp_name = f"{args.exp_name}__{args.env_id}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(log_dir="runs/"+exp_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_steps = args.num_steps #default 1000
    frame_size = (args.frame_size, args.frame_size)
    stack_size = args.num_stacked

    """ ENV SETUP """
    env = load_meltingpot(args.env_id)
    env = MeltingPotCompatibilityV0(env, render_mode="None")
    env = color_reduction_v0(env, 'full') # grayscale
    env = resize_v1(env, frame_size[0], frame_size[1]) # resize and stack images
    env = frame_stack_v1(env, stack_size=stack_size)

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    agent = Agent(num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_episodic_return = 0
    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_agents, stack_size, *frame_size)).to(device)#EPISODE
    actions = torch.zeros(num_steps, num_agents).to(device)
    logprobs = torch.zeros(num_steps, num_agents).to(device)
    rewards = torch.zeros(num_steps, num_agents).to(device)
    terms = torch.zeros(num_steps, num_agents).to(device)
    values = torch.zeros(num_steps, num_agents).to(device)#for each agent 

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = int(args.total_timesteps // num_steps)#2e6 is float

    for update in range(1, num_updates + 1):
        print("COLLECTING EXPERIENCE")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        total_episodic_return = 0

        next_obs, info = env.reset(seed=args.seed)
        # next_obs = next_obs[env.possible_agents[0]]
        # next_obs = np.swapaxes(next_obs,0,2)
        # next_obs = torch.from_numpy(np.expand_dims(next_obs, axis=0))#add aghent dim for pz compatability
        next_term = {env.possible_agents[0]: False}
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = batchify_obs(next_obs, device)
            terms[step] = batchify(next_term, device)
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs[step])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, term, trunc, info = env.step(unbatchify(action, env))
            rewards[step] = batchify(reward, device)
            total_episodic_return+= rewards[step].cpu().numpy()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(obs[step]).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - terms[step]#next_term is numpy
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terms[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gamma * nextnonterminal * lastgaelam#args gae lambda
            returns = advantages + values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(obs[:num_steps], start_dim=0, end_dim=1)#single transition (ignore agent)
        b_logprobs = torch.flatten(logprobs[:num_steps], start_dim=0, end_dim=1)
        b_actions = torch.flatten(actions[:num_steps], start_dim=0, end_dim=1)
        b_returns = torch.flatten(returns[:num_steps], start_dim=0, end_dim=1)
        b_values = torch.flatten(values[:num_steps], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(advantages[:num_steps], start_dim=0, end_dim=1)
        #SAMPLE FROM BUFFER AND UPDATE NETWORKS (Learner)

        # Optimizing the policy and value network
        b_inds = np.arange(len(b_obs))

        print("TRAINING")
        clipfracs = []
        for epoch in range(args.epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs) , args.batch_size):#minibatch
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        writer.add_scalar("mean_episodic_return",np.mean(total_episodic_return), global_step)

        print(f"Episode Length: {global_step}")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()