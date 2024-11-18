# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool
from dataclasses import dataclass
import tyro

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from utils import batchify_obs, batchify, unbatchify, layer_init, make_env

#BEGIN DEBUG
#END DEBUG

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = random.randint(0,np.iinfo(np.int32).max)
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "mpot_cleanrl"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    #TODO Capture Video
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""


    # Algorithm specific arguments
    env_id: str = "clean_up_simple" #"MiniGrid-Empty-5x5-v0"
    """the id of the environment"""
    total_timesteps: int = 1e7
    """total timesteps of the experiments"""
    batch_size: int = 32
    """the batch size"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    #TODO fix vectorisation
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 1000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1
    """coefficient of the entropy"""
    vf_coef: float = 0.1
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # added
    stack_size: int = 4
    """the number of stacked frames"""
    frame_size: int=64
    """the frame size of observations"""
    checkpoint_window: int=10
    """checkpoint window size"""
    shimmy: bool=False
    """shimmy env (melting pot)"""

class PPO(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
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
        return self.critic(self.feature_extractor(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.feature_extractor(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
def main():
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.env_id}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_dir="runs/"+run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    num_steps = args.num_steps #default 1000

    """ ENV SETUP """
    env = make_env(args.env_id,args.frame_size,args.stack_size, args.shimmy, args.capture_video, run_name)
    print(env)
    if(args.shimmy):
        num_agents = len(env.possible_agents)
        num_actions = env.action_space(env.possible_agents[0]).n
    else:
        num_agents = 1
        num_actions = env.action_space[0].n
    
    ppo = PPO(num_actions).to(device)
    optimizer = optim.Adam(ppo.parameters(), lr=args.learning_rate, eps=1e-5)

    total_episodic_return = 0
    apple_episodic_return = 0
    current_best = 0
    checkpoint_window = args.checkpoint_window
    window_episodic_return = np.zeros(checkpoint_window)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_agents, args.stack_size, *((args.frame_size, args.frame_size)))).to(device)#EPISODE
    actions = torch.zeros(num_steps, num_agents).to(device)
    logprobs = torch.zeros(num_steps, num_agents).to(device)
    rewards = torch.zeros(num_steps, num_agents).to(device)
    terms = torch.zeros(num_steps, num_agents).to(device)
    values = torch.zeros(num_steps, num_agents).to(device)#for each agent 

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = int(args.total_timesteps // num_steps)#2e6 is float
    index = 0
    for update in range(1, num_updates + 1):
        print("COLLECTING EXPERIENCE")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        total_episodic_return = 0
        apple_episodic_return = 0

        next_obs, info = env.reset(seed=args.seed)
        # next_obs = next_obs[env.possible_agents[0]]
        # next_obs = np.swapaxes(next_obs,0,2)
        # next_obs = torch.from_numpy(np.expand_dims(next_obs, axis=0))#add aghent dim for pz compatability
        if args.shimmy: 
            next_term = {env.possible_agents[0]: False}
        else: 
            next_term = {'agent': False} #TODO Fix
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = batchify_obs(next_obs, device)
            terms[step] = batchify(next_term, device)

            # ALGO LOGIC: action logic
            ppo.feature_extractor.eval() #TODO Check
            ppo.actor.eval() #TODO Check
            ppo.critic.eval() #TODO Check
            with torch.no_grad():
                action, logprob, _, value = ppo.get_action_and_value(obs[step])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, term, trunc, info = env.step(unbatchify(action, env))
            rewards[step] = batchify(reward, device)
            total_episodic_return += rewards[step].cpu().numpy()
        # bootstrap value if not done
        with torch.no_grad():
            next_value = ppo.get_value(obs[step]).reshape(1, -1)
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
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam#args gae lambda
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

        ppo.feature_extractor.train()
        ppo.actor.train()
        ppo.critic.train()
        print("TRAINING")
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs) , args.batch_size):#minibatch
                end = start + args.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = ppo.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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
                nn.utils.clip_grad_norm_(ppo.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if(update > checkpoint_window):
            window_episodic_return = np.delete(window_episodic_return,0)
            window_episodic_return = np.append(window_episodic_return, total_episodic_return)
        else:
            print("EARLY: ",window_episodic_return, total_episodic_return)
            window_episodic_return[index] = total_episodic_return
            index+=1

        print("MEAN: ",np.mean(window_episodic_return))
        if np.mean(window_episodic_return) > current_best:
            torch.save(ppo.feature_extractor.state_dict(), "model/feat_"+run_name)
            torch.save(ppo.actor.state_dict(), "model/actor_"+run_name)
            current_best = np.mean(window_episodic_return)
            print("SAVE")

        print(f"Episodic Return: {total_episodic_return}")
        print(f"Apple Episodic Return: {apple_episodic_return}")
        writer.add_scalar("mean_episodic_return",total_episodic_return, global_step)
        if args.track:
            wandb.log({"mean_episodic_return": total_episodic_return})

        print(f"Global Step: {global_step}")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Explained Variance: {explained_var.item()}")
        print("\n-------------------------------------------\n")

        # writer.add_text("info/learning_rate", optimizer.param_groups[0]["lr"], global_step)
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

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.track:
        import wandb
        sweep_configuration = {
            "method": "random",
            "metric": {"goal": "minimize", "name": "total_episodic_return"},
            "parameters": {
                "learning_rate": {"min": 0.00001, "max": 0.01},
            },
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_project_name)
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()