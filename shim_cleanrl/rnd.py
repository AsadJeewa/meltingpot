# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/#ppo_rnd_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from utils import batchify_obs, batchify, unbatchify, layer_init, make_env

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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    env_id: str = "clean_up_simple"
    """the id of the environment"""
    total_timesteps: int = 1e7
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_steps: int = 1000
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RND arguments
    update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    int_coef: float = 1.0
    """coefficient of extrinsic reward"""
    ext_coef: float = 2.0
    """coefficient of intrinsic reward"""
    int_gamma: float = 0.99
    """Intrinsic reward discount rate"""
    num_iterations_obs_norm_init: int = 50
    """number of iterations to initialize the observations normalization parameters"""

    #TODO match to my ppo
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # added
    stack_size: int = 4
    """the number of stacked frames"""
    frame_size: int=64
    """the frame size of observations"""
    checkpoint_window: int=10
    """checkpoint window size"""

#TODO Check RecordEpisodeStatistics
# class RecordEpisodeStatistics(gym.Wrapper):
#     def __init__(self, env, deque_size=100):
#         super().__init__(env)
#         self.episode_returns = None
#         self.episode_lengths = None

#     def reset(self, **kwargs):
#         observations = super().reset(**kwargs)
#         self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
#         self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
#         self.lives = np.zeros(self.num_envs, dtype=np.int32)
#         self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
#         self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
#         return observations

#     def step(self, action):
#         observations, rewards, dones, infos = super().step(action)
#         self.episode_returns += infos["reward"]
#         self.episode_lengths += 1
#         self.returned_episode_returns[:] = self.episode_returns
#         self.returned_episode_lengths[:] = self.episode_lengths
#         self.episode_returns *= 1 - infos["terminated"]
#         self.episode_lengths *= 1 - infos["terminated"]
#         infos["r"] = self.returned_episode_returns
#         infos["l"] = self.returned_episode_lengths
#         return (
#             observations,
#             rewards,
#             dones,
#             infos,
#         )


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RND(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 3, padding=1)),#stacked frame, pixel observations, out channels 32
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
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(512, 512), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, 512), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(512, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(512, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.feature_extractor(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.feature_extractor(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # feature_output = 7 * 7 * 64
        feature_output = 1024

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.exp_name}__{args.env_id}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id,args.frame_size,args.stack_size)
    num_steps = args.num_steps #default 1000
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape
    # envs.single_action_space = envs.action_space
    # envs.single_observation_space = envs.observation_space
    # envs = RecordEpisodeStatistics(envs)
    # assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    rnd = RND(num_actions).to(device)
    rnd_model = RNDModel(4, num_actions).to(device)
    combined_parameters = list(rnd.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    total_episodic_return = 0
    apple_episodic_return = 0
    curiosity_episodic_return = 0
    current_best = 0
    checkpoint_window = args.checkpoint_window
    window_episodic_return = np.zeros(checkpoint_window)

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(4, 64, 64)) #TODO check normalisation
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_agents, args.stack_size, *((args.frame_size, args.frame_size)))).to(device)#EPISODE
    actions = torch.zeros(num_steps, num_agents).to(device)
    logprobs = torch.zeros(num_steps, num_agents).to(device)
    rewards = torch.zeros(num_steps, num_agents).to(device)
    
    curiosity_rewards = torch.zeros(num_steps, num_agents).to(device)
    
    terms = torch.zeros(num_steps, num_agents).to(device)
    
    ext_values = torch.zeros(num_steps, num_agents).to(device)
    int_values = torch.zeros(num_steps, num_agents).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    num_updates = int(args.total_timesteps // num_steps)
    
    #TODO CHECK
    # print("Start to initialize observation normalization parameter.....")
    # next_ob = []
    # for step in range(args.num_steps * args.num_iterations_obs_norm_init):
    #     acs = np.random.randint(0, num_actions)
    #     next_obs, r, d, _ = envs.step(acs)
    #     next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

    #     if len(next_ob) % (args.num_steps * args.num_envs) == 0:
    #         next_ob = np.stack(next_ob)
    #         obs_rms.update(next_ob)
    #         next_ob = []
    # print("End to initialize...")
    index = 0
    for update in range(1, num_updates + 1):
        print("COLLECTING EXPERIENCE")
        next_obs, info = env.reset(seed=args.seed)
        next_term = {env.possible_agents[0]: False}
        total_episodic_return = 0
        curiosity_episodic_return = 0
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = batchify_obs(next_obs, device)
            terms[step] = batchify(next_term, device)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = rnd.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = rnd.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, term, trunc, info = env.step(unbatchify(action, env))
            rewards[step] = batchify(reward, device)
            total_episodic_return += rewards[step].cpu().numpy()
            curiosity_episodic_return += curiosity_rewards[step].cpu().numpy()
            rnd_next_obs = (
                (
                    (obs[step] - torch.from_numpy(obs_rms.mean).to(device)) # single agent
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
            ).float()
            target_next_feature = rnd_model.target(rnd_next_obs)
            # print("RND: ",obs[step].shape)
            # print("TARGET: ",rnd_next_obs.shape)
            # print(target_next_feature.shape)
            # exit()
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            
            #TODO CHECK Curiosity reward
            # for idx, d in enumerate(term):
            #     if d and info["lives"][idx] == 0:
            #         avg_returns.append(info["r"][idx])
            #         epi_ret = np.average(avg_returns)
            #         print(
            #             f"global_step={global_step}, episodic_return={info['r'][idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
            #         )
            #         writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
            #         writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
            #         writer.add_scalar(
            #             "charts/episode_curiosity_reward",
            #             curiosity_rewards[step][idx],
            #             global_step,
            #         )
            #         writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = rnd.get_value(obs[step])
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - terms[step]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - terms[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = torch.flatten(obs[:num_steps], start_dim=0, end_dim=1)#single transition (ignore agent)
        b_logprobs = torch.flatten(logprobs[:num_steps], start_dim=0, end_dim=1)
        b_actions = torch.flatten(actions[:num_steps], start_dim=0, end_dim=1)
        b_ext_advantages = torch.flatten(ext_advantages[:num_steps], start_dim=0, end_dim=1)
        b_int_advantages = torch.flatten(int_advantages[:num_steps], start_dim=0, end_dim=1)
        b_ext_returns = torch.flatten(ext_returns[:num_steps], start_dim=0, end_dim=1)
        b_int_returns = torch.flatten(int_returns[:num_steps], start_dim=0, end_dim=1)
        b_ext_values = torch.flatten(ext_values[:num_steps], start_dim=0, end_dim=1)

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())
        obs_rms.update(b_obs.cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        rnd_next_obs = (
            (
                (b_obs - torch.from_numpy(obs_rms.mean).to(device))
                / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
            ).clip(-5, 5)
        ).float()

        clipfracs = []
        print("TRAINING")
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs) , args.batch_size):#minibatch
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                _, newlogprob, entropy, new_ext_values, new_int_values = rnd.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
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
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        
        if(update > checkpoint_window):
            window_episodic_return = np.delete(window_episodic_return,0)
            window_episodic_return = np.append(window_episodic_return, total_episodic_return)
        else:
            print("EARLY: ",window_episodic_return, total_episodic_return)
            window_episodic_return[index] = total_episodic_return
            index+=1

        print("MEAN: ",np.mean(window_episodic_return))
        if np.mean(window_episodic_return) > current_best:
            torch.save(rnd.feature_extractor.state_dict(), "model/feat_"+run_name)
            torch.save(rnd.actor.state_dict(), "model/actor_"+run_name)
            current_best = np.mean(window_episodic_return)
            print("SAVE")

        print(f"Curiosity Return: {curiosity_episodic_return}")
        print(f"Episodic Return: {total_episodic_return}")
        print(f"Apple Episodic Return: {apple_episodic_return}")
        writer.add_scalar("mean_episodic_return",total_episodic_return, global_step)
        writer.add_scalar("curiosity_episodic_return",curiosity_episodic_return, global_step)

        print(f"Global Step: {global_step}")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print("\n-------------------------------------------\n")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()