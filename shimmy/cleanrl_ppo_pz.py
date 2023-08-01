import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
from gymnasium.wrappers import GrayScaleObservation
from pettingzoo.butterfly import pistonball_v6

mpot = True

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        if mpot:
            self.network = nn.Sequential(
                self._layer_init(nn.Conv2d(3, 32, 3, padding=1)),#pixel observations, out channels 32
                nn.MaxPool2d(2),
                nn.ReLU(),
                self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
                nn.MaxPool2d(2),
                nn.ReLU(),
                self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                self._layer_init(nn.Linear(128 * 11 * 11, 512)),
                nn.ReLU(),
            )
        else:
            self.network = nn.Sequential(
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
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)#predict actions 0,1,2 for EACH agent (policy)
        self.critic = self._layer_init(nn.Linear(512, 1))#predict value (given state)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))# get value of state for each agent

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden) #probabilities
        probs = Categorical(logits=logits) # create a distribution
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden) #actions, logprobs, _, values 


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    if mpot: 
        obs = np.stack([obs[a]['RGB'] for a in obs], axis=0)#store only values and ignore keys
    else:
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
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    if mpot:
        frame_size = (88, 88)
        stack_size = 3
        num_steps = 1500 #default 1000
    else: 
        frame_size = (64, 64)
        stack_size = 4
        num_steps = 125
    total_episodes = 20

    """ ENV SETUP """
    if mpot:
        env = load_meltingpot("clean_up")
        env = MeltingPotCompatibilityV0(env, render_mode="None")
        next_obs, info = env.reset(seed=None)
        #print(next_obs[list(next_obs.keys())[0]]["RGB"].shape)
        #env = GrayScaleObservation(env)
        #env = color_reduction_v0(env, 'full') # grayscale
    else:
        env = pistonball_v6.parallel_env(
            render_mode="None", continuous=False, num_steps=num_steps
        )

        env = color_reduction_v0(env, 'full') # grayscale
        env = resize_v1(env, frame_size[0], frame_size[1]) # resize and stack images
        env = frame_stack_v1(env, stack_size=stack_size)

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)

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
    tb = SummaryWriter()
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset(seed=None)
            
            # reset the episodic return
            total_episodic_return = 0

            step_count = 0
            # each episode has num_steps
            terminated = False
            for step in range(0, num_steps):
                #print(step_count)
                step_count+=1
                # rollover the observation
                obs = batchify_obs(next_obs, device) # for torch 
                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env) # for PZ
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                #tb.add_scalar("total_episodic_return_0", total_episodic_return[0], episode)

                # if we reach termination or truncation, end
                if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                    end_step = step
                    terminated = True
                    print("TERMINATED")
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
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        #SAMPLE FROM BUFFER AND UPDATE NETWORKS (Learner)
        
        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clip_fracs = []
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
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
                value = value.flatten()
                v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                v_clipped = b_values[batch_index] + torch.clamp(
                    value - b_values[batch_index],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                #print("V_LOSS!!!!!!!!!!!!!!!")
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        tb.add_scalar("mean_total_episodic_return",np.mean(total_episodic_return), episode)
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        tb.add_scalar("v_loss", v_loss.item(), episode)
        print(f"Policy Loss: {pg_loss.item()}")
        tb.add_scalar("pg_loss", pg_loss.item(), episode)
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

    agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]
'''