# NOTE taken from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# IMPORTS: 
# from gym.wrappers.normalize import RunningMeanStd


# # RND arguments
# update_proportion: float = 0.25
# """proportion of exp used for predictor update"""
# int_coef: float = 1.0
# """coefficient of extrinsic reward"""
# ext_coef: float = 2.0
# """coefficient of intrinsic reward"""
# int_gamma: float = 0.99
# """Intrinsic reward discount rate"""
# num_iterations_obs_norm_init: int = 50
# """number of iterations to initialize the observations normalization parameters"""

# INITIALIZE MODEL:
# agent = Agent(envs).to(device)
# rnd_model = RNDModel(4, envs.single_action_space.n).to(device)
# combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
# optimizer = optim.Adam(
#     combined_parameters,
#     lr=args.learning_rate,
#     eps=1e-5,
# )

# INTIIALIZE STORAGE: 
# reward_rms = RunningMeanStd()
# discounted_reward = RewardForwardFilter(args.int_gamma)
# obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
# actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
# logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
# rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
# curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

# COMPUTE CURIOSITY REWARD:
# next_obs, reward, done, info = envs.step(action.cpu().numpy())
# rewards[step] = torch.tensor(reward).to(device).view(-1)
# next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
# rnd_next_obs = (
#     (
#         (next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
#         / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
#     ).clip(-5, 5)
# ).float()
# target_next_feature = rnd_model.target(rnd_next_obs)
# predict_next_feature = rnd_model.predictor(rnd_next_obs)
# curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data


# AFTER DATA COLLECTION: reward normalization:
# curiosity_reward_per_env = np.array(
#     [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
# )
# mean, std, count = (
#     np.mean(curiosity_reward_per_env),
#     np.std(curiosity_reward_per_env),
#     len(curiosity_reward_per_env),
# )
# reward_rms.update_from_moments(mean, std**2, count)

# curiosity_rewards /= np.sqrt(reward_rms.var)


# USE INTRINSIC REWARDS: target computation
# with torch.no_grad():
    # ext_advantages = torch.zeros_like(rewards, device=device)
    # int_advantages = torch.zeros_like(curiosity_rewards, device=device)

    # ext_lastgaelam = 0
    # int_lastgaelam = 0
    # for t in reversed(range(args.num_steps)):
    #     if t == args.num_steps - 1:
    #         ext_nextnonterminal = 1.0 - next_done
    #         int_nextnonterminal = 1.0
    #         ext_nextvalues = next_value_ext
    #         int_nextvalues = next_value_int
    #     else:
    #         ext_nextnonterminal = 1.0 - dones[t + 1]
    #         int_nextnonterminal = 1.0
    #         ext_nextvalues = ext_values[t + 1]
    #         int_nextvalues = int_values[t + 1]
    #     ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
    #     int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
    #     ext_advantages[t] = ext_lastgaelam = (
    #         ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
    #     )
    #     int_advantages[t] = int_lastgaelam = (
    #         int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
    #     )
    # ext_returns = ext_advantages + ext_values
    # int_returns = int_advantages + int_values

# # flatten the batch
# b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
# b_logprobs = logprobs.reshape(-1)
# b_actions = actions.reshape(-1)
# b_ext_advantages = ext_advantages.reshape(-1)
# b_int_advantages = int_advantages.reshape(-1)
# b_ext_returns = ext_returns.reshape(-1)
# b_int_returns = int_returns.reshape(-1)
# b_ext_values = ext_values.reshape(-1)

# b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

# obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

# # Optimizing the policy and value network
# b_inds = np.arange(args.batch_size)

# rnd_next_obs = (
#     (
#         (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
#         / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
#     ).clip(-5, 5)
# ).float()


# UPDATE RND MODEL:
# for epoch in range(args.update_epochs):
#     np.random.shuffle(b_inds)
#     for start in range(0, args.batch_size, args.minibatch_size):
#         end = start + args.minibatch_size
#         mb_inds = b_inds[start:end]

#         predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
#         forward_loss = F.mse_loss(
#             predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
#         ).mean(-1)

#         mask = torch.rand(len(forward_loss), device=device)
#         mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
#         forward_loss = (forward_loss * mask).sum() / torch.max(
#             mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
#         )
#        ... 
#        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


import numpy as np


# NOTE taken from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/running_mean_std.py
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # NOTE: changed architecure of netowrks because in Hockey, the input is not an image => no convolutional layers needed, just fully connected layers
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size), 
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size), 
        )
        
        # feature_output = 7 * 7 * 64

        # # Prediction network
        # self.predictor = nn.Sequential(
        #     layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(feature_output, 512)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(512, 512)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(512, 512)),
        # )

        # # Target network
        # self.target = nn.Sequential(
        #     layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
        #     nn.LeakyReLU(),
        #     layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
        #     nn.LeakyReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(feature_output, 512)),
        # )

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