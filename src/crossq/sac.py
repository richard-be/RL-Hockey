import torch
from torch import nn
from torch import optim
import torchrl.modules
import numpy as np
import gymnasium as gym

from critic import QNetwork
from actor import GaussianPolicy

from memory import Memory


from dataclasses import dataclass, asdict, field


def compute_sac_critic_loss(q_functions: list[QNetwork], target_q_functions: list[QNetwork],
                             policy: GaussianPolicy, alpha: float, gamma: float,
                        observations: torch.Tensor, 
                        actions: torch.Tensor, 
                        rewards: torch.Tensor, next_observations: torch.Tensor, finished: torch.Tensor) -> torch.Tensor:
        
    with torch.no_grad():
        next_actions, log_prob, _ = policy.get_action(next_observations)
        next_q = torch.stack([target.q_value(next_observations, next_actions) for target in target_q_functions]).squeeze(-1)
        next_q = torch.min(next_q, dim=0).values
        next_q = next_q - alpha * log_prob.squeeze(-1)
        target = rewards + gamma * next_q * (1 - finished)


    # Compute ensemble q values
    q = torch.stack([q_func.q_value(observations, actions) for q_func in q_functions]).squeeze(-1)
        

    return ((q - target) ** 2).mean()
    


def compute_actor_loss(q_functions: list[QNetwork], policy: GaussianPolicy, alpha: float,
                        observations: torch.Tensor, 
                       ) -> torch.Tensor:
    
    actions, log_prob, _ = policy.get_action(observations)

    qs = torch.stack([q_func.q_value(observations, actions) for q_func in q_functions])

    q_value = torch.min(qs, dim=0).values


    return -(q_value - alpha * log_prob).mean(), -log_prob.mean(), qs.mean(dim=1)


def compute_alpha_loss(policy: GaussianPolicy, log_alpha: nn.Parameter, target_entropy: torch.Tensor, 
                       observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, log_prob, _ = policy.get_action(observations)  # [Bs, 1]
    alpha = torch.exp(log_alpha)


    return -(alpha * (log_prob + target_entropy)).mean()

@dataclass
class SACConfig:
    env_id: str = "Pendulum-v1"

    discount_factor: float = .99
    alpha: float = .3
    dynamic_alpha: bool = True  # Idea from Soft Actor-Critic Algorithms and Applications paper
    alpha_lr: float = 1e-3

    q_ensamble: int = 2

    target_update: int = 5
    tau: float = .995

    actor_lr: float = 1e-4
    q_lr: float = 1e-3
    batch_size: int = 256
    device: str = "cpu"
    adam_beta1: float = .9
    adam_beta2: float = .999
    train_steps: int = 100_000
    policy_delay: int = 3

    use_tensorboard: bool = True

    buffer_size: int = 100_000

    seed: int = 42


    
from torch.utils.tensorboard import SummaryWriter


class SACgent:

    def __init__(self, 
                 config: SACConfig):
        
        self.config = config
        
        self.env = gym.make(self.config.env_id)
        
        obs_dim = np.prod(self.env.observation_space.shape)
        action_dim = np.prod(self.env.action_space.shape)
        self.q_functions = [ QNetwork(observation_dim=obs_dim, action_dim=action_dim, hidden_dim=256).to(self.config.device) for _ in range(self.config.q_ensamble)]
        self.q_targets = [QNetwork(observation_dim=obs_dim, action_dim=action_dim, hidden_dim=256).to(self.config.device) for _ in range(self.config.q_ensamble)]

        for q_func, q_target in zip(self.q_functions, self.q_targets):
            q_target.load_state_dict(q_func.state_dict())

        self.policy = GaussianPolicy(min_action=torch.from_numpy(self.env.action_space.low), max_action=torch.from_numpy(self.env.action_space.high), 
                                     observation_dim=obs_dim, action_dim=action_dim, hidden_dim=256).to(self.config.device)
        

        # self.policy.to(self.config.device)
        # for q_func in self.q_functions:
        #     q_func.to(self.config.device)
        
        self.buffer = Memory(max_size=self.config.buffer_size)


        self.q_optimizer = optim.Adam([param for q_function in self.q_functions for param in q_function.parameters()], lr=self.config.q_lr
                                      , betas=[self.config.adam_beta1, self.config.adam_beta2]
                                      )

        
        self.policy_optimizer = optim.Adam(list(self.policy.parameters()), lr=self.config.actor_lr
                                           , betas=[self.config.adam_beta1, self.config.adam_beta2]
                                           )

        

        self.writer = SummaryWriter()

        if self.config.dynamic_alpha:
            self.entropy_target = -torch.tensor(action_dim, dtype=torch.float32).to(self.config.device)
            print(self.entropy_target)
            self.log_alpha = nn.Parameter(torch.zeros(1).to(self.config.device))  # we want alpha's value to be positive
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

        
        

    def store_transition(self, obs: np.array, act: np.array, next_obs: np.array,
                          reward: np.array, is_terminal: np.array) -> None:
        self.buffer.add_transition([obs, act, next_obs, reward, is_terminal])

    
    def sample_data(self, n_samples: int = 32):
        observation, action, next_observation, reward, is_terminal = self.buffer.sample(n_samples).T

        action = torch.from_numpy(np.stack(action)).to(torch.float32).to(self.config.device)
        next_observation = torch.from_numpy(np.stack(next_observation)).to(torch.float32).to(self.config.device)
        observation = torch.from_numpy(np.stack(observation)).to(torch.float32).to(self.config.device)
        is_terminal = torch.from_numpy(np.stack(is_terminal)).to(torch.float32).to(self.config.device)
        reward = torch.from_numpy(np.stack(reward)).to(torch.float32).to(self.config.device)

        return observation, action, next_observation, reward, is_terminal

    
    def act(self, observation: np.array, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            sample_action, _, det_action = self.policy.get_action(torch.from_numpy(observation).to(self.config.device).unsqueeze(0))
            return det_action.squeeze(0) if deterministic else sample_action.squeeze(0)
        
    
    def state(self):
        return [self.policy.state_dict()] + [q_func.state_dict() for q_func in self.q_functions] 
    
    def update_target(self):
        for q_func, target_q_func in zip(self.q_functions, self.q_targets):
            for q_param, target_param in zip(q_func.parameters(), target_q_func.parameters()):
                target_param.data.copy_(target_param.data * self.config.tau  + (1 - self.config.tau) * q_param.data)
    

    def fit(self, global_step: int) -> None:
        observation, action, next_observation, reward, is_terminal = self.sample_data(self.config.batch_size)

        if self.config.dynamic_alpha:
            alpha_loss = compute_alpha_loss(self.policy, self.log_alpha, self.entropy_target, observation)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha = torch.exp(self.log_alpha).item()

            self.writer.add_scalar("Value/Alpha", alpha, global_step)

            
        else:
            alpha = self.config.alpha

        critic_loss = compute_sac_critic_loss(self.q_functions, self.q_targets, self.policy, alpha, self.config.discount_factor,
                                           observation, action, reward, next_observation, is_terminal)
        
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        if self.config.use_tensorboard:
            self.writer.add_scalar("Loss/Q_Loss",  critic_loss.item(), global_step)

        

        if global_step % self.config.policy_delay == 0:
            for _  in range(self.config.policy_delay):
                actor_loss, entropy, values = compute_actor_loss(self.q_functions, self.policy, alpha, observation)
            
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                self.policy_optimizer.step()

            if self.config.use_tensorboard:
                self.writer.add_scalar("Loss/Actor_Loss",  actor_loss.item(), global_step)   
                self.writer.add_scalar("Loss/Entropy", entropy.item(), global_step) 
                for idx, val in enumerate(values):
                    self.writer.add_scalar(f"Value/Q{idx+1}", val, global_step)
        
            
  
    def train(self) -> None:
        observation, info = self.env.reset(seed=self.config.seed)
        episodic_return = 0

        if self.config.use_tensorboard:
            self.writer.add_hparams(asdict(self.config), dict())

        for step in range(self.config.train_steps):
            action = self.act(observation).cpu().numpy()

            next_observation, reward, terminated, truncated, info = self.env.step(action)


            episodic_return += reward

            
            self.store_transition(observation, action, next_observation, reward, terminated)

            observation = next_observation

            if terminated or truncated:
                observation, info = self.env.reset()

                if self.config.use_tensorboard:
                    self.writer.add_scalar("Env/Return", episodic_return, step)
                
                episodic_return = 0

            
            if step >= self.config.batch_size:  # wait till we have enough data for the full mini-batch
                self.fit(global_step=step)
                self.update_target()





