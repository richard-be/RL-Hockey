import torch
from torch import nn
from torch import optim
import numpy as np



from models.critic import QNetwork
from models.actor import GaussianPolicy, GaussianPolicyConfig
from models.feedforward import NNConfig, NormalizationConfig, get_gradient_norm

from utils.elr import parameter_snapshot, measaure_effecitve_learning_rate
from utils.hooks import register_relu_hooks, compute_dead_relu_metrics

from memory import Memory


from dataclasses import dataclass


def compute_critic_loss(q_functions: list[QNetwork], policy: GaussianPolicy, alpha: float, gamma: float,
                        observations: torch.Tensor, 
                        actions: torch.Tensor, 
                        rewards: torch.Tensor, next_observations: torch.Tensor, finished: torch.Tensor, 
                        q_targets: list[QNetwork] | None = None) -> torch.Tensor:
        
    with torch.no_grad():
        policy.eval()
        next_actions, log_prob, _ = policy.get_action(next_observations)  # (Batch_size, 1), (Batch_size, 1)
        policy.train()

    # Compute ensemble q values (#Q_functions, 2 * Batch_size, 1)
    qs = torch.stack([q_func.q_value(torch.concat([observations, next_observations]), torch.concat([actions, next_actions])) for q_func in q_functions])

    # split into current and next q values (#Q_functions, 2 * Batch_size, 1) -> (#Q_functions, Batch_size, 1), (#Q_functions, Batch_size, 1)
    q, next_q = torch.split(qs, qs.shape[1] // 2, dim=1)

    # We also want to compute joint forward pass for target network but only keep next_q
    with torch.no_grad():
        if q_targets:
            qs_targets = torch.stack([q_func.q_value(torch.concat([observations, next_observations]), torch.concat([actions, next_actions])) for q_func in q_targets])
            _, next_q = torch.split(qs_targets, qs_targets.shape[1] // 2, dim=1)


    
        next_q = torch.min(next_q, dim=0).values  # (Batch_size, 1)
        next_q = next_q - alpha * log_prob
        next_q = next_q.squeeze(-1)  # (Batch_size, 1) -> (Batch_size, ) 
        target = rewards + gamma * next_q * (1 - finished)  #  rewards and finished have (batch_size) shapes

    q = q.squeeze(-1)  # (Batch_size, 1) -> (Batch_size)        

    return ((q - target) ** 2).mean()
    


def compute_actor_loss(q_functions: list[QNetwork], policy: GaussianPolicy, alpha: float,
                        observations: torch.Tensor, 
                       ) -> torch.Tensor:
    
    actions, log_prob, _ = policy.get_action(observations)


    #  Use stored batch norm statistics
    for q_func in q_functions:
        q_func.eval()

    qs = torch.stack([q_func.q_value(observations, actions) for q_func in q_functions])  # (#Q_functions, Batch Size, 1)

    for q_func in q_functions:
        q_func.train()

    q_value = torch.min(qs, dim=0).values  # (#Q_functions, Batch size, 1) -> (Batch Size, 1)


    return -(q_value - alpha * log_prob).mean(), -log_prob.mean(), qs.mean(dim=1)


def compute_alpha_loss(policy: GaussianPolicy, log_alpha: nn.Parameter, target_entropy: torch.Tensor, 
                       observations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        policy.eval()
        _, log_prob, _ = policy.get_action(observations)  # [Bs, 1]
        policy.train()

    alpha = torch.exp(log_alpha)


    return -(alpha * (log_prob + target_entropy)).mean()




@dataclass
class CrossQAgentConfig:
    discount_factor: float = .99
    alpha: float = .3
    dynamic_alpha: bool = True  # Idea from Soft Actor-Critic Algorithms and Applications paper
    alpha_lr: float = 1e-4

    q_ensamble: int = 2

    actor_lr: float = 1e-3
    q_lr: float = 1e-3

    batch_size: int = 256
    device: str = "cuda"

    adam_beta1: float = .9
    adam_beta2: float = .999

    policy_delay: int = 3

    q_hidden_dim: int = 512
    policy_hidden_dim: int = 256
    batch_norm_type: str = "BRN"
    batch_norm_warmup: int = 100_000
    batchn_norm_momentum: float = .01

    buffer_size: int = 1_000_000

    target: bool = False  # Target network in CrossQ reintroduced by https://doi.org/10.48550/arXiv.2502.07523
    tau: float = 5e-3  # Polyak tau for target updates

    utd: int = 1

    weight_norm: bool = False
    weight_decay: float = 1e-2

    clip_grad: bool = False


class CrossQAgent:

    def __init__(self, 
                 config: CrossQAgentConfig, obs_dim: int, action_dim: int,
                   action_low: np.array, action_high: np.array):
        
        self.config = config
        

        q_config = NNConfig(input_dim=obs_dim + action_dim, hidden_dim=self.config.q_hidden_dim, output_dim=1, 
                            normalization_config=NormalizationConfig(type=self.config.batch_norm_type, 
                                                                     momentum=self.config.batchn_norm_momentum, 
                                                                     warmup_steps=self.config.batch_norm_warmup), weight_norm=self.config.weight_norm)
        

        
        policy_config = GaussianPolicyConfig(input_dim=obs_dim, hidden_dim=self.config.policy_hidden_dim, action_dim=action_dim,
                                             normalization_config=NormalizationConfig(type=self.config.batch_norm_type, 
                                                                     momentum=self.config.batchn_norm_momentum, 
                                                                     warmup_steps=self.config.batch_norm_warmup))

        self.q_functions = [ QNetwork(config=q_config).to(self.config.device) for _ in range(self.config.q_ensamble)]


        self.activations = [register_relu_hooks(q_func) for q_func in self.q_functions]


        if config.target:
            self.q_target_functions = [ QNetwork(config=q_config).to(self.config.device) for _ in range(self.config.q_ensamble)]

            # initialize weights
            for q_func, q_target in zip(self.q_functions, self.q_target_functions):
                q_target.load_state_dict(q_func.state_dict())

        self.policy = GaussianPolicy(min_action=torch.from_numpy(action_low), max_action=torch.from_numpy(action_high), 
                                     config=policy_config).to(self.config.device)
        

        
        self.buffer = Memory(max_size=self.config.buffer_size)

        if config.weight_norm:
            non_weight_decay_params = [param for q_function in self.q_functions for name, param in q_function.named_parameters() if not ("output_layers" in name and "weight" in name)]
            weight_decay_params = [param for q_function in self.q_functions for name, param in q_function.output_layers.named_parameters() if "weight" in name]
            self.q_optimizer = optim.AdamW([{"params": non_weight_decay_params, "weight_decay": 0},
                                            {"params": weight_decay_params}], lr=self.config.q_lr
                                        , betas=[self.config.adam_beta1, self.config.adam_beta2], weight_decay=config.weight_decay
                                        )

        else:
            self.q_optimizer = optim.Adam([param for q_function in self.q_functions for param in q_function.parameters()], lr=self.config.q_lr
                                        , betas=[self.config.adam_beta1, self.config.adam_beta2]
                                        )

        
        self.policy_optimizer = optim.Adam(list(self.policy.parameters()), lr=self.config.actor_lr
                                           , betas=[self.config.adam_beta1, self.config.adam_beta2]
                                           )

        

        if self.config.dynamic_alpha:
            self.entropy_target = -torch.tensor(action_dim, dtype=torch.float32).to(self.config.device)
            print(self.entropy_target)
            self.log_alpha = nn.Parameter(torch.zeros(1).to(self.config.device))  # we want alpha's value to be positive
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

    @torch.no_grad()
    def update_target(self):
        for q_func, target_q_func in zip(self.q_functions, self.q_target_functions):
            for q_param, target_param in zip(q_func.parameters(), target_q_func.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.config.tau ) + self.config.tau * q_param.data)

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
            self.policy.eval()
            sample_action, _, det_action = self.policy.get_action(torch.from_numpy(observation).to(torch.float32).to(self.config.device).unsqueeze(0))
            self.policy.train()
            return det_action.squeeze(0) if deterministic else sample_action.squeeze(0)
        
    
    def state(self):
        return [self.policy.state_dict()] + [q_func.state_dict() for q_func in self.q_functions] 
    

    def load_state(self, state) -> None:
        policy_state, *q_staes = state
        self.policy.load_state_dict(policy_state)
        for q_state, q_func in zip(q_staes, self.q_functions):
            q_func.load_state_dict(q_state)
    
    # TODO: Write somewhere down that it is unstable without gradient clipping
    def learn(self, update_policy: bool) -> dict[str, float]:
        logs = {}
        for _ in range(self.config.utd):
            observation, action, next_observation, reward, is_terminal = self.sample_data(self.config.batch_size)

            

            if self.config.dynamic_alpha:
                alpha_loss = compute_alpha_loss(self.policy, self.log_alpha, self.entropy_target, observation)

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()

                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.log_alpha, 1.0)

                logs['alpha_grad_norm'] = self.log_alpha.grad.detach().norm(2).item()

                self.alpha_optimizer.step()

                alpha = torch.exp(self.log_alpha).item()

                logs['alpha_value'] = alpha
                
            else:
                alpha = self.config.alpha

            critic_loss = compute_critic_loss(self.q_functions, self.policy, alpha, self.config.discount_factor,
                                            observation, action, reward, next_observation, is_terminal, q_targets= self.q_target_functions if self.config.target else None)
            
            # snapshots = []
            # for q_func in self.q_functions:
            #     snapshots.append(parameter_snapshot(q_func, [layer_name for layer_name in dict(q_func.named_parameters()).keys() if 
            #                                             ("dense" in layer_name or "output_layers" in layer_name) and "weight" in layer_name]))
            self.q_optimizer.zero_grad()
            critic_loss.backward()

            if self.config.clip_grad:
                for q_func in self.q_functions:
                    torch.nn.utils.clip_grad_norm_(q_func.parameters(), 1.0)

            logs["critic_loss"] = critic_loss.item()
            logs["q_grad_norms"] = [get_gradient_norm(q_func) for q_func in self.q_functions]
            logs["q_weight_norms"] = [torch.sum(q_func.get_weight_norms()) for q_func in self.q_functions]


            self.q_optimizer.step()

            relu_stats_list = []
            elrs_list = []
            for idx, q_func in enumerate(self.q_functions):
                elrs = measaure_effecitve_learning_rate(q_func, names=[layer_name for layer_name in dict(q_func.named_parameters()).keys() if 
                                                        ("dense" in layer_name or "output_layers" in layer_name) and "weight" in layer_name])
                relu_stats = compute_dead_relu_metrics(self.activations[idx])
                relu_stats_list.append(relu_stats)
                elrs_list.append(elrs)
            logs[f"critic_relu_stats"] = relu_stats_list
            logs[f"critic_elrs"] = elrs_list

            if self.config.weight_norm:
                for q_func in self.q_functions:
                    q_func.normalize_weights_()

            
            if update_policy:
                actor_loss, entropy, values = compute_actor_loss(self.q_functions, self.policy, alpha, observation)
            
                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                if self.config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

                logs['actor_loss'] = actor_loss.item()
                logs['entropy'] = entropy.item()
                logs['actor_grad_norm'] = get_gradient_norm(self.policy)

                self.policy_optimizer.step()

            if self.config.target:
                    self.update_target()
        return logs



    