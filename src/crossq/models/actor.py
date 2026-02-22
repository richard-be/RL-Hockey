from models.feedforward import FeedForward, NNConfig
import math

import torch
from torch import nn
import numpy as np
from torch import distributions

from dataclasses import dataclass, field

@dataclass
class GaussianPolicyConfig(NNConfig):
    action_dim: int = 1
    num_heads: int = field(init=False, default=2)
    output_dim: int | list[int] = field(init=False)
    output_act: None | nn.Module | list[None | nn.Module] = field(init=False, default_factory=lambda: [None, nn.Tanh()])


    def __post_init__(self):
        self.output_dim = [self.action_dim, self.action_dim]



class GaussianPolicy(FeedForward):

    
    def __init__(self, min_action, max_action, config: GaussianPolicyConfig,
                max_log_std: int = 2, min_log_std: int = -4,
                *args, **kwargs):
        super().__init__(config=config,
                          *args, **kwargs)

        # We need this parameters for enforcing action bounds
        # after computing unbounded sample actions we squash it with tanh
        # to be between [-1, 1]. squashed action's scale is 2 and we need to adjust it to action's scale
        # 
        self.register_buffer("action_scale", 
                             (max_action - min_action) / 2.0)  
        
        self.register_buffer("action_offset",
                            (max_action + min_action) / 2.0)
        
        self.log_std_scale = (max_log_std - min_log_std) / 2.0
        self.log_std_offset = (max_log_std + min_log_std) / 2.0
 

    def get_action(self, observations):
        mean, squashed_log_std = self.forward(observations)
        log_std = self.log_std_scale * squashed_log_std + self.log_std_offset


        dist = distributions.Normal(loc=mean, scale=torch.exp(log_std))
        unbounded_actions = dist.rsample()  # reparametrization trick
        # Here we are enforcing action bounds
        squashed_actions = nn.Tanh()(unbounded_actions) 
        actions = self.action_offset + self.action_scale * squashed_actions

        optimal_action = nn.Tanh()(mean) * self.action_scale + self.action_offset
        # prob_act = prob_unb * det(d_act/d_unb)^-1
        # act = offset + scale * tanh(unb) => d_act/d_unb  = diag[ (1 - tanh(unb)^2) * scale ] (because tanh is an element-wise operator)
        # prob_act = prob_unb * product[ (1 - tanh(unb_i)^2) * scale  ]^-1
        # log_prob_act = log_prob_unb  - sum[ log( (1 - tanh(unb_i)^2) * scale )  ]
        # print(dist.log_prob(unbounded_actions).shape, torch.sum(torch.log((1 - squashed_actions ** 2) * self.action_scale), dim=1).shape)
        log_prob =  torch.sum(dist.log_prob(unbounded_actions) - torch.log((1 - squashed_actions ** 2) * self.action_scale + 1e-5),
                              dim=1, keepdim=True)  # adding 1e-5 for stability. unbounded action log prob are inside of the sum only because for some reason pytorch
                                                    # returns dimension-wise factored probabilities 
        return actions, log_prob, optimal_action

    def act(self, observations: np.array) -> np.array:
        with torch.no_grad():
            self.eval()
            action, _, _ = self.get_action(torch.from_numpy(observations).to(torch.float32).unsqueeze(0))
            self.train()
            return action.squeeze(0).detach().cpu().numpy()
