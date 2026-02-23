import torch.nn as nn
import numpy as np 
import torch
import torch.nn.functional as F
import copy 

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # allow either vectorized env or single env as argument
        obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
        act_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space

        self.fc1 = nn.Linear(np.array(obs_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(act_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (act_space.high - act_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (act_space.high + act_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


    # NOTE: added here to support act() for self-play: 
    def act(self, obs): 
        with torch.no_grad(): 
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
            action = self.forward(obs)
            return action.cpu().numpy()[0]

    # NOTE: added here to support self-play:
    def clone(self):
        clone = copy.deepcopy(self)
        clone.eval()
        for p in clone.parameters():
            p.requires_grad = False
        return clone
