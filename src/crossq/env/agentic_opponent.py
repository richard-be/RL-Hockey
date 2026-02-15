from typing import Protocol
import hockey.hockey_env as h_env

import numpy as np
import torch

import random
import gymnasium as gym

from models.actor import GaussianPolicy
from copy import deepcopy


class AgenticOpponent(Protocol):

    def act(self, obs):
        ...


class CrossQOpponent:
    def __init__(self, actor: GaussianPolicy, device: str = "cpu"):
        self.actor = actor
        self.device = device

    def act(self, obs):
        with torch.no_grad():
            _, _, action = self.actor.get_action(torch.from_numpy(obs).to(torch.float32).to(self.device).unsqueeze(0))
            return action.squeeze(0).detach().cpu().numpy()
    

def construct_crossq_opponent(policy: GaussianPolicy, device: str = "cpu"):
    policy = deepcopy(policy)
    policy.eval()
    return CrossQOpponent(actor=policy, device=device)
    

class OpponentPool:
    def __init__(self, latest_agent: AgenticOpponent | None, 
                 latest_agent_score: float,
                 window_size: int,
                 play_against_latest_model_ratio: float):
                 
        self.window_size = window_size
        self.play_against_latest_model_ratio = play_against_latest_model_ratio
        self.opponent_pool: list[AgenticOpponent] = list([latest_agent] if latest_agent else [])
        self.score_pool: list[float] = list([latest_agent_score])
        self.current_agent_idx = -1
        
    def add_agent(self, agent:AgenticOpponent, score: float) -> None:
        if len(self.opponent_pool) == self.window_size:
            self.opponent_pool.pop(0)
            self.score_pool.pop(0)
        self.score_pool.append(score)
        self.opponent_pool.append(agent)
        if self.current_agent_idx:
            self.current_agent_idx -= 1

    def update_opponent_score(self, new_score) -> None:
        self.score_pool[self.current_agent_idx] = new_score

    def get_opponent_score(self) -> float:
        return self.score_pool[self.current_agent_idx]

    def sample_opponent(self) -> AgenticOpponent:
        if random.random() <= self.play_against_latest_model_ratio:
            self.current_agent_idx = len(self.opponent_pool) - 1
            return self.opponent_pool[self.current_agent_idx]  
        else:
            # weighted choice by elo scores
            scores = np.array(self.score_pool)
            scores = scores / scores.sum()

            idx = np.random.choice(list(range(len(self.opponent_pool))), p=scores).item()
            self.current_agent_idx = idx
            return self.opponent_pool[idx]
            

class HockeyEnv_AgenticOpponent(h_env.HockeyEnv):
    def __init__(self, opponent: AgenticOpponent, 
                 mode=h_env.Mode.NORMAL):
        super().__init__(mode=mode, keep_mode=True)
        self.opponent = opponent

    def step(self, action):
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])
        return super().step(action2)


class HockeyEnv_SelfPlay(h_env.HockeyEnv):
    def __init__(self, opponent_pool: OpponentPool, 
                 mode=h_env.Mode.NORMAL,
                 swap_steps: int = 30_000,
                 ):
        super().__init__(mode=mode, keep_mode=True)
        self.pool = opponent_pool
        self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)
        self.swap_steps = swap_steps
        self.num_steps = 0

    # def add_agent(self, agent: AgenticOpponent, score: float | None = None) -> None:
    #     self.pool.add_agent(agent, score=score if score is not None else self.default_score)

    def _swap_agent(self) -> None:
        self.opponent = self.pool.sample_opponent()

    def get_opponent_score(self) -> float:
        return self.pool.get_opponent_score()

    def update_opponent_score(self, new_score: float) -> None:
        self.pool.update_opponent_score(new_score=new_score)

    def step(self, action):
        if self.num_steps % self.swap_steps == 0:
            self._swap_agent()
            
        self.num_steps += 1
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])

        return super().step(action2)
    

