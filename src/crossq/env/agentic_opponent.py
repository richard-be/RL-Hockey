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
            self.actor.eval()
            _, _, action = self.actor.get_action(torch.from_numpy(obs).to(torch.float32).to(self.device).unsqueeze(0))
            self.actor.train()
            return action.squeeze(0).detach().cpu().numpy()
    

def construct_crossq_opponent(policy: GaussianPolicy, device: str = "cpu", copy: bool = True):
    if copy:
        policy = deepcopy(policy)
        policy.eval()
    return CrossQOpponent(actor=policy, device=device)
    

class OpponentPool:
    def __init__(self,
                 window_size: int,
                 play_against_latest_model_ratio: float,
                 current_policy: AgenticOpponent | None = None,
                 fixed_agents: list[tuple[AgenticOpponent, float]] | None = None):
                 
        self.window_size = window_size
        self.play_against_latest_model_ratio = play_against_latest_model_ratio
        self.fixed_agent_buffer_size = len(fixed_agents)
        self.opponent_pool: dict[str, AgenticOpponent] = dict()
        self.score_pool: dict[str, float] = dict()
        self.current_policy = current_policy
        if fixed_agents:
            for idx, (agent, score) in enumerate(fixed_agents):
                self.opponent_pool[f"fixed_{idx}"] = agent
                self.score_pool[f"fixed_{idx}"] = score
        self.agent_idx = 0
        
    def add_agent(self, agent:AgenticOpponent, score: float) -> None:
        self.opponent_pool[f"agent_{self.agent_idx}"] = agent
        self.score_pool[f"agent_{self.agent_idx}"] = score
        self.agent_idx += 1

    def update_opponent_score(self, new_score: float, agent: str) -> None:
        self.score_pool[agent] = new_score

    def get_opponent_score(self, agent: str) -> float | None:
        return self.score_pool[agent] if agent != 'current' else None
    
    def sample_fixed_agent(self) -> tuple[AgenticOpponent, str]:
        agent = np.random.choice([agent_name for agent_name in self.opponent_pool.keys() if "fixed" in agent_name]).item()
        return self.opponent_pool[agent], agent

    def sample_opponent(self) -> tuple[AgenticOpponent, str]:
        if random.random() <= self.play_against_latest_model_ratio:
            return self.current_policy, "current"
        else:
            ids = [f"fixed_{idx}" for idx in range(self.fixed_agent_buffer_size)] + [f"agent_{idx}" for idx in range(max(self.agent_idx - self.window_size, 0), self.agent_idx)]
            agent = np.random.choice(ids).item()
            return self.opponent_pool[agent], agent
            

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
                 warmup_period: int = 50_000
                 ):
        self.opponent_idx = "current"
        super().__init__(mode=mode, keep_mode=True)
        self.pool = opponent_pool
        self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)
        self.swap_steps = swap_steps
        self.warmup_period = warmup_period
        self.num_steps = 0
        

    # def add_agent(self, agent: AgenticOpponent, score: float | None = None) -> None:
    #     self.pool.add_agent(agent, score=score if score is not None else self.default_score)

    def _swap_agent(self) -> None:
        self.opponent, self.opponent_idx = self.pool.sample_opponent()

    def _sample_fixed_opponent(self) -> None:
        self.opponent, self.opponent_idx = self.pool.sample_fixed_agent()

    def get_opponent_score(self) -> float | None:
        return self.pool.get_opponent_score(agent=self.opponent_idx)

    def update_opponent_score(self, new_score: float) -> None:
        self.pool.update_opponent_score(new_score=new_score, agent=self.opponent_idx)

    def _get_info(self):
        info =  super()._get_info()
        info['opponent'] = self.opponent_idx
        return info

    def step(self, action):
        if self.num_steps % self.swap_steps == 0:
            if self.num_steps <= self.warmup_period:
                self._sample_fixed_opponent()
            else:
                self._swap_agent()

            self._get_info

        self.num_steps += 1
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])

        return super().step(action2)
    

