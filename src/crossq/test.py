import os
import sys

from pathlib import Path


curr = Path(os.path.dirname(__file__))

sys.path.append(str(curr.parent.absolute()))

import hydra

from hydra.core.config_store import ConfigStore

import torch

import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
from gymnasium.wrappers import RecordEpisodeStatistics

from dataclasses import dataclass, field
from functools import partial



from env.agentic_opponent import HockeyEnv_SelfPlay, HockeyEnv_AgenticOpponent, AgenticOpponent, construct_crossq_opponent, OpponentPool




@dataclass
class EvalConfig:
    player_path: str
    opponent_path: str | None = None
    env_id: str = "HalfCheetah-v5"
    mode: h_env.Mode = h_env.Mode.NORMAL
    
    render_mode: str = "human"
    opponent_type: str = "custom"  # oponnet type ['basic', 'custom']
    
    # for basic opponent
    weak_mode: bool = False
    num_games: int = 100



def load_actor(run_name: str, env, device="cpu"):
    from td3.algorithm.td3 import Actor as TD3Actor
    from sac.agent.sac import Actor as SACActor
    from models.actor import GaussianPolicy as CrossQActor
    from models.actor import GaussianPolicyConfig as CrossQActorConfig

    def add_act_method(actor):
        def act(self, obs): 
            with torch.no_grad(): 
                obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
                action = self.forward(obs)
                if isinstance(action, tuple):
                    action = action[0]
                return action.cpu().numpy()[0]
        if not hasattr(actor, "act"):
            setattr(actor, "act", act.__get__(actor, actor.__class__))

    assert ":" in run_name, "Weight path must be in format <algorithm>:<path>"
    algorithm, path = run_name.split(":", 1)
    
    actor_types = {
        "sac": SACActor,
        "td3": TD3Actor,
        "crq": CrossQActor, 
    }

    assert algorithm in actor_types, f"Unknown algorithm {algorithm}. Supported algorithms: {list(actor_types.keys())}"
    actor_class = actor_types[algorithm]
    if actor_class is None:
        raise ValueError(f"Unknown external model type: {algorithm}")
    

    actor_state = torch.load(path, map_location=device)
    # TODO: this could be done better by creating different load functions for each model 
    if (isinstance(actor_state, tuple) or isinstance(actor_state, list)) and len(actor_state) == 3:   
        # if it's a td3 model, just load the actor state
        actor_state = actor_state[0] 

    print(f"Loaded model from {path}")
    
    if not hasattr(env, "single_observation_space"):
        env.single_observation_space = env.observation_space
    if not hasattr(env, "single_action_space"):
        env.single_action_space = env.action_space

    if algorithm == "crq":
        config = CrossQActorConfig(input_dim=np.prod(env.single_observation_space.shape),
                          action_dim=np.prod(env.single_action_space.shape))
        actor = actor_class(min_action=torch.from_numpy(env.single_action_space.low), 
                            max_action=torch.from_numpy(env.single_action_space.high), 
                            config=config)
    else:
        actor = actor_class(env)
    actor.load_state_dict(actor_state)
    add_act_method(actor)

    return actor.to(device)

def create_environment(env_config: EvalConfig, custom_opponent: None | AgenticOpponent = None) -> gym.Env:
    if env_config.mode == h_env.Mode.NORMAL:
        if env_config.opponent_type == 'custom':
            env = HockeyEnv_AgenticOpponent(opponent=custom_opponent)
        else:
            env = h_env.HockeyEnv_BasicOpponent(weak_opponent=env_config.weak_mode)
    else:
        env = h_env.HockeyEnv(mode=env_config.mode)
    if env_config.render_mode:
            env.render = partial(env.render, mode=env_config.render_mode)   
            env.render_mode = env_config.render_mode
    env = RecordEpisodeStatistics(env)
    return env


cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="eval_config", node=EvalConfig)

@hydra.main(version_base=None, config_name="eval_config")
def eval(config: EvalConfig):
    print("here")
    wins = 0
    env = create_environment(config)

    agent = load_actor(config.player_path, env)
    agent = construct_crossq_opponent(agent)

    if config.opponent_path:
        print("HERE")
        opponent = load_actor(config.opponent_path, env)
        env.unwrapped.opponent = opponent
    for _ in range(config.num_games):
        observation, info = env.reset()
        env.observation_space.dtype = np.float32
        while True:
            # env.render()
            action = agent.act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                winner = info['winner']
                if winner == 1:
                    wins += 1
                break

    print(wins / config.num_games)
    env.close()


if __name__ == "__main__":
    eval()