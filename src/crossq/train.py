
import hydra

from hydra.core.config_store import ConfigStore

from collections import defaultdict

import os
import random
import torch

import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


from torch.utils.tensorboard import SummaryWriter


from dataclasses import dataclass, field, asdict

from crossq import CrossQAgent, CrossQAgentConfig


@dataclass
class EnvConfig:
    env_id: str = "HalfCheetah-v5"
    mode: h_env.Mode = h_env.Mode.NORMAL
    basic_opponent: bool = True
    weak_mode: bool = False

@dataclass
class CrossQConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent_config: CrossQAgentConfig = field(default_factory=CrossQAgentConfig)
    
    train_steps: int = 1_000_000
    learning_starts: int = 10_000
    save_freq: int = 500_000

    use_tensorboard: bool = True
    

    seed: int = 42


def is_hockey(env_id: str) -> bool:
    return env_id.startswith("Hockey")


def get_opponent(env_config: EnvConfig):
    if env_config.basic_oponent:
        opponent = h_env.BasicOpponent(weak=env_config.weak_mode)
    else:
        # TODO agent loading for sel-play
        ...
    return opponent


def create_environment(env_config: EnvConfig) -> gym.Env:
    if is_hockey(env_config.env_id):
        if env_config.mode == h_env.Mode.NORMAL:
            env = h_env.HockeyEnv_BasicOpponent(weak_opponent=env_config.weak_mode)
        else:
            env = h_env.HockeyEnv(mode=env_config.mode)
    else:
        env = gym.make(env_config.env_id)
        
    env = RecordEpisodeStatistics(env)
    return env


# def eval(agent: CrossQAgent, 
#               env_config: EnvConfig):
#     env = create_environment(env_config)
#     env = RecordVideo(env)

#     observation, info = env.reset()
#     while True:
#         action = agent.act(observation).cpu().numpy()
#         observation, _, terminated, truncated, info = env.step(action)

#         if terminated or truncated:
#             break



cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=CrossQConfig)

@hydra.main(version_base=None, config_name="config")
def fit_cross_q(config: CrossQConfig):
    # set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    

    #initialize environment
    env = create_environment(config.env)

    agent = CrossQAgent(config=config.agent_config, 
                        obs_dim=np.prod(env.observation_space.shape),
                        action_dim=np.prod(env.action_space.shape),
                        action_low=env.action_space.low,
                        action_high=env.action_space.high)

    if config.use_tensorboard:
        writer = SummaryWriter(comment=f"CrossQ-{config.env}")


    observation, info = env.reset(seed=config.seed)
    num_episodes = 0
    running_stats = defaultdict(int)

    for _ in range(config.learning_starts):
        action = agent.act(observation).cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.store_transition(observation, action, next_observation, reward, terminated)

        observation = next_observation
        if truncated or terminated:
            observation, info = env.reset()
    
    for step in range(config.train_steps):
        action = agent.act(observation).cpu().numpy()
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        agent.store_transition(observation, action, next_observation, reward, terminated)
        observation = next_observation
        if truncated or terminated:
            num_episodes += 1
            if config.use_tensorboard:
                writer.add_scalar("Env/Return", info['episode']['r'], step)
                writer.add_scalar("Env/Length", info['episode']['l'], step)
                # print(is_hockey(config.env.env_id))
                if is_hockey(config.env.env_id):
                    winner = info['winner']
                    if winner == 0:
                        running_stats['draw'] += 1
                    elif winner == 1:
                        running_stats['win'] +=1
                    else:
                        running_stats['loss'] += 1
                    
                    writer.add_scalar("Env/WinRate", running_stats['win'] / num_episodes, step)
                    writer.add_scalar("Env/DrawRate", running_stats['draw'] / num_episodes, step)
                    writer.add_scalar("Env/LossRate", running_stats['loss'] / num_episodes, step)
                    writer.add_scalar("Env/WinLossRatio", running_stats['win'] / running_stats['loss'] if running_stats['loss'] != 0 else 0, step)

            observation, info = env.reset()
        
        
        agent.learn(global_step=step, writer=writer if config.use_tensorboard else None)

        if (step + 1) % config.save_freq == 0:
            os.makedirs(f"models/crossq/{step + 1}/", exist_ok=True)
            torch.save(agent.state(), f"models/crossq/{step + 1}/model.pkl") 

    if config.use_tensorboard:
        writer.add_hparams(asdict(config), dict())

    os.makedirs(f"models/crossq/final/", exist_ok=True)
    torch.save(agent.state(), f"models/crossq/final/model.pkl") 




if __name__ == "__main__":
    fit_cross_q()