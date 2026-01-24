import hydra

from hydra.core.config_store import ConfigStore

from collections import defaultdict

import os
import time

import random
import torch

import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field, asdict
from functools import partial

from crossq import CrossQAgent, CrossQAgentConfig
from env.agentic_opponent import HockeyEnv_SelfPlay, HockeyEnv_AgenticOpponent, AgenticOpponent, construct_crossq_opponent
from metrics.elo import update_elo_ratings


@dataclass
class EnvConfig:
    env_id: str = "HalfCheetah-v5"
    mode: h_env.Mode = h_env.Mode.NORMAL
    render_mode: str | None = None
    opponent_type: str = "selfplay"  # oponnet type ['basic', 'selfplay', 'custom']
    
    # for basic opponent
    weak_mode: bool = False

    # selfplay elements
    play_against_latest_model_ratio: float = .5
    window_size: int = 10
    swap_steps: int = 100_000
    opponent_save_steps: int = 50_000

    initial_elo: float = 12000.0
    k_factor: int = 16

@dataclass
class CrossQConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent_config: CrossQAgentConfig = field(default_factory=CrossQAgentConfig)
    
    train_steps: int = 1_500_000
    learning_starts: int = 10_000
    save_freq: int = 100_000
    log_freq: int = 100
    eval_freq: int = 50_000

    use_tensorboard: bool = True

    seed: int = 42


def is_hockey(env_id: str) -> bool:
    return env_id.startswith("Hockey")


def create_environment(env_config: EnvConfig, custom_opponent: None | AgenticOpponent = None  # for custom opponent and selfplay type env
                       ) -> gym.Env:
    if is_hockey(env_config.env_id):
        if env_config.mode == h_env.Mode.NORMAL:
            if env_config.opponent_type == 'selfplay' and custom_opponent:
                env = HockeyEnv_SelfPlay(agent=custom_opponent, 
                                         play_against_latest_model_ratio=env_config.play_against_latest_model_ratio,
                                         window_size=env_config.window_size,
                                         default_score=env_config.initial_elo)
            elif env_config.opponent_type == 'custom' and custom_opponent:
                env = HockeyEnv_AgenticOpponent(opponent=custom_opponent)
            else:
                env = h_env.HockeyEnv_BasicOpponent(weak_opponent=env_config.weak_mode)

        else:
            env = h_env.HockeyEnv(mode=env_config.mode)

        if env_config.render_mode:
                env.render = partial(env.render, mode=env_config.render_mode)   
                env.render_mode = env_config.render_mode
    else:
        env = gym.make(env_config.env_id, render_mode=env_config.render_mode)
        
    env = RecordEpisodeStatistics(env)
    return env


def eval(agent: CrossQAgent, 
              env_config: EnvConfig, writer: SummaryWriter,
              global_step: int,
              identifier: str):
    if is_hockey(env_config.env_id):
        weak_opponent_env = create_environment(EnvConfig(env_id=env_config.env_id,
                                      mode=h_env.Mode.NORMAL,
                                      opponent_type="basic",
                                      weak_mode=True,
                                      render_mode="rgb_array",
                                     ))
        strong_opponent_env = create_environment(EnvConfig(env_id=env_config.env_id,
                                      mode=h_env.Mode.NORMAL,
                                      opponent_type="basic",
                                      weak_mode=False,
                                      render_mode="rgb_array",
                                      ))
        # defense_env = create_environment(EnvConfig(env_id=env_config.env_id,
        #                               mode=h_env.Mode.TRAIN_DEFENSE,
        #                               render_mode="rgb_array",
        #                               ))
        # shooting_env = create_environment(EnvConfig(env_id=env_config.env_id,
        #                               mode=h_env.Mode.TRAIN_SHOOTING,
        #                               render_mode="rgb_array",
        #                               ))
        selfgame_env = create_environment(EnvConfig(env_id=env_config.env_id,
                                      mode=h_env.Mode.NORMAL,
                                      opponent_type="custom",
                                      render_mode="rgb_array",
                                      ), custom_opponent=construct_crossq_opponent(agent.policy, device=agent.config.device))
        
        
        
        envs = {"weak_opponent": RecordVideo(weak_opponent_env, episode_trigger=lambda _: True, video_folder=f"videos/crossq/{identifier}/{global_step}", 
                                             name_prefix="weak_opponent"), 
                "strong_opponent": RecordVideo(strong_opponent_env, episode_trigger=lambda _: True, video_folder=f"videos/crossq/{identifier}/{global_step}", 
                                               name_prefix="strong_opponent"),
                "self_opponent": RecordVideo(selfgame_env, episode_trigger=lambda _: True, video_folder=f"videos/crossq/{identifier}/{global_step}", 
                                             name_prefix="self_opponent")
               }
    else:
        env = create_environment(env_config)
        env = RecordVideo(env)
        envs = {"basic": env}
    
    
    for name, env in envs.items():
        observation, info = env.reset()
        while True:
            action = agent.act(observation).cpu().numpy()
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                writer.add_scalar(f"Eval/Return_{name}", info['episode']['r'], global_step)
                if is_hockey(env_config.env_id):
                    winner = info['winner']
                    writer.add_scalar(f"Eval/Winner_{name}", winner, global_step)
                env.close()
                break


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

    if config.env.opponent_type == "selfplay":
        env.close()
        elo_score = config.env.initial_elo
        env: HockeyEnv_SelfPlay = create_environment(config.env, custom_opponent=construct_crossq_opponent(agent.policy, device=config.agent_config.device))

    identifier = f"CrossQ-{config.env.opponent_type}-{config.env.env_id}-{config.seed}-{int(time.time())}"
    if config.use_tensorboard:
        
        writer = SummaryWriter(log_dir=f"runs/crossq/{identifier}")
        


    observation, info = env.reset(seed=config.seed)
    num_episodes = 0
    running_stats = defaultdict(int)

    for _ in range(config.learning_starts):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.store_transition(observation, action, next_observation, reward, terminated or truncated)

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
                    if config.env.opponent_type == 'selfplay':
                        elo_score_opponent = env.unwrapped.get_opponent_score()
                        elo_score, elo_score_opponent = update_elo_ratings(elo_score, elo_score_opponent, k_factor=config.env.k_factor, result=winner)
                        env.unwrapped.update_opponent_score(new_score=elo_score_opponent)
                        writer.add_scalar("Env/ELO_A", elo_score, step)
                        writer.add_scalar("Env/ELO_B", elo_score_opponent, step)

            observation, info = env.reset()
        
        update_policy = step % config.agent_config.policy_delay == 0
        logs = agent.learn(update_policy=update_policy)

        if config.agent_config.dynamic_alpha:
            alpha_grad_norm = logs['alpha_grad_norm']
            alpha_value = logs['alpha_value']

        if update_policy:
            actor_loss = logs['actor_loss']
            entropy = logs['entropy']
            actor_grad_norm = logs['actor_grad_norm']

        
        critic_loss = logs["critic_loss"] 
        q_grad_norms = logs["q_grad_norms"]


        if (step + 1) % config.log_freq == 0:
            if config.agent_config.dynamic_alpha:
                writer.add_scalar("Grad/Alpha", alpha_grad_norm, step)
                writer.add_scalar("Value/Alpha", alpha_value, step)


            writer.add_scalar("Loss/Q_Loss", critic_loss, step)
            for idx, q_norm in enumerate(q_grad_norms):
                writer.add_scalar(f"Grad/Q_{idx}", q_norm, step)


            writer.add_scalar("Loss/Actor_Loss",  actor_loss, step)   
            writer.add_scalar("Loss/Entropy", entropy, step) 
            writer.add_scalar("Grad/Actor",actor_grad_norm, step)

        if (step + 1) % config.save_freq == 0:
            os.makedirs(f"models/crossq/{step + 1}/", exist_ok=True)
            torch.save(agent.state(), f"models/crossq/{step + 1}/model.pkl") 

        if config.env.opponent_type == "selfplay":
            if (step + 1) % config.env.opponent_save_steps == 0:
                env.unwrapped.add_agent(construct_crossq_opponent(agent.policy, device=config.agent_config.device), elo_score)
            if (step + 1) % config.env.swap_steps == 0:
                env.unwrapped.swap_agent()
                observation, info = env.reset()

        if (step + 1) % config.eval_freq == 0:
            eval(agent, config.env, writer, step, identifier=identifier)

    os.makedirs(f"models/crossq/final/", exist_ok=True)
    torch.save(agent.state(), f"models/crossq/final/model.pkl") 
    env.close()




if __name__ == "__main__":
    fit_cross_q()