import sys
import os

from pathlib import Path

base_path = Path(os.path.dirname(__file__)).parent
sys.path.append(str(base_path))

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
from dataclasses import dataclass, field
from functools import partial

from memory import ReplayBuffer


from crossq import CrossQAgent, CrossQAgentConfig
from env.agentic_opponent import HockeyEnv_SelfPlay, HockeyEnv_AgenticOpponent, AgenticOpponent, construct_crossq_opponent, OpponentPool
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
    swap_steps: int = 30_000
    opponent_save_steps: int = 50_000
    warmup_period: int = 60_000

    initial_elo: float = 1200.0
    k_factor: int = 16

@dataclass
class CrossQConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent_config: CrossQAgentConfig = field(default_factory=CrossQAgentConfig)
    
    train_steps: int = 2_000_000
    learning_starts: int = 10_000
    save_freq: int = 100_000
    log_freq: int = 900
    eval_freq: int = 50_000

    use_tensorboard: bool = True
    num_envs: int = 10

    seed: int = 42


def is_hockey(env_id: str) -> bool:
    return env_id.startswith("Hockey")


def create_environment(env_config: EnvConfig, custom_opponent: None | AgenticOpponent = None,  # for custom opponent and selfplay type env
                       opponent_pool: OpponentPool | None = None) -> gym.Env:
    if is_hockey(env_config.env_id):
        if env_config.mode == h_env.Mode.NORMAL:
            if env_config.opponent_type == 'selfplay' and opponent_pool:
                env = HockeyEnv_SelfPlay(opponent_pool=opponent_pool, 
                                         swap_steps=env_config.swap_steps,
                                         warmup_period=env_config.warmup_period)
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
        env = create_environment(EnvConfig(env_id=env_config.env_id,
                                      mode=env_config.mode,
                                      opponent_type=env_config.opponent_type,
                                    #   render_mode="rgb_array",
                                      ))
        # env = RecordVideo(env, episode_trigger=lambda _: True, video_folder=f"videos/crossq/{identifier}/{global_step}", 
        #                                      name_prefix="eval")
        envs = {"basic": env}
        
    
    
    for name, env in envs.items():
        observation, info = env.reset()
        env.observation_space.dtype = np.float32
        print(observation.shape, observation.dtype)
        while True:
            action = agent.act(observation[np.newaxis, :]).cpu().numpy()
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
    # env = create_environment(config.env)
    if config.env.opponent_type == "selfplay":
        elo_score = config.env.initial_elo
        opponent_pool = OpponentPool(window_size=config.env.window_size,
                                     play_against_latest_model_ratio=config.env.play_against_latest_model_ratio,
                                     fixed_agents=[(h_env.BasicOpponent(), 1200),
                                                   (h_env.BasicOpponent(weak=False), 1500)])
    envs = gym.vector.SyncVectorEnv([partial(create_environment, config.env, 
                                             opponent_pool=opponent_pool if config.env.opponent_type == "selfplay" else None) for _ in  range(config.num_envs)])

    envs.single_observation_space.dtype = np.float32
    agent = CrossQAgent(config=config.agent_config, 
                        obs_dim=np.prod(envs.single_observation_space.shape),
                        action_dim=np.prod(envs.single_action_space.shape),
                        action_low=envs.single_action_space.low,
                        action_high=envs.single_action_space.high,
                        replay_buffer=ReplayBuffer(config.agent_config.buffer_size,
                                                   observation_space=envs.single_observation_space,
                                                   action_space=envs.single_action_space,
                                                   device=config.agent_config.device,
                                                   n_envs=config.num_envs,
                                                   handle_timeout_termination=False))

    

    if config.env.opponent_type == "selfplay":
        opponent_pool.current_policy = construct_crossq_opponent(agent.policy, device=config.agent_config.device, copy=False)
        # opponent_pool.add_agent(construct_crossq_opponent(agent.policy, device=config.agent_config.device), elo_score)
    #     env.close()
    #     elo_score = config.env.initial_elo
    #     env: HockeyEnv_SelfPlay = create_environment(config.env, custom_opponent=construct_crossq_opponent(agent.policy, device=config.agent_config.device))

    identifier = f"CrossQ-{config.env.opponent_type}-{config.env.env_id}-{config.seed}-{int(time.time())}-{config.agent_config.q_lr}-{config.agent_config.buffer_size}-{config.agent_config.target}-{config.agent_config.batch_norm_type}"
    if config.use_tensorboard:
        
        writer = SummaryWriter(log_dir=f"runs/crossq/{identifier}")

    observation, info = envs.reset(seed=config.seed)
    num_episodes = 0
    running_stats = defaultdict(int)

    for _ in range(config.learning_starts):
        action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        next_observation, reward, terminated, truncated, info = envs.step(action)
        agent.store_transition(observation, action, next_observation, reward, terminated, info)

        observation = next_observation

    
    for step in range(config.train_steps):
        action = agent.act(observation).cpu().numpy()
        next_observation, reward, terminated, truncated, info = envs.step(action)

        real_next_obs = next_observation.copy()
        for idx, trunc in enumerate(truncated):
            if trunc and "final_observation" in info:
                real_next_obs[idx] = info["final_observation"][idx]
        
        agent.store_transition(observation, action, real_next_obs, reward, terminated, info)
        observation = next_observation
        if "final_info" in info:
            if config.use_tensorboard:
                for env_index, inf in enumerate(info["final_info"]):
                    if inf is None: continue
                    num_episodes += 1
                    writer.add_scalar("Env/Return", inf['episode']['r'], step)
                    writer.add_scalar("Env/Length", inf['episode']['l'], step)
                    if is_hockey(config.env.env_id):
                        winner = inf['winner']
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
                            opponent = inf['opponent']
                            elo_score_opponent = opponent_pool.get_opponent_score(agent=opponent)
                            elo_score, elo_score_opponent = update_elo_ratings(elo_score, elo_score_opponent if elo_score_opponent else elo_score,
                                                                                k_factor=config.env.k_factor, result=winner)
                            if elo_score_opponent is not None:
                                opponent_pool.update_opponent_score(new_score=elo_score_opponent, agent=opponent)
                            # env.unwrapped.update_opponent_score(new_score=elo_score_opponent)
                            writer.add_scalar("ELO/ELO_Self", elo_score, step)
                            writer.add_scalar(f"ELO/ELO_{opponent}", elo_score_opponent, step)

            # observation, info = envs.reset()
        
        update_policy = (step + 1) % config.agent_config.policy_delay == 0
        compute_logs = (step + 1) % config.log_freq == 0
        logs = agent.learn(update_policy=update_policy, compute_logs=compute_logs)

        if (step + 1) % config.log_freq == 0:

            if config.agent_config.dynamic_alpha:
                alpha_grad_norm = logs['alpha_grad_norm']
                alpha_value = logs['alpha_value']

            if update_policy:
                actor_loss = logs['actor_loss']
                entropy = logs['entropy']
                actor_grad_norm = logs['actor_grad_norm']

            q_weight_norms = logs["q_weight_norms"]
            actor_weight_norms = logs["actor_weight_norms"]
            critic_loss = logs["critic_loss"] 
            q_grad_norms = logs["q_grad_norms"]
            q_relu_stats = logs["critic_relu_stats"] 
            q_elrs = logs["critic_elrs"] 

            if config.agent_config.dynamic_alpha:
                writer.add_scalar("Grad/Alpha", alpha_grad_norm, step)
                writer.add_scalar("Value/Alpha", alpha_value, step)


            writer.add_scalar("Loss/Q_Loss", critic_loss, step)
            for idx, q_norm in enumerate(q_grad_norms):
                for layer_name, norm in q_norm.items():
                    writer.add_scalar(f"Grad/Q_{idx}_{layer_name}", norm, step)

            for idx, q_w_norm in enumerate(q_weight_norms):
                for layer_name, norm in q_w_norm.items():
                    writer.add_scalar(f"Value/Q_{idx}_{layer_name}_Weight_Norm", norm, step)
            
            for layer_name, norm in actor_weight_norms.items():
                    writer.add_scalar(f"Value/Actor_{layer_name}_Weight_Norm", norm, step)


            for idx, elrs in enumerate(q_elrs):
                for name, elr in elrs.items():
                    writer.add_scalar(f"ELR/Q_{idx}_{name}", elr, step)

            for idx, relu_stats in enumerate(q_relu_stats):
                for name, relu_stat_dict in relu_stats.items():
                    writer.add_scalar(f"ReLU/Dead_Q_{idx}_{name}", relu_stat_dict['dead'], step)
                    writer.add_scalar(f"ReLU/Dead_Ratio_Q_{idx}_{name}", relu_stat_dict['dead_ratio'], step)

            writer.add_scalar("Loss/Actor_Loss",  actor_loss, step)   
            writer.add_scalar("Loss/Entropy", entropy, step) 
            for layer_name, norm in actor_grad_norm.items():
                writer.add_scalar(f"Grad/Actor_{layer_name}", norm, step)

        if (step + 1) % config.save_freq == 0:
            os.makedirs(f"models/crossq/{identifier}/{step + 1}/", exist_ok=True)
            torch.save(agent.state(), f"models/crossq/{identifier}/{step + 1}/model.pkl") 

        if is_hockey(config.env.env_id) and config.env.opponent_type == "selfplay":
            if (step + 1) % config.env.opponent_save_steps == 0:  # start adding previous checkpints after warmup period
                opponent_pool.add_agent(construct_crossq_opponent(agent.policy, device=config.agent_config.device), elo_score)
         

        if (step + 1) % config.eval_freq == 0:
            eval(agent, config.env, writer, step, identifier=identifier)

    os.makedirs(f"models/crossq/{identifier}/final/", exist_ok=True)
    torch.save(agent.state(), f"models/crossq/{identifier}/final/model.pkl") 
    envs.close()




if __name__ == "__main__":
    fit_cross_q()