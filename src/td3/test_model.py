import tyro 
import json 
from algorithm.env import make_hockey_env, make_hockey_env_self_play, HockeyPlayer, make_hockey_eval_env
from hockey.hockey_env import HockeyEnv_BasicOpponent, BasicOpponent

from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
from typing import Optional
from hockey.hockey_env import Mode 
from algorithm.td3 import Actor
import time 
from algorithm.evaluation import run_evaluation

@dataclass
class Args:
    env_id: str = "HockeyOne-v0"
    exp_name: str = None
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    time: str = "latest"
    """the time of the experiment"""
    run_name: Optional[str] = None 

    num_envs: int = 10
    weak_opponent = True 
    hockey_mode = Mode.NORMAL
    n_episodes: int = 10
    render: bool = True

    use_default_opponents: bool = True
    opponents: Optional[tuple] = ("") 

def find_latest_time(pattern): 
    import glob 
    latest_time = 0
    for f in glob.glob(pattern, root_dir="runs"):
        try: 
            time = f.split("__")[-1]
            time = int(time) 
            latest_time = max(latest_time, time)
        except: 
            print("Skipping", f)
            continue
    return latest_time


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.run_name: 
        args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__"
        if args.time == "latest": 
            args.time = str(find_latest_time(args.run_name+"*"))
        args.run_name += args.time
    # hyperparams = open(f"runs/{args.run_name}/args.json")
    print(args.run_name)

    opponents = None
    if args.opponents:
        opponents = [x.split(":") for x in args.opponents]
        
    results = run_evaluation(
        run_name=args.run_name,
        n_episodes=args.n_episodes, 
        render=args.render, 
        seed=args.seed, 
        hockey_mode=args.hockey_mode, 
        use_default_opponents=args.use_default_opponents,
        custom_opponents=opponents,
        exp_name=args.exp_name, 
        num_envs=args.num_envs
    )

    for opponent, stats in results.items():
        print("Opponent:", opponent)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # # TRY NOT TO MODIFY: start the game
    # def run_episode(max_steps=500): 
    #     obs, _ = env.reset(seed=args.seed)
    #     accum_reward = np.zeros(args.num_envs)
    #     is_winner = [False] 
    #     is_loser = [False] 

    #     for _ in range(max_steps): 
    #         actions = actor(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            
    #         # TRY NOT TO MODIFY: execute the game and log data.
    #         obs, reward, term, trunc, info = env.step(actions)
    #         accum_reward += reward

    #         if args.render:
    #             env.render()
    #         if term: 
    #             is_winner = info["winner"] == 1 
    #             is_loser = info["winner"] == -1
    #             break
    #     return accum_reward.mean(), np.sum(is_winner), np.sum(is_loser)
    
    # for opponent_name, opponent in [("weak", BasicOpponent(True)), ("strong", BasicOpponent(False)), ("self", HockeyPlayer(actor.clone()))]: 
    #     print(f"Testing against {opponent_name} opponent...")
    #     for hockey_env in env.envs: 
    #         hockey_env.opponent = opponent

    #     n_won = 0 
    #     n_lost = 0
    #     n_draw = 0
    #     for _ in range(args.n_episodes):
    #         reward, is_winner, is_loser = run_episode()
    #         is_draw = 1 - is_winner - is_loser
    #         # print(f"Reward: {reward}; Won: {is_winner}, Lost: {is_loser}, Draw: {is_draw}")
    #         n_won += is_winner
    #         n_lost += is_loser
    #         n_draw += is_draw
    #     print(f"Win rate: {(n_won/args.n_episodes):.3f}, Lost rate: {(n_lost/args.n_episodes):.3f}, Draw rate: {(n_draw/args.n_episodes):.3f}")

    # envs.close()
