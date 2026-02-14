import tyro 
import json 
from algorithm.env import make_env

import random
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
from typing import Optional
from hockey.hockey_env import Mode 
from algorithm.td3 import Actor
import time 

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

    num_envs: int = 1
    weak_opponent = False 
    hockey_mode = Mode.NORMAL
    n_episodes: int = 10
    render: bool = False

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
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 

    device = "cpu"

    # env setup
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed+i, True, i, args.run_name, 250, True, weak_opponent=args.weak_opponent, mode=args.hockey_mode) for i in range(args.num_envs)])

    model_path = f"runs/{args.run_name}/{args.exp_name}.cleanrl_model"
    actor_state, qf1_state, qf2_state = torch.load(model_path, map_location=device)
    print(f"loaded model from {model_path}")
    actor = Actor(env)    
    actor.load_state_dict(actor_state)
    actor.to(device)
    
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    def run_episode(max_steps=250): 
        obs, _ = env.reset(seed=args.seed)
        accum_reward = np.zeros(args.num_envs)
        is_winner = [False] 

        for _ in range(max_steps): 
            actions = actor(torch.Tensor(obs).to(device)).detach().cpu().numpy()
            
            # TRY NOT TO MODIFY: execute the game and log data.
            obs, reward, term, trunc, info = env.step(actions)
            accum_reward += reward

            if args.render:
                env.render()
            if term: 
                is_winner = info["winner"] == 1 
                break
        return accum_reward.mean(), np.sum(is_winner)
    

    n_won = 0 
    for _ in range(args.n_episodes):
        reward, is_winner = run_episode()
        print(f"Reward: {reward}; Won: {is_winner}")
        n_won += is_winner
    print(f"Win rate: {(n_won/args.n_episodes):.3f}")
    env.close()
