import tyro 
from dataclasses import dataclass
from typing import Optional
from hockey.hockey_env import Mode 

from .algorithm.evaluation import run_evaluation

@dataclass
class Args:
    env_id: str = "HockeyOne-v0"
    player_path: Optional[str] = None
    expname: Optional[str] = None
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    time: str = "latest"
    
    model_checkpoint = "latest"

    weight_dir: str = "models/td3"

    num_envs: int = 10
    weak_opponent = True 
    hockey_mode = Mode.NORMAL
    n_episodes: int = 10
    render: bool = True

    default_opponents: bool = True
    opponents: Optional[tuple] = ("") 

def find_latest_time(pattern, root_dir): 
    import glob 
    latest_time = 0
    for f in glob.glob(pattern, root_dir=root_dir):
        try: 
            time = f.split("__")[-1].split(".")[0]
            time = int(time) 
            latest_time = max(latest_time, time)
        except: 
            print("Skipping", f, time)
            continue
    return latest_time


def main():
    args = tyro.cli(Args)
    player_path = args.player_path

    if not player_path: 
        player_path = f"{args.expname}__{args.seed}__"
        if args.time == "latest": 
            args.time = str(find_latest_time(player_path+"*", f"models/td3/{args.env_id}/"))
        player_path += args.time

        model_weights_dir = f"{args.weight_dir}/{args.env_id}/{player_path}/"
        checkpoint = find_latest_time(f"*.model", model_weights_dir) if args.model_checkpoint == "latest" else args.model_checkpoint

        player_path = f"td3:{model_weights_dir}{checkpoint}.model"
    print(player_path) 

    opponents = None
    if args.opponents:
        opponents = [x.split(":", 1) for x in args.opponents]
        
    results = run_evaluation(
        player_path=player_path,
        n_episodes=args.n_episodes, 
        render=args.render, 
        seed=args.seed, 
        hockey_mode=args.hockey_mode, 
        use_default_opponents=args.default_opponents,
        custom_opponents=opponents,
    )

    for opponent, stats in results.items():
        print("Opponent:", opponent)
        for key, value in stats.items():
            print(f"  {key}: {value}")
