#based on cleanrl sac_continuous action implementation
import tyro
import random
import time
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from agent.sac import Actor
import hockey.hockey_env as h_env
from dataclasses import dataclass
import os
import env.custom_hockey as c_env



def make_env(env_id, seed, weak_opponent, env_mode, opponent):
    print(env_id)
    print(env_mode)
    def thunk():
        if opponent=="basic":
            env = h_env.HockeyEnv_BasicOpponent(mode=h_env.Mode[env_mode], weak_opponent=weak_opponent)
        elif opponent=="human":
            env = c_env.HockeyEnv_HumanOppoent(mode=h_env.Mode[env_mode])
        else:
            env = c_env.HockeyEnv_Custom_BasicOpponent(mode=h_env.Mode[env_mode], weak_opponent=weak_opponent) 
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    weak_opponent: bool = False
    """which opponent to test against"""
    num_games: int = 5
    """how many games to play"""
    seed: int = 1
    """random seed"""
    env_id: str="Hockey-v0"
    """which env"""
    num_envs: int=1
    """how many envs in parallel"""
    actor_file: str=""
    """the torch file from where to load the actor network"""
    opponent_file: str=""
    """the torch file from where to load the opponent actor network"""
    env_mode: str="NORMAL"
    """env play mode"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    render: bool = True
    """if toggled, window will be rendered"""
    opponent: str="basic"
    """which opponent to play against"""

if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.weak_opponent}__{args.actor_file}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    if len(args.opponent_file) > 0:
        opponent = torch.load(os.path.join("models"), args.opponent_file, map_location=device) #also state dict load?
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, args.weak_opponent, args.env_mode, args.opponent) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs)
    actor.load_state_dict(torch.load(os.path.join("models/sac", args.actor_file), map_location=device))
    actor.to(device)
    #actor = torch.load(os.path.join("actors", args.actor_file), map_location=device) #also state dict load?


    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    episode_count = 0
    global_step = 0
    while episode_count < args.num_games:
        # ALGO LOGIC: put action logic here
        actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if args.render:
            envs.envs[0].render()
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for env_index, info in enumerate(infos["final_info"]):
                if info is not None:
                    episode_count += 1
                    print(f"episode={episode_count}, global_step={global_step}, env={env_index}, winner={info['winner']}, episodic_return={info['episode']['r']}, episode_length={info['episode']['l']}")
                    break

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 1000 == 0:
            print("SPS:", int(global_step / (time.time() - start_time)))
        global_step+=1
    envs.close()
