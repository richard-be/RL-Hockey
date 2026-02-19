import torch 
import numpy as np
from hockey.hockey_env import BasicOpponent, Mode
from time import sleep
import random
from .env import HockeyPlayer, make_hockey_eval_env, OpponentActor
from .td3 import Actor
import gymnasium as gym
from tqdm import tqdm

def _evaluate_opponent_pool(
    eval_envs, 
    unwrapped_eval_envs, 
    n_eval_episodes: int,
    actor, 
    eval_opponents, 
    device: torch.device = torch.device("cpu"),
    render: bool = False, 
):
    actor.eval()

    n_envs = len(eval_envs.envs)

    # this is a bit hacky, but make sure the that
    one_starts = [False for _ in unwrapped_eval_envs] 
    def run_episode(): 
        obs, _ = eval_envs.reset()
        for i, env in enumerate(unwrapped_eval_envs):
            env.one_starts = one_starts[i] 
            one_starts[i] = not one_starts[i]

        done = np.zeros(n_envs, dtype=bool) 
        accum_rewards = np.zeros(n_envs)
        won = np.zeros(n_envs)
        lost = np.zeros(n_envs)
        max_steps = 250

        while not done.all() and max_steps > 0:
            max_steps -= 1

            actions = actor.act(obs)
            next_obs, rewards, terms, truncs, infos = eval_envs.step(actions)
            if render and not done[0]:
                eval_envs.envs[0].render()

            # only keep track of rewards for envs that were not done in last steps  
            accum_rewards[~done] += rewards[~done]

            for i, terminated in enumerate(terms): 
                if terminated and not done[i]: # just finished now
                    won[i] = infos["winner"][i] == 1
                    lost[i] = infos["winner"][i] == -1
            
            done |= (terms | truncs)
            obs = next_obs

        draw_rate = (1 - won - lost).mean()
        return accum_rewards.mean(), won.mean(), lost.mean(), draw_rate
    
    def run_against_opponent(opponents): 
        # use same opponent for all envs if only one opponent is provided
        if not isinstance(opponents, list): 
            opponents = [opponents] * len(unwrapped_eval_envs)

        for env, opponent in zip(unwrapped_eval_envs, opponents):
            env.opponent = opponent

        rewards, win_rates, loss_rates, draw_rates = [], [], [], []
        for _ in range(n_eval_episodes):
            mean_reward, win_rate, loss_rate, draw_rate = run_episode()
            rewards.append(mean_reward)
            win_rates.append(win_rate)
            loss_rates.append(loss_rate)
            draw_rates.append(draw_rate)
        return np.mean(rewards), np.mean(win_rates), np.mean(loss_rates), np.mean(draw_rates)
    

    results = dict() 
    for opponent_name, opponents in eval_opponents:
        if render: 
            print(f"Evaluating against {opponent_name} opponent...")
        reward, win_rate, lose_rate, draw_rate = run_against_opponent(opponents)
        results[opponent_name] = {
            "reward": reward,
            "win_rate": win_rate,
            "lose_rate": lose_rate,
            "draw_rate": draw_rate,
        }
        if render: 
            print("Done. Next opponent.")   
            sleep(1)

    actor.train()
    return results

def _set_seed(seed: int = 42): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

def _load_external_actor(run_name, envs, device="cpu"):
    # from .externals import Actor as SACActor, add_act_method
    from .td3 import Actor as TD3Actor
    from .externals.sac import Actor as SACActor
    from .externals.utils import add_act_method
    # from .externals.crossq import Actor as CRQActor

    external_actor_types = {
        "sac": SACActor,
        "td3": TD3Actor,
        "crq": None, 
    }

    LEN_PREFIX = 3
    # assume rest of run name is path to model
    model_type = run_name[:LEN_PREFIX+1]
    assert model_type[LEN_PREFIX] == "-" # format is ext__{model_type}-{path}
    path = run_name[len(model_type):]

    actor_state = torch.load(path, map_location=device)
    if isinstance(actor_state, tuple) and len(actor_state) == 3:   
        actor_state = actor_state[0] # if it's a td3 model, just load the actor state
    print(f"Loaded external model from {path}")
    
    actor_type = external_actor_types.get(model_type[:LEN_PREFIX], None)
    if actor_type is None:
        raise ValueError(f"Unknown external model type: {model_type[:LEN_PREFIX]}")
    
    actor = actor_type(envs)
    actor.load_state_dict(actor_state)
    add_act_method(actor)

    return actor.to(device)

def _load_actor(run_name, envs, exp_name=None, device="cpu"):
    if run_name and run_name.startswith("ext__"):
        return _load_external_actor(run_name[len("ext__"):], envs, device)
    
    print("Loading actor of run", run_name)
    exp_name = run_name.split("__")[1] if exp_name is None else exp_name
    print("name", exp_name)
    model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
    actor_state, qf1_state, qf2_state = torch.load(model_path, map_location=device)

    actor = Actor(envs)    
    actor.load_state_dict(actor_state)
    actor.to(device)
    actor.eval()
    return actor 


def _setup_eval_envs(hockey_mode, num_envs, seed=42):
    # env setup
    player = HockeyPlayer(None, player_num=-1)
    envs = gym.vector.SyncVectorEnv([make_hockey_eval_env(seed + 1, mode=hockey_mode) for i in range(num_envs)])
    eval_envs_unwrapped = [env.unwrapped for env in envs.envs]
    return player, envs, eval_envs_unwrapped

def evaluate(
    eval_envs, 
    unwrapped_eval_envs, 
    n_eval_episodes: int,
    actor, 
    is_self_play: bool = False,
    unwrapped_train_envs = None, 
    default_opponents = True,
    custom_opponents = None,
    device: torch.device = torch.device("cpu"),
    render: bool = False
):
    eval_opponents = []
    if default_opponents:
        eval_opponents.extend([
            ("weak", [OpponentActor(weak=True) for _ in unwrapped_eval_envs]),
            ("strong", [OpponentActor(weak=False) for _ in unwrapped_eval_envs]),
        ])

    if is_self_play:
        assert len(eval_envs.envs) == len(unwrapped_train_envs), "number of envs must match"
        eval_opponents.append(("current", [env.opponent for env in unwrapped_train_envs]))

    if custom_opponents is not None:
        for name, opponent in custom_opponents:
            opponent = _load_actor(opponent, eval_envs)
            eval_opponents.append((name, [opponent for _ in unwrapped_eval_envs]))

    return _evaluate_opponent_pool(eval_envs, unwrapped_eval_envs, n_eval_episodes, actor, eval_opponents, device, render)


def run_evaluation(run_name, n_episodes=10, render=True, seed=42, hockey_mode=Mode.NORMAL, use_default_opponents=True, custom_opponents=None, exp_name=None, num_envs=10, device = "cpu"):
    _set_seed(seed) 
    player, envs, eval_envs_unwrapped = _setup_eval_envs(hockey_mode, num_envs, seed)

    actor = _load_actor(run_name, envs, exp_name, device)
    player.actor = actor

    results = evaluate(envs, eval_envs_unwrapped, n_episodes, actor, default_opponents=use_default_opponents, custom_opponents=custom_opponents, render=render, device=device)
    envs.close()
    return results 

def run_evaluation_multiple_runs(run_names, n_episodes=10, render=True, seed=42, hockey_mode=Mode.NORMAL, num_envs=10):
    _set_seed(seed) 
    player, envs, eval_envs_unwrapped = _setup_eval_envs(hockey_mode, num_envs, seed)
    actors = []
    for run_name in run_names:
        if run_name in ["weak_opponent", "strong_opponent"]:
            actors.append(OpponentActor(weak=(run_name == "weak_opponent")))
        else:
            actors.append(_load_actor(run_name, envs))

    opponent_pool = list(zip(run_names, actors))

    all_results = dict()
    for i, actor in tqdm(enumerate(actors), total=len(actors), desc="Evaluating runs"):
        
        player.actor = actor
        # evaluate this actor againts all actors (including itself if it is a self-play run)
        results = _evaluate_opponent_pool(envs, eval_envs_unwrapped, n_episodes, actor, opponent_pool, render=render)
        all_results[run_names[i]] = results
        
    return all_results