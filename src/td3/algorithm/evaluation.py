import torch 
import numpy as np
from hockey.hockey_env import Mode
from time import sleep
import random
from ..env import HockeyPlayer, make_hockey_eval_env, OpponentActor, load_actor
from tqdm import tqdm

def _evaluate_opponent_pool(
    env, 
    unwrapped_env, 
    n_eval_episodes: int,
    actor, 
    eval_opponents, 
    render: bool = False, 
    seed: int = 42,
):
    actor.eval()

    def run_episode(): 
        obs, _ = env.reset(seed=seed)
        accum_reward = 0 
        episode_len = 0 
        done = False
        outcome = 0 

        while not done:
            action = actor.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            if render: 
                env.render()

            # only keep track of rewards for envs that were not done in last steps  
            accum_reward += reward
            episode_len += 1

            if term:
                outcome = info["winner"]
            
            done = term or trunc
            obs = next_obs
        return accum_reward, outcome, episode_len
    
    def run_against_opponent(opponent): 
        unwrapped_env.opponent = opponent

        rewards = np.zeros(n_eval_episodes)
        outcomes = np.zeros(n_eval_episodes)
        episode_lengths = np.zeros(n_eval_episodes)

        for i in range(n_eval_episodes):
            mean_reward, outcome, episode_len = run_episode()
            rewards[i] = mean_reward
            outcomes[i] = outcome
            episode_lengths[i] = episode_len

        return np.mean(rewards), np.mean(outcomes == 1), np.mean(outcomes == -1), np.mean(outcomes == 0), np.mean(episode_lengths)
    

    results = dict() 
    for opponent_name, opponents in eval_opponents:
        if render: 
            print(f"Evaluating against {opponent_name} opponent...")

        reward, win_rate, lose_rate, draw_rate, avg_episode_len = run_against_opponent(opponents)
        results[opponent_name] = {
            "reward": reward,
            "win_rate": win_rate,
            "lose_rate": lose_rate,
            "draw_rate": draw_rate,
            "avg_episode_len": avg_episode_len,
        }
        if render: 
            print("Done. Next opponent.")   
            sleep(1)

    # TODO: only set to train if was in train before 
    actor.train()
    return results

def _set_seed(seed: int = 42): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

def setup_eval_env(hockey_mode, seed=42):
    # env setup
    player = HockeyPlayer(None, player_num=-1, player_name="player")
    env = make_hockey_eval_env(seed, mode=hockey_mode)()
    env_unwrapped = env.unwrapped
    return player, env, env_unwrapped

def evaluate(
    env, 
    unwrapped_env,
    n_eval_episodes: int,
    actor, 
    device = "cpu",
    is_self_play: bool = False,
    unwrapped_train_envs = None,
    default_opponents = True,
    custom_opponents = None,
    render: bool = False, 
    seed: int = 42,
):
    eval_opponents = []
    if default_opponents:
        eval_opponents.extend([
            ("weak", HockeyPlayer(OpponentActor(weak=True), player_num=len(eval_opponents), player_name="Weak")),
            ("strong", HockeyPlayer(OpponentActor(weak=False), player_num=len(eval_opponents)+1, player_name="Strong")),
        ])

    if is_self_play:
        assert unwrapped_train_envs is not None
        for i, env in enumerate(unwrapped_train_envs):
            eval_opponents.append((f"current_{i}", env.opponent))

    if custom_opponents is not None:
        for name, opponent in custom_opponents:
            if isinstance(opponent, str):
                opponent = load_actor(opponent, env, device=device)
            # otherwise assume it's already an actor
            opponent.eval()
            eval_opponents.append((name, HockeyPlayer(opponent, player_num=len(eval_opponents), player_name=name)))

    results = _evaluate_opponent_pool(env, unwrapped_env, n_eval_episodes, actor, eval_opponents, render, seed=seed)
    
    current_results = {}
    
    # for self-play evaluation, there now are stats "current_i" for all 1...n_env envs 
    # instead, take the aveage over all envs 
    for opponent_name, opponent_results in results.items():
        if opponent_name.startswith("current_"):
            for r, value in opponent_results.items():
                if r not in current_results:
                    current_results[r] = []
                current_results[r].append(value)
    if unwrapped_train_envs is not None and is_self_play:
        for i, _ in enumerate(unwrapped_train_envs): 
            results.pop(f"current_{i}")
        results["current"] = {r: np.mean(values) for r, values in current_results.items()}

    return results

def run_evaluation(player_path, n_episodes=10, render=True, seed=42, hockey_mode=Mode.NORMAL, use_default_opponents=True, custom_opponents=None, device = "cpu"):
    _set_seed(seed) 
    player, env, env_unwrapped = setup_eval_env(hockey_mode, seed)

    actor = load_actor(player_path, env, device)
    player.actor = actor

    results = evaluate(env, env_unwrapped, n_episodes, actor, device=device, default_opponents=use_default_opponents, custom_opponents=custom_opponents, render=render, seed=seed)
    env.close()
    return results 

def run_evaluation_multiple_runs(player_paths, n_episodes=10, render=True, seed=42, hockey_mode=Mode.NORMAL):
    _set_seed(seed) 
    player, env, env_unwrapped = setup_eval_env(hockey_mode, seed)

    actors = []
    for run_name in player_paths:
        if run_name in ["weak_opponent", "strong_opponent"]:
            actors.append(OpponentActor(weak=(run_name == "weak_opponent")))
        else:
            actors.append(load_actor(run_name, env))

    opponent_pool = []
    for i, player_path in enumerate(player_paths):
        opponent_pool.append((player_path, HockeyPlayer(actors[i], player_num=i, player_name=player_path)))
        
    all_results = dict()
    for i, actor in tqdm(enumerate(actors), total=len(actors), desc="Evaluating runs"):
        
        player.actor = actor
        # evaluate this actor againts all actors (including itself)
        results = _evaluate_opponent_pool(env, env_unwrapped, n_episodes, actor, opponent_pool, render=render, seed=seed)
        all_results[player_paths[i]] = results
        
    return all_results