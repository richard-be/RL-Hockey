import sys
import os

from pathlib import Path

from collections import defaultdict


base_path = Path(os.path.dirname(__file__)).parent.parent


sys.path.append(str(base_path))
sys.path.append(os.path.join(base_path, "crossq"))
print(base_path)


from dataclasses import dataclass
import numpy as np

from src.client.utils.actors import Actor
from src.client.utils.factory import construct_actor, construct_ensemble
import hockey.hockey_env as h_env
from src.client.utils.env import HockeyEnv_AgenticOpponent


def run_episode(env, actor: Actor, render=False) -> int: 
        obs, _ = env.reset()
        done = False
        outcome = 0 

        while not done:
            action = actor.act(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            if render: 
                env.render()

            if term:
                outcome = info["winner"]
            
            done = term or trunc
            obs = next_obs
        return  outcome


def run_match(player1: Actor, 
              player2: Actor,
              num_episodes: int = 100) -> np.array:
    
    outcomes = []
    env = HockeyEnv_AgenticOpponent(opponent=player2)
    for _ in range(num_episodes):
        outcomes.append(run_episode(env, player1))
    return np.array(outcomes)
         


@dataclass
class Player:
    name: str
    algorithm: str
    weight_path: str

@dataclass
class PlayerEnsemble:
    name: str
    algorithm: str
    players: list[Player]

BASIC_PLAYER_POOL = [
    Player("CrossQ", "crq", "models/crossq/model.pkl"),
    Player("SAC 1m", "sac", "models/sac/sac_2_1_1000000_1771781495.pkl"),
    Player("SAC 4m", "sac", "models/sac/sac_0.0_False_2.0_0.05_4000000_1769853898.pkl"), 

]
TD3_PLAYER_POOL = [
    Player("TD3", "td3", "models/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771317357/1000000.model"), 
    Player("TD3 (new)", "td3", "models/td3/HockeyOne-v0/ensemble/rnd_0x5-1_sp_1_opps_sac4m-sac1m-crossq0_timesteps_2e6__42__1771891165/1220000.model"), 
]

PLAYER_POOL = [
    PlayerEnsemble("Mean Action Ensemble", "mean", players=TD3_PLAYER_POOL),
    PlayerEnsemble("Random Action Ensemble", "random", players=TD3_PLAYER_POOL),
    # PlayerEnsemble("Greedy Action Ensemble", "greedy", players=BASIC_PLAYER_POOL),
    # PlayerEnsemble("Weighted Mean Action Ensemble", "weighted", players=BASIC_PLAYER_POOL),
] + BASIC_PLAYER_POOL + TD3_PLAYER_POOL


def run_tournament() -> None:
    env = h_env.HockeyEnv_BasicOpponent()
    players = {}
    win_rates = defaultdict(int)
    for player in PLAYER_POOL:
        if isinstance(player, Player):
            players[player.name] = construct_actor(player.algorithm, player.weight_path, env)
        else:
            players[player.name] = construct_ensemble(player.algorithm, {sub_player.algorithm: sub_player.weight_path
                                                                          for sub_player in player.players}, env)
     
    
    for name, player in players.items():
        for other_name, other_player in players.items():
            if name == other_name: continue
            

            result = run_match(player, other_player)
            win_rates[name] += np.sum(result == 1)
            win_rates[other_name] += np.sum(result == -1)
            print(f"{name} vs {other_name}\n Player 1's Win: {np.mean(result == 1)} Draw: {np.mean(result == 0)} Player2's Win {np.mean(result == -1)}")

    for name, wins in win_rates.items():
        print(f"{name}:\nNumber of Wins: {wins}\n")


if __name__ == "__main__":
    run_tournament()
