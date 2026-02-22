
import hockey.hockey_env as h_env
from utils.actors import Actor
import gymnasium as gym
import numpy as np


# def compute_expected_scores(rating_a: float, rating_b: float) -> tuple[float, float]:

#     e_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
#     e_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

#     return e_a, e_b


# def update_elo_ratings(old_rating_a: float, 
#                        old_rating_b: float, 
#                        k_factor: int,
#                        result: int) -> tuple[float, float]:
#     # results is the outcome of the game:
#     # 1 - agent a has won
#     # 0 - draw
#     # -1 - agent b has won
#     e_a, e_b = compute_expected_scores(old_rating_a, old_rating_b)  # first compute the  expected scores of the agents

#     if result == 1:
#         score_a, score_b = 1.0, 0.0
#     elif result == -1:
#         score_a, score_b = 0.0, 1.0
#     else:
#         score_a, score_b = 0.5, 0.5

#     elo_a = old_rating_a + k_factor * (score_a - e_a)
#     elo_b = old_rating_b + k_factor * (score_b - e_b)

#     return elo_a, elo_b


# class HockeyPlayer():
#     def __init__(self, actor: Actor, player_num, elo=1200, player_name=""): 
#         self.actor = actor
#         self.player_num = player_num
#         self.elo = elo  
#         self.player_name = player_name

#     def act(self, obs):
#         return self.actor.act(obs) 

#     def update_elo(self, outcome, opponent_elo):
#         """outcome: 1 if opponent wins, -1 if opponent loses, 0 for draw"""
#         assert outcome in [1, -1, 0], "Invlaid outcome for Elo update"
#         self.elo = compute_new_elo(self.elo, opponent_elo, outcome)

class HockeyEnv_AgenticOpponent(h_env.HockeyEnv):
    def __init__(self, opponent: Actor, 
                 mode=h_env.Mode.NORMAL):
        super().__init__(mode=mode, keep_mode=True)
        self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)
        self.opponent = opponent

    def step(self, action):
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])
        return super().step(action2)