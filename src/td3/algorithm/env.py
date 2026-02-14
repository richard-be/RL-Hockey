from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode, HockeyEnv, BasicOpponent
import gymnasium as gym
from gymnasium import spaces
import numpy as np  
from typing import List
import torch 
from algorithm.td3 import Actor

# NOTE: original env creation function 
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env.enabled = False
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
# END OF NOTE 

# NOTE: ADDED OWN CODE HERE
# To support video 
class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        env.render_mode = "rgb_array"

    def render(self, **kwargs): 
        return self.env.render(mode=self.env.render_mode)

def wrap_hockey_env(env, seed, idx, capture_video=False, run_name=None, max_episode_steps=None):
    if capture_video and idx == 0:
        env = RenderWrapper(env)
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.enabled = False
    if max_episode_steps is not None: 
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def compute_new_elo(player_elo, opponent_elo, outcome):
  """Compute new Elo for a player based on outcome and opponent's Elo."""
  # NOTE: this is based on https://huggingface.co/learn/deep-rl-course/unit7/self-play#the-elo-score-to-evaluate-our-agent
  
  K = 16 if player_elo > 1500 else 32  # higher K for lower Elo
  expected_score = 1 / (1 + 10**((opponent_elo - player_elo) / 400))
  actual_score = {1: 1, -1: 0, 0: 0.5}[outcome]
  return player_elo + K * (actual_score - expected_score)


class HockeyPlayer():
  def __init__(self, actor: Actor, elo=1200): 
    self.actor = actor
    self.elo = elo  

  def act(self, obs):
    return self.actor.act(obs) 
  
  def update_elo(self, outcome, opponent_elo):
    """outcome: 1 if opponent wins, -1 if opponent loses, 0 for draw"""
    # Simple Elo update based on reward
    assert outcome in [1, -1, 0], "Invlaid outcome for Elo update"
    self.elo = compute_new_elo(self.elo, opponent_elo, outcome)

class HockeyEnv_CustomPlayers(HockeyEnv_BasicOpponent):
    def __init__(self, player: HockeyPlayer, mode=Mode.NORMAL):        
        self.opponent_pool = [
            # First start with the basic opponents, add more during training
            HockeyPlayer(BasicOpponent(weak=True)),
            HockeyPlayer(BasicOpponent(weak=False)), 
        ]
        self.player = player
        self.opponent_selection_temp = 100 # temperature that allows for random opponents TODO: find parameter 
        self.diff_elo_to_add_opponent = 100 # if player has improved by this much since last opponent was added, add new opponent to pool; TODO: find parameter
        self.last_player_elo_when_adding_opponent = self.player.elo

        super().__init__(mode=mode, weak_opponent=False) 

        # self.level = 0 
        # self.current_level_time = 0 
        # self.current_win_rate = 0.0
        # self.min_steps_to_next_level = 100
        # self.win_rate_threshold_to_next_level = 0.8
        # self.opponent_probabilities = [[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]] # probabilities of selecting each opponent for each level

        # self.opponent = self.select_opponent()



    def reset(self, one_starting=None, mode=None, seed=None, options=None): 
        # Re-select opponent on reset => dynamic opponent selection based on current level
        obs, info = super().reset(one_starting, mode, seed, options)
        self.opponent = self.select_opponent()
        # self.level -= 1 
        # self.next_level()
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        # todo: truncate if n steps is too long? => currently there will never be draw because trunc is always false and term only if winner
        if term: 
            # update statistics if episode is over 
            outcome = self.winner
            # update elos: 

            player_elo = self.player.elo
            opponent_elo = self.opponent.elo

            self.player.update_elo(outcome, opponent_elo)
            self.opponent.update_elo(- outcome, player_elo)

            # n_wins_currently = self.current_win_rate * self.current_level_time
            # n_wins_currently += (outcome == 1)

            # self.current_level_time += 1 
            # self.current_win_rate = n_wins_currently / self.current_level_time

            # # TODO: what parameters make sense here? 
            # if self.current_win_rate > self.win_rate_threshold_to_next_level and self.current_level_time > self.min_steps_to_next_level:
            #     self.next_level()

            if self.player.elo > self.last_player_elo_when_adding_opponent + self.diff_elo_to_add_opponent: 
                # if player has improved significantly since last opponent was added, add new opponent to pool
                self.add_actor_to_opponent_pool(self.player.actor.clone())
                self.last_player_elo_when_adding_opponent = self.player.elo

        return obs, reward, term, trunc, info

    def select_opponent(self) -> HockeyPlayer:
        # Select opponent based on probabilities of current level
        # current_probabilities = self.opponent_probabilities[self.level] 
        # return np.random.choice(self.opponent_pool, p=current_probabilities)
        elos = np.array([opponent.elo for opponent in self.opponent_pool])
        elo_diffs = np.abs(elos - self.player.elo)
        selection_probabilities = torch.softmax(-torch.tensor(elo_diffs) / (self.opponent_selection_temp), dim=0).numpy()
        return np.random.choice(self.opponent_pool, p=selection_probabilities)
    
    # def next_level(self, rand_chance=0.1):
    #     n_levels = len(self.opponent_probabilities)
    #     self.level += 1
    #     # cap level to max defined probabilities
    #     self.level = min(self.level, n_levels-1)

    #     # with low probability swith to random level to make sure agent doesn't forget old opponents
    #     if np.random.rand() < rand_chance: 
    #         self.level = np.random.randint(0, n_levels)

    #     self.current_level_time = 0 
    #     self.current_win_rate = 0.0

    def add_actor_to_opponent_pool(self, actor): 
        # for level in range(len(self.opponent_probabilities)): 
        #     # add new opponent with low probability to each level
        #     self.opponent_probabilities[level].append(0.1) 
        #     # renormalize probabilities
        #     total = sum(self.opponent_probabilities[level])
        #     self.opponent_probabilities[level] = [p / total for p in self.opponent_probabilities[level]]
        # self.opponent_probabilities.append([0.1] * len(self.opponent_pool) + [0.9]) # new level with new opponent as main opponent
        self.opponent_pool.append(HockeyPlayer(actor))


def make_hockey_env(seed, idx, capture_video, run_name, max_episode_steps=None, mode=Mode.NORMAL, weak_opponent=False):
    def thunk(): 
        env = HockeyEnv_BasicOpponent(mode=mode, weak_opponent=weak_opponent)
        return wrap_hockey_env(env, seed, idx, capture_video=capture_video, run_name=run_name, max_episode_steps=max_episode_steps)
    return thunk

def make_hockey_env_self_play(seed, idx, capture_video, run_name, player: HockeyPlayer, max_episode_steps=None, mode=Mode.NORMAL):
    def thunk(): 
        env = HockeyEnv_CustomPlayers(player, mode=mode)
        return wrap_hockey_env(env, seed, idx, capture_video=capture_video, run_name=run_name, max_episode_steps=max_episode_steps)
    return thunk