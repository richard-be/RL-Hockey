import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
import heapq

class OpponentResetWrapper(gym.Wrapper):
  def __init__(self, env, opponent_sampler, episode_count):
    super().__init__(env)
    self.opponent_sampler = opponent_sampler
    self.episode_count = episode_count

  def reset(self, **kwargs):
    id, opponent = self.opponent_sampler.sample_opponent(self.episode_count.value)
    self.current_opponent_id = id
    self.env.set_opponent(opponent)
    return self.env.reset(**kwargs)
  
class EloWrapper(gym.Wrapper):
    def __init__(self, env, elo_system):
        super().__init__(env)
        self.current_opponent_id = None
        self.elo_system = elo_system

    def reset(self, **kwargs):
        # kwargs should include opponent_id
        self.current_opponent_id = kwargs.get("opponent_id")
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            opponent_id = self.env.current_opponent_id
            score = info.get("winner") # in {-1, 0, 1}
            score = (score + 1) / 2
            self.elo_system.update_elo("self_0", opponent_id, score)
        return obs, reward, terminated, truncated, info
  
class OpponentSampler():
  """samples opponent, keeps dict of ids and players"""
  def __init__(self, self_play_len):
    self.opponents = {"easy": h_env.BasicOpponent(), "hard": h_env.BasicOpponent(weak=False)}
    self.self_play_pool = []
    self.custom_opponent_pool = []
    self.self_play_len = self_play_len
    self.deleted_actors = ["easy", "hard", "self_0"]

  def sample_opponent(self, global_episode):
    probs = self.get_probs(global_episode)
    choice = np.random.choice(["easy", "hard", "custom", "self"], p=probs)
    if choice == "easy":
      opponent = self.opponents["easy"]
      opponent_id = "easy"
    elif choice == "custom" and len(self.custom_opponent_pool) > 0:
        opponent_id = np.random.choice(self.custom_opponent_pool)
        opponent = self.opponents[opponent_id]
    elif choice == "self" and len(self.self_play_pool) > 0:
        opponent_id = np.random.choice(self.self_play_pool)
        opponent = self.opponents[opponent_id]
    else:
      opponent = self.opponents["hard"] #if custom or self dont exist, just use more hard basic enemies
      opponent_id = "hard"
    return opponent_id, opponent    
  
  def get_probs(self, global_episode):
    if global_episode < 25e2:
      probs = [1, 0, 0, 0]
    elif global_episode < 6e3:
      probs = [0.2, 0.8, 0, 0]
    else:
      probs = [0.1, 0.2, 0.3, 0.4]
    return probs
         
  def add_self_play_opponent(self, frozen_actor, frozen_index):
    self.opponents[f"self_{frozen_index}"] = frozen_actor

  def update_self_play_pool(self, elo_dict):
    elo_dict_selected = {k: v for k, v in elo_dict.items() if k not in self.deleted_actors} #delete the active actor (and maybe basic and custom actors, idk yet) 
    self.self_play_pool = heapq.nlargest(self.self_play_len, elo_dict_selected)

  def add_custom_opponent(self, opponent, name):
     self.opponents[name] = opponent
     self.custom_opponent_pool.append(name)

class EpisodeCounter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value
    
class EloSystem:
  def __init__(self, k = 32):
    self.elo_dict = {"self_0": 1500, "easy": 1500, "hard": 1500}
    self.k = k

  def get_elo_dict(self):
    return self.elo_dict
  
  def get_elo(self, player):
    return self.elo_dict[player]
  
  def register_player(self, player, elo=1500.0):
    if player not in self.elo_dict:
      self.elo_dict[player] = elo
    else:
      raise ValueError("Player already registered")

  def update_elo(self, player_a, player_b, score_a):
    elo_a = self.elo_dict[player_a]
    elo_b = self.elo_dict[player_b]
    prob_a = 1/(1+10**((elo_b-elo_a)/400))
    prob_b = 1/(1+10**((elo_a-elo_b)/400))
    elo_a_new = elo_a + self.k * (score_a-prob_a)
    elo_b_new = elo_b + self.k * (1-score_a-prob_b)
    self.elo_dict[player_a] = elo_a_new
    self.elo_dict[player_b] = elo_b_new

