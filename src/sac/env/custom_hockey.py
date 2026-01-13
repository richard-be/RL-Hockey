import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
from collections import deque
import torch
from itertools import accumulate

class HockeyEnv_Custom(h_env.HockeyEnv):
  def get_reward(self, info):
    r = self._compute_reward()
    r += info["reward_closeness_to_puck"]
    r += info["reward_touch_puck"]
    r += info["reward_puck_direction"]
    return float(r)
  
class HockeyEnv_Custom_BasicOpponent(HockeyEnv_Custom):
  def __init__(self, mode=h_env.Mode.NORMAL, weak_opponent=False):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = h_env.BasicOpponent(weak=weak_opponent)
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)

  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)
  
class HockeyEnv_Custom_CustomOpponent(HockeyEnv_Custom):
  def __init__(self, opponent, device, mode=h_env.Mode.NORMAL):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = opponent
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)
    self.device = device

  def set_opponent(self, opponent):
    self.opponent = opponent

  def get_opponent_name(self):
    return type(self.opponent).__name__

  def step(self, action):
    ob2 = self.obs_agent_two()
    if type(self.opponent).__name__ == "BasicOpponent":
      a2 = self.opponent.act(ob2)
    else:
      with torch.no_grad():
        a2, _, _ = self.opponent.get_action(torch.Tensor(ob2).unsqueeze(0).to(self.device))
        a2 = a2.detach().cpu().numpy().squeeze()

    action2 = np.hstack([action, a2])
    return super().step(action2)

class HockeyEnv_HumanOppoent(h_env.HockeyEnv):
  def __init__(self, mode=h_env.Mode.NORMAL):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = h_env.HumanOpponent(self, player=2)
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)

  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)
  
class OpponentResetWrapper(gym.Wrapper):
  def __init__(self, env, opponent_sampler, episode_count):
    super().__init__(env)
    self.opponent_sampler = opponent_sampler
    self.episode_count = episode_count

  def reset(self, **kwargs):
    opponent = self.opponent_sampler.sample_opponent(self.episode_count.value)
    self.env.set_opponent(opponent)
    return self.env.reset(**kwargs)
  
class OpponentSampler():
  def __init__(self, self_play_len, custom_opponent_pool=[]):
    self.easy = h_env.BasicOpponent()
    self.hard = h_env.BasicOpponent(weak=False)
    self.self_play_pool = deque(maxlen=self_play_len)
    self.custom_opponent_pool = custom_opponent_pool

  def sample_opponent(self, global_episode):
    probs = self.get_probs(global_episode)
    choice = np.random.choice(["easy", "hard", "custom", "self"], p=probs)
    if choice == "easy":
      opponent = self.easy
    elif choice == "custom" and len(self.custom_opponent_pool) > 0:
        opponent = np.random.choice(self.custom_opponent_pool)
    elif choice == "self" and len(self.self_play_pool) > 0:
        opponent = np.random.choice(self.self_play_pool)
    else:
      opponent = self.hard #if custom or self dont exist, just use more hard basic enemies
    return opponent    
  
  def get_probs(self, global_episode):
    if global_episode < 1e4:
      probs = [1, 0, 0, 0]
    elif global_episode < 3e4:
      probs = [0.5, 0.4, 0, 0.1]
    elif global_episode < 4e4:
      probs = [0.2, 0.4, 0, 0.4]
    else:
      probs = [0.1, 0.4, 0, 0.5]
    return probs
         
  def add_self_play_opponent(self, frozen_actor):
    self.self_play_pool.append(frozen_actor)

class EpisodeCounter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value