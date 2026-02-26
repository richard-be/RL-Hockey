import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
import heapq
import torch
from itertools import accumulate

"""Different hockey env variants for reward tuning and manual opponent setting"""

class HockeyEnv_Custom(h_env.HockeyEnv):
  def get_reward(self, info):
    r = self._compute_reward()
    r += info["reward_closeness_to_puck"]
    #r += info["reward_touch_puck"]
    #r += info["reward_puck_direction"]
    return float(r)
  
class HockeyEnv_Custom_BasicOpponent(HockeyEnv_Custom):
  """Env for playing against basic opponent"""
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
  """Env for playing against custom opponent that can be reset for each episode"""
  def __init__(self, opponent, device, mode=h_env.Mode.NORMAL):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = opponent
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = gym.spaces.Box(-1, +1, (4,), dtype=np.float32)
    self.device = device

  def set_opponent(self, opponent):
    self.opponent = opponent

  def step(self, action):
    ob2 = self.obs_agent_two()
    if type(self.opponent).__name__ == "BasicOpponent":
      a2 = self.opponent.act(ob2)
    else:
      with torch.no_grad():
        a2 = self.opponent.act(np.array(ob2))

    action2 = np.hstack([action, a2])
    return super().step(action2)

class HockeyEnv_HumanOppoent(h_env.HockeyEnv):
  """env for playing against human opponent for fun purposes"""
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
  

