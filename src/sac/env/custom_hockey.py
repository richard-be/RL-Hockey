import numpy as np
import hockey.hockey_env as h_env
from gymnasium import spaces


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
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)
  
class HockeyEnv_Custom_CustomOpponent(HockeyEnv_Custom):
  def __init__(self, opponent, mode=h_env.Mode.NORMAL):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = opponent
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)
  