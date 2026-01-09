import numpy as np
import hockey.hockey_env as h_env
from gymnasium import spaces

class HockeyEnv_CustomOpponent(h_env.HockeyEnv):
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