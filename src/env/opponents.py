import numpy as np 
from hockey_env import HockeyEnv, SCALE, MAX_ANGLE, Mode
from gymnasium import spaces
from gymnasium.envs.registration import register

class BasicOpponent():
  def __init__(self, weak=True, keep_mode=True):
    self.weak = weak
    self.keep_mode = keep_mode
    self.phase = np.random.uniform(0, np.pi)

  def act(self, obs, verbose=False):
    alpha = obs[2]
    p1 = np.asarray([obs[0], obs[1], alpha])
    v1 = np.asarray(obs[3:6])
    puck = np.asarray(obs[12:14])
    puckv = np.asarray(obs[14:16])
    # print(p1,v1,puck,puckv)
    target_pos = p1[0:2]
    target_angle = p1[2]
    self.phase += np.random.uniform(0, 0.2)

    time_to_break = 0.1
    if self.weak:
      kp = 0.5
    else:
      kp = 10
    kd = 0.5

    # if ball flies towards our goal or very slowly away: try to catch it
    if puckv[0] < 30.0 / SCALE:
      dist = np.sqrt(np.sum((p1[0:2] - puck) ** 2))
      # Am I behind the ball?
      if p1[0] < puck[0] and abs(p1[1] - puck[1]) < 30.0 / SCALE:
        # Go and kick
        target_pos = [puck[0] + 0.2, puck[1] + puckv[1] * dist * 0.1]
      else:
        # get behind the ball first
        target_pos = [-210 / SCALE, puck[1]]
    else:  # go in front of the goal
      target_pos = [-210 / SCALE, 0]
    target_angle = MAX_ANGLE * np.sin(self.phase)
    shoot = 0.0
    if self.keep_mode and obs[16] > 0 and obs[16] < 7:
      shoot = 1.0

    target = np.asarray([target_pos[0], target_pos[1], target_angle])
    # use PD control to get to target
    error = target - p1
    need_break = abs((error / (v1 + 0.01))) < [time_to_break, time_to_break, time_to_break * 10]
    if verbose:
      print(error, abs(error / (v1 + 0.01)), need_break)

    action = np.clip(error * [kp, kp / 5, kp / 2] - v1 * need_break * [kd, kd, kd], -1, 1)
    if self.keep_mode:
      return np.hstack([action, [shoot]])
    else:
      return action


class HumanOpponent():
  def __init__(self, env, player=1):
    import pygame
    self.env = env
    self.player = player
    self.a = 0

    if env.screen is None:
      env.render()

    self.key_action_mapping = {
      pygame.K_LEFT: 1 if self.player == 1 else 2,  # Left arrow key
      pygame.K_UP: 4 if self.player == 1 else 3,  # Up arrow key
      pygame.K_RIGHT: 2 if self.player == 1 else 1,  # Right arrow key
      pygame.K_DOWN: 3 if self.player == 1 else 4,  # Down arrow key
      pygame.K_w: 5,  # w
      pygame.K_s: 6,  # s
      pygame.K_SPACE: 7,  # space
    }

    print('Human Controls:')
    print(' left:\t\t\tleft arrow key left')
    print(' right:\t\t\tarrow key right')
    print(' up:\t\t\tarrow key up')
    print(' down:\t\t\tarrow key down')
    print(' tilt clockwise:\tw')
    print(' tilt anti-clockwise:\ts')
    print(' shoot :\tspace')

  def act(self, obs):
    import pygame
    keys = pygame.key.get_pressed()
    action = 0
    for key in self.key_action_mapping.keys():
      if keys[key]:
        action = self.key_action_mapping[key]
    return self.env.discrete_to_continous_action(action)


class HockeyEnv_BasicOpponent(HockeyEnv):
  def __init__(self, mode=Mode.NORMAL, weak_opponent=False):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = BasicOpponent(weak=weak_opponent)
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)
  
class HockeyEnv_CustomOpponent(HockeyEnv):
  def __init__(self, opponent, mode=Mode.NORMAL):
    super().__init__(mode=mode, keep_mode=True)
    self.opponent = opponent
    # linear force in (x,y)-direction, torque, and shooting
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)

try:
#   register(
#     id='Hockey-v0',
#     entry_point='hockey.hockey_env:HockeyEnv',
#     kwargs={'mode': 0}
#   )
  register(
    id='Hockey-One-v0',
    entry_point='hockey.hockey_env:HockeyEnv_BasicOpponent',
    kwargs={'mode': 0, 'weak_opponent': False}
  )
except Exception as e:
  print(e)
