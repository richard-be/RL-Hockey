from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode, HockeyEnv, BasicOpponent
import gymnasium as gym
from gymnasium import spaces
import numpy as np  
from typing import List
import torch 
from .td3 import Actor

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
    assert outcome in [1, -1, 0], "Invlaid outcome for Elo update"
    self.elo = compute_new_elo(self.elo, opponent_elo, outcome)

class HockeyEnv_CustomPlayers(HockeyEnv_BasicOpponent):
    def __init__(self, player: HockeyPlayer, initial_opponents=[("weak", 1200), ("strong", 1500)], mode=Mode.NORMAL):        
        self.opponent_pool = []
        assert len(initial_opponents) > 0, "At least one initial opponent must be provided"
        for opponent_type, opponent_elo in initial_opponents:
            if opponent_type == "weak":
                self.opponent_pool.append(HockeyPlayer(OpponentActor(weak=True), elo=opponent_elo))
            elif opponent_type == "strong":
                self.opponent_pool.append(HockeyPlayer(OpponentActor(weak=False), elo=opponent_elo))
            else:
                raise ValueError(f"Invalid opponent type: {opponent_type}")
            
        self.player = player
        self.opponent_selection_temp = 100 # temperature that allows for random opponents TODO: find parameter 
        self.diff_elo_to_add_opponent = 100 # if player has improved by this much since last opponent was added, add new opponent to pool; TODO: find parameter
        self.last_player_elo_when_adding_opponent = self.player.elo

        super().__init__(mode=mode, weak_opponent=False) 


    def reset(self, one_starting=None, mode=None, seed=None, options=None): 
        obs, info = super().reset(one_starting, mode, seed, options)
        self.opponent = self.select_opponent()
        # self.level -= 1 
        # self.next_level()

        return obs, info

    def step(self, action):
        # taken form HockeyEnv_BasicOpponent.step(): 
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])

        obs, reward, term, trunc, info = super().step(action2)
        info["obs_agent_two"] = ob2
        info["action_agent_two"] = a2
        info["reward_agent_two"] = -reward

        # obs, reward, term, trunc, info = super().step(action)
        # todo: truncate if n steps is too long? => currently there will never be draw because trunc is always false and term only if winner
        if term: 
            # update statistics if episode is over 
            outcome = self.winner
            # update elos: 

            player_elo = self.player.elo
            opponent_elo = self.opponent.elo

            self.player.update_elo(outcome, opponent_elo)
            self.opponent.update_elo(- outcome, player_elo)

            if self.player.elo > self.last_player_elo_when_adding_opponent + self.diff_elo_to_add_opponent: 
                # if player has improved significantly since last opponent was added, add new opponent to pool
                self.add_actor_to_opponent_pool(self.player.actor.clone(), self.player.elo)
                self.last_player_elo_when_adding_opponent = self.player.elo

        return obs, reward, term, trunc, info

    def select_opponent(self) -> HockeyPlayer:
        elos = np.array([opponent.elo for opponent in self.opponent_pool])
        elo_diffs = np.abs(elos - self.player.elo)
        selection_probabilities = torch.softmax(-torch.tensor(elo_diffs) / (self.opponent_selection_temp), dim=0).numpy()
        return np.random.choice(self.opponent_pool, p=selection_probabilities)
    

    def add_actor_to_opponent_pool(self, actor, elo=1200):
        self.opponent_pool.append(HockeyPlayer(actor, elo))

class OpponentActor(): 
    def __init__(self, weak: bool, keep_mode=True): 
        self.opponent = BasicOpponent(weak)
    def eval(self): 
        pass
    def train(self): 
        pass 
    def act(self, obs_batch): 
        if len(obs_batch.shape) == 1: 
            return self.opponent.act(obs_batch)
         
        actions = [self.opponent.act(obs) for obs in obs_batch]
        return np.stack(actions).reshape(obs_batch.shape[0], -1)

def make_hockey_env(seed, idx, capture_video, run_name, max_episode_steps=None, mode=Mode.NORMAL, weak_opponent=False):
    def thunk(): 
        env = HockeyEnv_BasicOpponent(mode=mode, weak_opponent=weak_opponent)
        return wrap_hockey_env(env, seed, idx, capture_video=capture_video, run_name=run_name, max_episode_steps=max_episode_steps)
    return thunk

def make_hockey_env_self_play(seed, idx, capture_video, run_name, player: HockeyPlayer, initial_opponents=[("weak", 1200), ("strong", 1500)], max_episode_steps=None, mode=Mode.NORMAL):
    def thunk(): 
        env = HockeyEnv_CustomPlayers(player, initial_opponents=initial_opponents, mode=mode)
        return wrap_hockey_env(env, seed, idx, capture_video=capture_video, run_name=run_name, max_episode_steps=max_episode_steps)
    return thunk


def make_hockey_eval_env(seed, mode=Mode.NORMAL):
    def thunk():
        env = HockeyEnv_BasicOpponent(mode=mode)
        env.action_space.seed(seed)
        return env
    return thunk