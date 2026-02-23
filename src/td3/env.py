from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode, HockeyEnv, BasicOpponent
import gymnasium as gym
from gymnasium import spaces
import numpy as np  
from typing import List
import torch 
from .algorithm.models import Actor
import pygame

# NOTE: original env creation function 
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env.enabled = False
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env, stats_key="episode_stats")
        env.action_space.seed(seed)
        return env
    return thunk
# END OF NOTE 

# NOTE: ADDED OWN CODE HERE
# To support video 
class RenderWrapper(gym.Wrapper):
    
    def __init__(self, env, render_mode="rgb_array"):
        super().__init__(env)
        env.render_mode = render_mode

        self.has_opponent = hasattr(env, "opponent")
        if self.has_opponent:
            pygame.font.init()
            self.font = pygame.font.SysFont('Courier New', 15)
            self.last_opponent = env.opponent
            self.opponent_changed = False

    def reset(self, **kwargs): 
        return_val = super().reset(**kwargs)
        if self.has_opponent and self.last_opponent != self.env.opponent:
            self.opponent_changed = True

        return return_val

    def render(self, **kwargs): 
        return_val = self.env.render(mode=self.env.render_mode)
        
        if self.has_opponent:
            opponent = self.env.opponent
            text_surface = self.font.render(f"Opponent {opponent.player_num}: {opponent.player_name}", False, (0, 0, 0))
            self.env.screen.blit(text_surface, (0, 0))
            pygame.display.flip()

        return return_val

def wrap_hockey_env(env, seed, idx, capture_video=False, run_name=None):
    env = RenderWrapper(env, render_mode="rgb_array" if capture_video else "human")

    if capture_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.enabled = False
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


def compute_new_elo(player_elo, opponent_elo, outcome, K=16):
  """Compute new Elo for a player based on outcome and opponent's Elo."""
  # NOTE: this is based on https://huggingface.co/learn/deep-rl-course/unit7/self-play#the-elo-score-to-evaluate-our-agent
  
  expected_score = 1 / (1 + 10**((opponent_elo - player_elo) / 400))
  actual_score = {1: 1, -1: 0, 0: 0.5}[outcome]
  return player_elo + K * (actual_score - expected_score)


class HockeyPlayer():
    def __init__(self, actor: Actor, player_num, elo=1200, player_name=""): 
        self.actor = actor
        self.player_num = player_num
        self.elo = elo  
        self.player_name = player_name

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
                self.opponent_pool.append(HockeyPlayer(OpponentActor(weak=True), elo=opponent_elo, player_num=len(self.opponent_pool), player_name="Weak"))
            elif opponent_type == "strong":
                self.opponent_pool.append(HockeyPlayer(OpponentActor(weak=False), elo=opponent_elo, player_num=len(self.opponent_pool), player_name="Strong"))
            else:
                raise ValueError(f"Invalid opponent type: {opponent_type}")
        self.player = player
        self.opponent_selection_temp = 200 # temperature that allows for random opponents
        self.diff_elo_to_add_opponent = 150 # if player has improved by this much since last opponent was added, add new opponent to pool
        self.last_player_elo_when_adding_opponent = self.player.elo
        super().__init__(mode=mode, weak_opponent=False) 


    def reset(self, one_starting=None, mode=None, seed=None, options=None): 
        obs, info = super().reset(one_starting, mode, seed, options)
        self.opponent = self.select_opponent()

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

        if term: 
            # update statistics if episode is over 
            outcome = self.winner
            # update elos: 

            player_elo = self.player.elo
            opponent_elo = self.opponent.elo

            self.player.update_elo(outcome, opponent_elo)
            # self.opponent.update_elo(- outcome, player_elo)

            if self.player.elo > self.last_player_elo_when_adding_opponent + self.diff_elo_to_add_opponent: 
                # if player has improved significantly since last opponent was added, add new opponent to pool
                self.add_actor_to_opponent_pool(self.player.actor.clone(), self.player.player_name, self.player.elo)
                self.last_player_elo_when_adding_opponent = self.player.elo

        return obs, reward, term, trunc, info

    def select_opponent(self, min_prob=0.1) -> HockeyPlayer:
        elos = np.array([opponent.elo for opponent in self.opponent_pool])
        elo_diffs = np.abs(elos - self.player.elo)
        selection_probabilities = torch.softmax(-torch.tensor(elo_diffs) / (self.opponent_selection_temp), dim=0).numpy()

        selection_probabilities = min_prob + (1 - min_prob) * selection_probabilities
        selection_probabilities /= selection_probabilities.sum()  # normalize to sum to 1
        return np.random.choice(self.opponent_pool, p=selection_probabilities)
    

    def add_actor_to_opponent_pool(self, actor, player_name, elo=1200):
        self.opponent_pool.append(HockeyPlayer(actor, player_num=len(self.opponent_pool), elo=elo, player_name=player_name))

class OpponentActor(): 
    """A wrapper to use BasicOpponent as an actor for self-play training"""
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

def make_hockey_env(seed, idx, capture_video, run_name, mode=Mode.NORMAL, weak_opponent=False):
    def thunk(): 
        env = HockeyEnv_BasicOpponent(mode=mode, weak_opponent=weak_opponent)
        return wrap_hockey_env(env, seed, idx, capture_video=capture_video, run_name=run_name)
    return thunk

def make_hockey_env_self_play(seed, idx, capture_video, run_name, player: HockeyPlayer, initial_opponents=[("weak", 1200), ("strong", 1500)], mode=Mode.NORMAL):
    def thunk(): 
        env = HockeyEnv_CustomPlayers(player, initial_opponents=initial_opponents, mode=mode)
        return wrap_hockey_env(env, seed, idx, capture_video=capture_video, run_name=run_name)
    return thunk


def make_hockey_eval_env(seed, idx=0, mode=Mode.NORMAL):
    def thunk():
        env = HockeyEnv_BasicOpponent(mode=mode)
        env.action_space.seed(seed)
        return wrap_hockey_env(env, seed, idx, capture_video=False, run_name=None)
    return thunk

def load_actor(run_name: str, env, device="cpu"):
    from .algorithm.models import Actor as TD3Actor
    from ..sac.agent.sac import Actor as SACActor

    def add_act_method(actor):
        def act(self, obs): 
            with torch.no_grad(): 
                obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(next(self.parameters()).device)
                action = self.forward(obs)
                if isinstance(action, tuple):
                    action = action[0]
                return action.cpu().numpy()[0]
        if not hasattr(actor, "act"):
            setattr(actor, "act", act.__get__(actor, actor.__class__))

    assert ":" in run_name, "Weight path must be in format <algorithm>:<path>"
    algorithm, path = run_name.split(":", 1)
    
    actor_types = {
        "sac": SACActor,
        "td3": TD3Actor,
        "crq": None, 
    }

    assert algorithm in actor_types, f"Unknown algorithm {algorithm}. Supported algorithms: {list(actor_types.keys())}"
    actor_class = actor_types[algorithm]
    if actor_class is None:
        raise ValueError(f"Unknown external model type: {algorithm}")
    

    actor_state = torch.load(path, map_location=device)
    # TODO: this could be done better by creating different load functions for each model 
    if isinstance(actor_state, tuple) and len(actor_state) == 3:   
        # if it's a td3 model, just load the actor state
        actor_state = actor_state[0] 

    print(f"Loaded model from {path}")
    
    if not hasattr(env, "single_observation_space"):
        env.single_observation_space = env.observation_space
    if not hasattr(env, "single_action_space"):
        env.single_action_space = env.action_space

    actor = actor_class(env)
    actor.load_state_dict(actor_state)
    add_act_method(actor)

    return actor.to(device)

def disable_gradients(actor):
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False
    return actor