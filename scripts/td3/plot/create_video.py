# NOTE: This script was only for testing; the function to plot q-values was written using ChatGPT, but is not used in the final project.

import torch 
from src.td3.env import make_hockey_eval_env, load_actor
from src.td3.algorithm.models import Actor, QNetwork
from hockey.hockey_env import BasicOpponent, HockeyEnv_BasicOpponent

# path = "models/td3/HockeyOne-v0/final/no_sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_1__42__1771959759/1000000.model"
# path = "models/td3/HockeyOne-v0/intrinsic_rewards/rnd_1-0_sp_1__42__1771854378/1000000.model"
# path = "models/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771854378/1000000.model"
# path = "models/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771854378/100000.model"
# path = "models/td3/HockeyOne-v0/intrinsic_rewards/rnd_0x5-1_sp_1__42__1771854378/100000.model"
# path = "models/td3/HockeyOne-v0/final/no_sp/rnd/rnd_0x5-1_pr_0__42__1771959759/1000000.model"
# path = "models/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_0_sp_1__42__1772009409/1000000.model"
path = "models/td3/HockeyOne-v0/final/no_sp/rnd/rnd_1-0_pr_0__42__1772039473/1000000.model"
# opponent_path = "td3:models/td3/HockeyOne-v0/final/sp/pn/pn_0x2_pr_1_pr-intr-factor_0_sp_1__42__1772009398/1000000.model"
# path = "models/td3/HockeyOne-v0/final/sp/rnd/rnd_0x5-1_pr_1_pr-intr-factor_1_sp_1__42__1772009409/1000000.model"
out_path = "rnd_exploration_only"

n_episodes = 10
render_mode = "rgb_array"
device = "cpu" 
# env = HockeyEnv_BasicOpponent()
env = make_hockey_eval_env(1, capture_video=True, run_name=out_path)()
# env.opponent = load_actor(opponent_path, env, device)

actor_state, q1_state, q2_state = torch.load(path, map_location=device)

if not hasattr(env, "single_observation_space"):
    env.single_observation_space = env.observation_space
if not hasattr(env, "single_action_space"):
    env.single_action_space = env.action_space

actor = Actor(env)
actor.load_state_dict(actor_state)
actor.eval()

qf1 = QNetwork(env) 
qf1.load_state_dict(q1_state)
qf1.eval()

qf2 = QNetwork(env) 
qf2.load_state_dict(q2_state)
qf2.eval()

# env.opponent = actor

for i in range(n_episodes):
    obs, _ = env.reset(seed=42+i)
    done = False 
    while not done: 
        action = actor.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        frame = env.render()