import torch 
from src.sac.agent.sac import Actor
from src.sac.agent.sac import SoftQNetwork
import src.sac.env.custom_hockey as c_env
import gymnasium as gym

def make_env(seed, weak_opponent,  env_mode="NORMAL"):
    def thunk():
        env = c_env.HockeyEnv_Custom_BasicOpponent(env_mode, weak_opponent)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv(
        [make_env(1, True) for i in range(4)]
    )

device = torch.device("cpu")


actor = Actor(envs).to(device)
q_1 = SoftQNetwork(envs).to(device)
q_2 = SoftQNetwork(envs).to(device)

actor.load_state_dict(torch.load("models/sac/sac_2_1_1000000_1771781495.pkl", map_location=device))
q_1.load_state_dict(torch.load("models/sac/sac_0.0_False_3000000_1771891644_2500000_q.pkl", map_location=device))
q_2.load_state_dict(torch.load("models/sac/sac_2_1_3000000_1771844787_2500000_q.pkl", map_location=device))

torch.save((actor, q_1, q_2), "models/sac/combined_models.model")