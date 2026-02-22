import numpy as np 
import gymnasium as gym
from src.td3.algorithm.evaluation import load_actor

def run(env, agent, n_episodes=100, noise=0, render=True):
    rewards = []
    observations = []
    actions = []
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(2000):
            action = agent.act(state)
            state, reward, done, _trunc, _info = env.step(action)
            if render:
                env.render()

            observations.append(state)
            actions.append(action)
            ep_reward += reward
            if done or _trunc:
                break
        rewards.append(ep_reward)
        ep_reward = 0
    print(f'Mean reward: {np.mean(rewards)}')
    observations = np.asarray(observations)
    actions = np.asarray(actions)
    return observations, actions, reward


path = "td3:models/td3/HalfCheetah-v5/rnd_0x5-1__42__1771627022.model"
env = gym.make("HalfCheetah-v5", render_mode="human")
agent = load_actor(path, env)
run(env, agent, n_episodes=10)
