import gymnasium as gym
from PIL import Image
import torch
import optparse
import os
from mbrl.util.common import load_hydra_cfg, create_one_dim_tr_model

from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac_pranz24 import SAC
import numpy as np
from hockey.hockey_env import BasicOpponent
import pathlib 
from mbrl.types import TransitionBatch
import time 

def eval_model(dynamics_model, state, action, next_state, reward, done, trunc, _info):
    state_labels = [
        "x pos player one",
        "y pos player one",
        "angle player one",
        "x vel player one",
        "y vel player one",
        "angular vel player one",
        "x player two",
        "y player two",
        "angle player two",
        "y vel player two",
        "y vel player two",
        "angular vel player two",
        "x pos puck",
        "y pos puck",
        "x vel puck",
        "y vel puck",
        "time left player has puck",
        "time left other player has puck"
    ]
    tb = TransitionBatch(
        state.reshape(1, -1), 
        action.reshape(1, -1), 
        next_state.reshape(1, -1), 
        np.array([reward]),
        np.array([done]), 
        np.array([trunc]),
    )

    model_score, _ = dynamics_model.eval_score(tb)
    model_score = model_score.mean(dim=0)[0]

    # if model_score.max() > 10: 
        # # NOTE: last item of model prediciton is reward => might be out of bounds here
        # max_err_idx = torch.argsort(model_score, descending=True)[:3]
        # print([(state_labels[i], model_score[i]) for i in max_err_idx])
        # time.sleep(1)


def load_dynamics_model(model_dir, env, cfg,):
    # because the original code uses torch.load without weights_only=True, replace here weights_only=False
    dynamics_model = create_one_dim_tr_model(cfg, env.observation_space.shape, env.action_space.shape)
    # dynamics_model.load(model_directory)

    dynamics_model.model.load_state_dict(torch.load(pathlib.Path(model_dir) / dynamics_model._MODEL_FNAME, weights_only=False)["state_dict"])
    if dynamics_model.input_normalizer: 
        dynamics_model.input_normalizer.load(model_dir)
    return dynamics_model

def get_latest_run_dir(results_dir):
    def get_highest_folder(directory): 
        sub_dirs = sorted([e.name for e in os.scandir(directory) if e.is_dir()], reverse=True)
        return directory+"/"+sub_dirs[0] if len(sub_dirs) > 0 else None   
    
    dir_most_recent_date = get_highest_folder(results_dir) 
    return get_highest_folder(dir_most_recent_date) if dir_most_recent_date is not None else None

def run_hockey_env(agent, env, dynamics_model, n_episodes, max_timesteps, save_gif, weak_opponent=False): 
    rewards = []
    opponent = BasicOpponent(weak=weak_opponent) 

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        
        obs, _ = env.reset()
        obs_agent2 = env.obs_agent_two()

        for _ in range(max_timesteps):
            env.render()
            a1 = agent.act(obs)
            a2 = opponent.act(obs_agent2)


            # print(torch.tensor(np.hstack([obs, a1])).repeat((7, 1)).shape)
            # model_pred1, model_pred2 = dynamics_model(torch.tensor(np.hstack([obs, a1])).repeat((7, 1)).to(torch.float))
            # print(model_pred1.shape)
            # print(model_pred2.shape)
            obs_old = obs 
            obs, r, d, t, _ = env.step(a1) #np.hstack([a1,a2]))    

            # tb = TransitionBatch(obs_old, a1, obs, np.array(r), d, t)
            # dynamics_model.eval_score(tb)

            ep_reward += r
            obs_agent2 = env.obs_agent_two()


            if save_gif:
                 img = env.render()
                 img = Image.fromarray(img)
                 img.save(f'./gif/{ep:02}-{t:03}.jpg')
            if d or t: break
        rewards.append(ep_reward)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    return rewards 


def run_gym_env(agent, env, dynamics_model, n_episodes, max_timesteps, save_gif): 
    rewards = []

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _info = env.reset()
        for t in range(max_timesteps):
            env.render()

            action = agent.act(state)

            env_update = env.step(action)
            eval_model(dynamics_model, state, action, *env_update)
            state, reward, done, trunc, _ = env_update

            ep_reward += reward
            if save_gif:
                 img = env.render()
                 img = Image.fromarray(img)
                 img.save(f'./gif/{ep:02}-{t:03}.jpg')
            if done or trunc: break
        rewards.append(ep_reward)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    return rewards 

def test():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env',action='store', type='string',
                        #  dest='env_name',default="HalfCheetah-v4",
                         dest='env_name',default="Hockey-One-v0",
                         help='Environment (default %default)')
    optParser.add_option('-d', '--dir', action='store', type='string',
                         dest='directory', default='outputs/mbpo/default',
                         help='Model train directory (default %default)')
    optParser.add_option('-r', '--run', action='store', dest='run',
                         default='latest',
                         help="Specify run (default %default)")
    optParser.add_option('-s', '--render',action='store_true',
                         default="True",
                         dest='render',
                         help='render Environment if given')
    optParser.add_option('-g', '--gif',action='store_true',
                         dest='gif',
                         help='render Environment into animaged gif')

    opts, args = optParser.parse_args()
    ############## Hyperparameters ##############
    env_name = opts.env_name
    # creating environment
    render_mode = "human" if opts.render else None
    save_gif = opts.gif
    if save_gif:
        if render_mode is not None:
            print("overwrite render-mode to image")
        render_mode = "rgb_array"
        import os
        os.makedirs('./gif', exist_ok=True)
        print("to get a gif for episode 1 run \"convert -delay 1x30 ./gif/01_* ep01.gif\"")

    is_hockey = "Hockey" in opts.env_name

    if is_hockey: 
        env = gym.make(env_name)
    else: 
        env = gym.make(env_name, render_mode = render_mode)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape
    max_timesteps = 300

    if opts.run == "latest": 
        model_directory = get_latest_run_dir(opts.directory+"/gym___"+opts.env_name)
    else: 
        model_directory = opts.directory + "/" + opts.run

    #############################################

    n_episodes = 10
    max_timesteps = 300

    print(model_directory)

    cfg = load_hydra_cfg(model_directory)
    agent = SACAgent(SAC(env.observation_space.shape[0], env.action_space, cfg.algorithm.agent.args))
    agent.sac_agent.load_checkpoint(model_directory+"/sac.pth")
    dynamics_model = load_dynamics_model(model_directory, env, cfg)

    # if is_hockey: 
    #     run_hockey_env(agent, env, dynamics_model, n_episodes, max_timesteps, save_gif)
    # else: 
    run_gym_env(agent, env, dynamics_model, n_episodes, max_timesteps, save_gif)

    env.close()


if __name__ == '__main__':
    test()
