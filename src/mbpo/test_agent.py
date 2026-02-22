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
from mbrl.types import TransitionBatch
from mbrl.models import ModelEnv
import time 
from algorithm.mbpo import model_env_sample
import imageio
from PIL import Image, ImageDraw
from util import get_latest_run_dir, load_dynamics_model
from algorithm.hockey_connector import get_hockey_reward

def create_gif(truth_images, vague_images, true_rews, pred_rews, alea_uncerts, epist_uncerts, out_path):
    with imageio.get_writer(out_path, fps=20) as writer:
        for truth, vague, true_rew, pred_rew, alea_uncert, epist_uncert in zip(truth_images, vague_images, true_rews, pred_rews, alea_uncerts, epist_uncerts):
            truth = truth.convert("RGBA")
            vague = vague.convert("RGBA")
        
            vague.putalpha(int(0.3 * 255))
            truth.paste(vague, (0, 0), mask = vague)
            draw = ImageDraw.Draw(truth)
            draw.text((50, 20), f"True rew: {true_rew:5.5f}; Pred rew: {pred_rew:5.5f}")
            draw.text((50, 450), f"Alea uncert: {alea_uncert:5.5f}; Epist uncert: {epist_uncert:5.5f}")

            # frames.append(truth)
            writer.append_data(np.array(truth))
    print("Saved to", out_path)

import matplotlib.pyplot as plt

def plot_reward_prediction_diagnostics(true_rewards, predicted_rewards, bins=50):
    true_rewards = np.asarray(true_rewards)
    predicted_rewards = np.asarray(predicted_rewards)

    assert true_rewards.shape == predicted_rewards.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. True vs Predicted scatter ---
    ax = axes[0]
    ax.scatter(true_rewards, predicted_rewards, s=5, alpha=0.4)
    min_r = min(true_rewards.min(), predicted_rewards.min())
    max_r = max(true_rewards.max(), predicted_rewards.max())
    ax.plot([min_r, max_r], [min_r, max_r])  # y = x reference
    ax.set_xlabel("True reward")
    ax.set_ylabel("Predicted reward")
    ax.set_title("True vs Predicted Reward")

    # --- 2. Reward distributions ---
    ax = axes[1]
    ax.hist(true_rewards, bins=bins, alpha=0.5, label="True")
    ax.hist(predicted_rewards, bins=bins, alpha=0.5, label="Predicted")
    ax.set_title("Reward Distributions")
    ax.legend()

    # --- 3. Absolute error vs true reward ---
    ax = axes[2]
    abs_error = np.abs(true_rewards - predicted_rewards)
    ax.scatter(true_rewards, abs_error, s=5, alpha=0.4)
    ax.set_xlabel("True reward")
    ax.set_ylabel("|Prediction error|")
    ax.set_title("Absolute Error vs True Reward")

    # plt.tight_layout()
    # plt.show()


# def eval_model(dynamics_model, state, action, next_state, reward, done, trunc, _info):
#     # NOTE: reward might also be at index 1 => check again!
#     state_labels = [
#         "x pos player one",
#         "y pos player one",
#         "angle player one",
#         "x vel player one",
#         "y vel player one",
#         "angular vel player one",
#         "x player two",
#         "y player two",
#         "angle player two",
#         "y vel player two",
#         "y vel player two",
#         "angular vel player two",
#         "x pos puck",
#         "y pos puck",
#         "x vel puck",
#         "y vel puck",
#         "time left player has puck",
#         "time left other player has puck",
#         "reward" 
#     ]
#     tb = TransitionBatch(
#         state.reshape(1, -1), 
#         action.reshape(1, -1), 
#         next_state.reshape(1, -1), 
#         np.array([reward]),
#         np.array([done]), 
#         np.array([trunc]),
#     )

#     # model_score, _ = dynamics_model.eval_score(tb)
#     # print("Variance of predictions", uncertainty)
#     # if model_score.max() > 10: 
#     #     max_err_idx = torch.argsort(model_score, descending=True)[:3]
#     #     print("".join([f"\n\t{(state_labels[i], model_score[i].item(), uncertainty[i].item())}" for i in max_err_idx]))
#         # time.sleep(1)


#     with torch.no_grad():
#         model_in, target = dynamics_model._process_batch(tb)      
#         output, output_variance = dynamics_model.model.forward(model_in, use_propagation=False)  

#     assert dynamics_model.target_is_delta
#     assert len(dynamics_model.no_delta_list) == 0

#     pred_output = output.mean(dim=0).reshape(-1).numpy().astype(np.float64)

#     # target is actually detla of state to target, i.e. next_state - state 
#     # => get next state by adding state ([:-1] because reward is predicted at last idx)
#     pred_output = pred_output[:-1] + state 

#     # output_variance.shape: [7, 1, 19]
#     # ! output variance is actually log variance !
#     return pred_output

def run_gym_env(agent, env, model_env, dynamics_model, render_mode, n_episodes, max_timesteps, save_gif, out_dir=None): 
    rewards = []
    n_won = 0 

    predicted_rewards = []
    real_rewards = []

    fake_env2 = gym.make("Hockey-One-v0")

    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state, _info = env.reset()
        model_state = model_env.reset(
            initial_obs_batch=state.reshape(1, -1),
            return_as_np=True,
        )

        anim = [[], []]
        rews = [[], []]
        aleatoric_uncertainties = []
        epistemic_uncertainties = []
        
        for t in range(max_timesteps):
            if not save_gif and render_mode is not None:
                env.render()

            action = agent.act(state)

            env_update = env.step(action)

            pred_next_obs, pred_rewards, pred_dones, model_state, alea_uncertainties, epist_uncertainties = model_env_sample(
                model_env, action.reshape(1, -1), model_state, 
            )
            aleatoric_uncertainties.append(alea_uncertainties[0].item())
            epistemic_uncertainties.append(epist_uncertainties[0].item())

            state, reward, done, trunc, info = env_update
            model_state["obs"] = state.reshape(1, -1)
            if pred_rewards.item() == 10 or  pred_rewards.item() == -10:
                print("predicted goal", t) 

            if done: 
                if not reward == 0: 
                    if pred_rewards.item() == 0: 
                        print("MISSED GOAL")
                        print("done at ", t)
                        print("reward IS", reward)
                        print("predicted", pred_rewards.item())
                        # input("CONTINUE?")
                    # re_pred = model_env.reward_fn(None, torch.tensor(pred_next_obs))
                    # print("repredicted", re_pred.item())

            # print("predicted reward error", reward - pred_rewards[0, 0])

            ep_reward += reward
            predicted_rewards.append(pred_rewards[0, 0])
            real_rewards.append(reward)

            if save_gif:
                anim[0].append(Image.fromarray(env.render(mode="rgb_array")))
                rews[0].append(reward)
                
                env.set_state(pred_next_obs[0].astype(np.float64))
                
                anim[1].append(Image.fromarray(env.render(mode="rgb_array")))
                rews[1].append(pred_rewards[0, 0])
                min
                
                env.set_state(state)
            if done: 
                if "winner" in info: 
                    n_won += (info["winner"] == 1) 
                break 
            if trunc: break

        if save_gif: 
            create_gif(anim[0], anim[1], rews[0], rews[1], aleatoric_uncertainties, epistemic_uncertainties, f"{out_dir}/test_ep{ep}.gif")

        rewards.append(ep_reward)
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    
    print("predicted rewards:", "min", np.min(predicted_rewards),"mean", np.mean(predicted_rewards), "max", np.max(predicted_rewards))
    print("real rewards:", "min", np.min(real_rewards),"mean", np.mean(real_rewards), "max", np.max(real_rewards))

    plot_reward_prediction_diagnostics(real_rewards, predicted_rewards)
    return rewards, n_won

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
                         dest='render',
                         help='render Environment if given')
    optParser.add_option('-g', '--gif',action='store_true',
                         dest='gif',
                         help='render Environment into animaged gif')
    
    optParser.add_option('-n', '--n_episodes',action='store',
                         default=10, 
                         dest='n_episodes',
                         help='number of episodes to run')

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
        # import os
        # os.makedirs(GIF_PATH+"/env", exist_ok=True)
        # os.makedirs(GIF_PATH+"/model", exist_ok=True)
        # print("to get a gif for episode 1 run \"convert -delay 1x30 ./gif/01_* ep01.gif\"")

    is_hockey = "Hockey" in opts.env_name

    if is_hockey: 
        env = gym.make(env_name)
        env.observation_space.dtype = np.float64
        # model_env = gym.make(env_name)
    else: 
        env = gym.make(env_name, render_mode = render_mode)
        # model_env = gym.make(env_name, render_mode = render_mode)

    def termination_fn(x, y): 
        return torch.zeros(1) 
    
    torch_generator = torch.Generator(device="cpu")
    # if cfg.seed is not None:
    #     torch_generator.manual_seed(cfg.seed)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape
    max_timesteps = 300

    if opts.run == "latest": 
        model_directory = get_latest_run_dir(opts.directory+"/gym___"+opts.env_name)
    else: 
        model_directory = f"{opts.directory}/gym___{opts.env_name}/{opts.run}"

    #############################################

    n_episodes = int(opts.n_episodes)
    max_timesteps = 300

    cfg = load_hydra_cfg(model_directory)
    cfg.device = "cpu"
    agent = SACAgent(SAC(env.observation_space.shape[0], env.action_space, cfg.algorithm.agent.args))
    load_checkpoint(agent.sac_agent, model_directory+"/sac.pth", evaluate=True)
    dynamics_model = load_dynamics_model(model_directory, env, cfg)


    # fake_env = gym.make("Hockey-One-v0")
    # fake_env.reset()
    
    # def reward_fn(actions, observs): 
    #     def get_reward(i): 
    #         fake_env.reset()
    #         fake_env.set_state(observs[i].numpy().astype(np.float64))
    #         _, reward,  _, _ , _ = fake_env.step(actions[i])
    #         return reward
    #     return torch.tensor([get_reward(i) for i in range(len(observs))]).unsqueeze(1)

    # model_env = ModelEnv(env, dynamics_model, termination_fn, reward_fn, generator=torch_generator)

    reward_fn = get_hockey_reward() if True else None 
    model_env = ModelEnv(env, dynamics_model, termination_fn, reward_fn, generator=torch_generator)

    print("render:", render_mode)
    print(opts.render)

    rewards, n_won = run_gym_env(agent, env, model_env, dynamics_model, render_mode, n_episodes, max_timesteps, save_gif, model_directory)
    env.close()
    print(f"Won {n_won} of {n_episodes} ({(n_won/n_episodes):.3f})")

def load_checkpoint(self, ckpt_path, evaluate=False):
    print("Loading models from {}".format(ckpt_path))
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()


if __name__ == '__main__':
    test()
