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
from mbrl.models import ModelEnv
import time 
from algorithm.mbpo import model_env_sample
import imageio
from PIL import Image, ImageDraw
from util import get_latest_run_dir

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

def load_dynamics_model(model_dir, env, cfg,):
    # because the original code uses torch.load without weights_only=True, replace here weights_only=False
    dynamics_model = create_one_dim_tr_model(cfg, env.observation_space.shape, env.action_space.shape)
    # dynamics_model.load(model_directory)

    model_weights_path = pathlib.Path(model_dir) / dynamics_model._MODEL_FNAME
    if not model_weights_path.exists(): 
        print("Warning: dynamics model weights not found.")
        return dynamics_model
    
    dynamics_model.model.load_state_dict(torch.load(model_weights_path, weights_only=False, map_location=torch.device('cpu'))["state_dict"])
    if dynamics_model.input_normalizer: 
        dynamics_model.input_normalizer.load(model_dir) #, map_location=torch.device('cpu'))
    return dynamics_model

def run_gym_env(agent, env, model_env, dynamics_model, n_episodes, max_timesteps, save_gif, out_dir=None): 
    rewards = []
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
            if not save_gif:
                env.render()

            action = agent.act(state)

            env_update = env.step(action)
            # pred_state = eval_model(dynamics_model, state, action, *env_update)
            pred_next_obs, pred_rewards, pred_dones, model_state, alea_uncertainties, epist_uncertainties = model_env_sample(
                model_env.dynamics_model, model_env.termination_fn, model_env._rng, action.reshape(1, -1), model_state, 
            )
            aleatoric_uncertainties.append(alea_uncertainties[0].item())
            epistemic_uncertainties.append(epist_uncertainties[0].item())

            state, reward, done, trunc, _ = env_update
            model_state["obs"] = state.reshape(1, -1)

            ep_reward += reward
            if save_gif:
                anim[0].append(Image.fromarray(env.render(mode="rgb_array")))
                rews[0].append(reward)
                
                env.set_state(pred_next_obs[0].astype(np.float64))
                
                anim[1].append(Image.fromarray(env.render(mode="rgb_array")))
                rews[1].append(pred_rewards[0, 0])
                
                env.set_state(state)
            if done or trunc: break

        if save_gif: 
            create_gif(anim[0], anim[1], rews[0], rews[1], aleatoric_uncertainties, epistemic_uncertainties, f"{out_dir}/test_ep{ep}.gif")

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
                         default="False",
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
        # import os
        # os.makedirs(GIF_PATH+"/env", exist_ok=True)
        # os.makedirs(GIF_PATH+"/model", exist_ok=True)
        # print("to get a gif for episode 1 run \"convert -delay 1x30 ./gif/01_* ep01.gif\"")

    is_hockey = "Hockey" in opts.env_name

    if is_hockey: 
        env = gym.make(env_name)
        model_env = gym.make(env_name)
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

    n_episodes = 10
    max_timesteps = 300

    print(model_directory)

    cfg = load_hydra_cfg(model_directory)
    cfg.device = "cpu"
    agent = SACAgent(SAC(env.observation_space.shape[0], env.action_space, cfg.algorithm.agent.args))
    load_checkpoint(agent.sac_agent, model_directory+"/sac.pth", evaluate=True)
    dynamics_model = load_dynamics_model(model_directory, env, cfg)
    model_env = ModelEnv(env, dynamics_model, termination_fn, None, generator=torch_generator)

    run_gym_env(agent, env, model_env, dynamics_model, n_episodes, max_timesteps, save_gif, model_directory)
    env.close()


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
