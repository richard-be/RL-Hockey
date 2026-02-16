# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# NOTE: FIXED IMPORTS HERE 
from algorithm.buffers import ReplayBuffer
from algorithm.env import make_env, make_hockey_env, make_hockey_env_self_play, HockeyPlayer, HockeyEnv
from algorithm.evaluation import evaluate 
from algorithm.td3 import Actor, QNetwork

# NOTE: ADDED THESE IMPORTS 
from tqdm import tqdm 
import json
from hockey.hockey_env import HockeyEnv_BasicOpponent, Mode
from dataclasses import asdict

# NOTE: added here, adapted from rnd.py
from algorithm.rnd import RNDModel, RunningMeanStd
from algorithm.colored_noise import reset_noise

@dataclass
class Args:
    exp_name: str = "rnd_0_pn_0x2_rnd-lr_1e5_rnd-recompute_False_sp-reuse_True"
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "td3hockey"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HockeyOne-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    # NOTE: additional arguments added here: 
    """if the environment is hockey, it is not loaded via gym.make"""
    is_hockey: bool = True 
    """hockey specific arguments: the training mode"""
    hockey_mode: Mode = Mode.NORMAL
    """hockey specific arguments: whether to use a weak opponent"""
    weak_opponent: bool = False 
    is_self_play: bool = True
    self_play_initial_opponents = (("weak", 1200), ("strong", 1500)) # can disable strong opponent in self-play mode to only use it as validation opponent
    self_play_reuse_opponent_exp = False # add opponent's transition experience to replay buffer to increase sample efficiency of self-play training

    # colored noise parameters
    noise_type: str = "normal" # "normal" or "cn" (colored noise)
    noise_beta: float = 1.0
    noise_sigma: float = 0.2 # TODO 

    # TODO: is this necessary? 
    env_max_episode_steps: int = 252

    # rnd parameters 
    rnd_update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    rnd_int_coef: float = 0 # TODO 
    """coefficient of extrinsic reward"""
    rnd_ext_coef: float = 1
    """coefficient of intrinsic reward"""
    rnd_feature_size: int = 512
    # rnd_int_gamma: float = 0.99
    # """Intrinsic reward discount rate"""
    rnd_num_iterations_obs_norm_init: int = 50
    """number of iterations to initialize the observations normalization parameters"""
    rnd_max_intrinsic_reward: float = 1.0
    rnd_recompute_int_reward_on_replay: bool = False
    rnd_learning_rate: float = 1e-5
    
    config: str = None
    # NOTE: end of change


def load_yaml(path): 
    # Source - https://stackoverflow.com/a/1774043
    # Posted by Jonathan Holloway, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-02-16, License - CC BY-SA 4.0

    import yaml

    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.config is not None:
        config_args = load_yaml(args.config)
        # overwrite args with config args
        for key, value in config_args.items():
            setattr(args, key, config_args[key])

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    # NOTE: changed here to account for num_envs
    total_timesteps = args.total_timesteps # int(args.effective_timesteps / args.num_envs)
    learning_starts = args.learning_starts # int(args.effective_learning_starts / args.num_envs)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # NOTE: added this to keep args as json 
    args_dict = asdict(args) 
    args_dict["hockey_mode"] = args.hockey_mode.name
    json.dump(args_dict, open(f"runs/{run_name}/args.json", "w+"), indent=2)
    # END OF NOTE 

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # NOTE: changed env setup to support self-play
    if args.is_self_play:
        player = HockeyPlayer(None) # add actor after creating envs (depends on envs for obs/action space)

    def make_env_fn(idx, is_hockey=args.is_hockey, is_self_play=args.is_self_play): 
        if not is_hockey:
            return make_env(args.env_id, args.seed+idx, idx, args.capture_video, run_name)
        if is_self_play: 
            return make_hockey_env_self_play(args.seed+idx, idx, args.capture_video, run_name, player, initial_opponents=args.self_play_initial_opponents, mode=args.hockey_mode)
        else: 
            return make_hockey_env(args.seed+idx, idx, args.capture_video, run_name, mode=args.hockey_mode, weak_opponent=args.weak_opponent)

    envs = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(args.num_envs)])
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    envs.single_observation_space.dtype = np.float32

    def unwrap_env(env): 
        if isinstance(env, HockeyEnv): return env 
        else: return unwrap_env(env.env)  
    unwrapped_envs = [unwrap_env(env) for env in envs.env.envs]

    eval_envs = gym.vector.SyncVectorEnv([make_env_fn(i) for i in range(args.num_envs)])
    unwrapped_eval_envs = [unwrap_env(env) for env in eval_envs.envs]
    eval_envs.single_observation_space.dtype = np.float32
    # END OF NOTE 

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # NOTE: added here: add model to player for self-play
    if args.is_self_play:
        player.actor = actor

    # NOTE: added RND model and optimizer here
    # TODO: what should be the input and output dimensions of the RND model?
    # initialize model
    rnd_model = RNDModel(envs.single_observation_space.shape[0], args.rnd_feature_size).to(device)
    rnd_optimizer = optim.Adam(
        list(rnd_model.predictor.parameters()), 
        lr=args.rnd_learning_rate,
        eps=1e-5,
    )

    int_reward_rms = RunningMeanStd() # TODO: does this work? 
    obs_rms = RunningMeanStd(shape=(envs.single_observation_space.shape[0], )) # NOTE: adapted shape  
    # TODO: is this necessary? 
    # rewards = torch.zeros((total_timesteps, args.num_envs)).to(device)
    # curiosity_rewards = torch.zeros((total_timesteps, args.num_envs)).to(device)

    # END OF NOTE 

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False, #TODO: change??
    )
    start_time = time.time()

    # NOTE: CHANGED HERE: moved to seperate function
    def save_and_eval_model(current_step, n_eval_episodes = 1): 
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")

        if n_eval_episodes > 0: 
            # NOTE: changed evaluation
            # accum_rewards, n_won, n_lost = evaluate(eval_envs, actor, device, False, False)
            results = evaluate(eval_envs, unwrapped_eval_envs, n_eval_episodes, actor, is_self_play=args.is_self_play, unwrapped_train_envs=unwrapped_envs, device=device)
            for opponent_name, stats in results.items():
                writer.add_scalar(f"eval/{opponent_name}/acum_reward", stats["reward"], current_step)
                writer.add_scalar(f"eval/{opponent_name}/lose_rate", stats["lose_rate"], current_step) 
                writer.add_scalar(f"eval/{opponent_name}/win_rate", stats["win_rate"], current_step) 
                writer.add_scalar(f"eval/{opponent_name}/draw_rate", stats["draw_rate"], current_step)

    # NOTE: moved RND normalization to separate function: 
    def normalize_obs(obs, eps=1e-8, clamp_range=(-5.0, 5.0)): 
        if type(obs) == np.ndarray:
            obs = torch.from_numpy(obs).to(device)
        return ((obs - torch.tensor(obs_rms.mean, device=device, dtype=torch.float32)) / 
                        torch.sqrt(torch.tensor(obs_rms.var, device=device, dtype=torch.float32) + eps)).clamp(*clamp_range)
    
    def compute_intrinsic_reward(next_obs, update_obs_rms=True): 
        with torch.no_grad():
            rnd_next_obs = normalize_obs(next_obs) # NOTE: similified and moved to separate function
            predict_next_feature, target_next_feature = rnd_model(rnd_next_obs) # NOTE: removed mb_inds here!

            intrinsic_rewards = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).cpu().numpy()
        
        # Update running mean of intrinsic rewards
        if update_obs_rms:
            int_reward_rms.update(intrinsic_rewards)
        # Normalize intrinsic rewards and add to extrinsic rewards
        # TODO: normalize by mean as well => does this make sense? 
        # intrinsic_rewards = (intrinsic_rewards - int_reward_rms.mean) / np.sqrt(int_reward_rms.var + 1e-8)
        intrinsic_rewards /= np.sqrt(int_reward_rms.var + 1e-8)
        intrinsic_rewards = np.clip(intrinsic_rewards, 0, args.rnd_max_intrinsic_reward)
        
        return intrinsic_rewards
    
    # NOTE: TAKEN FROM RICHARD
    max_episode_length = args.env_max_episode_steps
    noise = np.zeros((args.num_envs, envs.single_action_space.shape[0], max_episode_length))
    for i in range(args.num_envs):
        noise = reset_noise(i, noise, args.noise_beta, max_episode_length, envs.single_action_space.shape[0])
    env_indices = np.arange(args.num_envs)
    episode_steps = np.zeros_like(env_indices)
    # NOTE: END OF RICHARD'S CODE 

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    action_clamp_low = torch.tensor(envs.single_action_space.low, device=device, dtype=torch.float32)
    action_clamp_high = torch.tensor(envs.single_action_space.high, device=device, dtype=torch.float32)


    for global_step in tqdm(range(total_timesteps), "Training agent"):
        # ALGO LOGIC: put action logic here
        if global_step < learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                # NOTE: added colored noise here 
                if args.noise_type == "normal": 
                    actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                    actions = actions.cpu().numpy()
                elif args.noise_type == "cn": 
                    actions = actions.cpu().numpy()
                    actions += args.noise_sigma * noise[env_indices, :, episode_steps]
                actions = actions.clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, extrinsic_rewards, terminations, truncations, infos = envs.step(actions)
        
        episode_steps += 1 # NOTE: ADDED HERE for noise

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos: 
            env_has_info = infos["_episode"]
            mean_episode_len = infos["episode"]['l'][env_has_info].mean()
            writer.add_scalar("charts/mean_episode_length", mean_episode_len, global_step)

        # NOTE: added here for RND
        # First update obs running means
        obs_rms.update(next_obs)
        if global_step < args.rnd_num_iterations_obs_norm_init: # wait for some data to accumulate in obs_rms before using it for normalization
            intrinsic_rewards = np.zeros_like(extrinsic_rewards)
        else: 
            # Compute curiosity rewards
            intrinsic_rewards = compute_intrinsic_reward(next_obs)

            # TODO: keep track of intrinsic and extrinsic rewards separately in the replay buffer?
            # total_rewards = args.rnd_ext_coef * extrinsic_rewards + args.rnd_int_coef * intrinsic_rewards

            writer.add_scalar("charts/intrinsic_reward", intrinsic_rewards.mean(), global_step) 
            writer.add_scalar("charts/extrinsic_reward", extrinsic_rewards.mean(), global_step) 
            # writer.add_scalar("charts/total_reward", total_rewards.mean(), global_step) 
        
        # END OF NOTE 

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # NOTE: added intrinsic rewards to replay buffer
        rb.add(obs, real_next_obs, actions, extrinsic_rewards, intrinsic_rewards, terminations, infos) 

        if args.self_play_reuse_opponent_exp: 
            # NOTE added re-using opponent experience in self-play
            assert "obs_agent_two" in infos and "action_agent_two" in infos and "reward_agent_two" in infos, "Missing opponent experience in infos"
            opponent_obs = infos["obs_agent_two"]
            opponent_actions = infos["action_agent_two"]
            opponent_extrinsic_rewards = infos["reward_agent_two"]
            # NOTE: used same intrinsic reward for opponent (next state is the same)
            rb.add(opponent_obs, real_next_obs, opponent_actions, opponent_extrinsic_rewards, intrinsic_rewards, terminations, infos)


        # NOTE: ADDED HERE for noise computation 
        for i, is_over in enumerate(terminations | truncations): 
            if is_over: 
                # print(f"env {i} over after {episode_steps[i]}")
                noise = reset_noise(i, noise, args.noise_beta, max_episode_length, envs.single_action_space.shape[0])
                episode_steps[i] = 0
        # END NOTE 

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            data = rb.sample(args.batch_size)

            # NOTE: added RND reward computation
            # rnd_next_obs = normalize_obs(data.next_observations)
            # predict_next_feature, target_next_feature = rnd_model(rnd_next_obs) # NOTE: removed mb_inds here!

            # curiosity_rewards = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).clamp(min=0, max=args.rnd_max_intrinsic_reward).detach()
            # END OF NOTE 

            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = torch.clamp((target_actor(data.next_observations) + clipped_noise), 
                    action_clamp_low, action_clamp_high
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

                # NOTE: added intrinsic reward here
                if args.rnd_recompute_int_reward_on_replay: 
                    data_intrinsic_rewards = torch.from_numpy(compute_intrinsic_reward(data.next_observations, False)).to(device)
                else: 
                    data_intrinsic_rewards = data.intrinsic_rewards
                data_total_rewards = args.rnd_ext_coef * data.extrinsic_rewards + args.rnd_int_coef * data_intrinsic_rewards
                
                next_q_value = data_total_rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # END OF NOTE 
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # NOTE: added RND loss here
            # TODO: update RND model every step here or directly after collecting transition? 
            rnd_next_obs = normalize_obs(data.next_observations) # NOTE: simplified here and moved to separate function
            predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs) # NOTE: removed mb_inds here!
            rnd_loss = F.mse_loss(
                predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
            ).mean(-1)

            mask = torch.rand(len(rnd_loss), device=device)
            mask = (mask < args.rnd_update_proportion).float()
            rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1], device=device, dtype=torch.float32))

            rnd_optimizer.zero_grad()
            rnd_loss.backward()
            rnd_optimizer.step()
            # end of NOTE 

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                
                writer.add_scalar("losses/rnd_loss", rnd_loss.item(), global_step)

                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                writer.add_scalar("charts/env_step", global_step*args.num_envs, global_step)
                
                writer.add_scalar("rnd/intrinsic_reward_replay", data_intrinsic_rewards.mean(), global_step)
                writer.add_scalar("rnd/extrinsic_reward_replay", data.extrinsic_rewards.mean(), global_step)
                writer.add_scalar("rnd/total_reward_replay", data_total_rewards.mean(), global_step)
                if args.rnd_recompute_int_reward_on_replay: 
                    writer.add_scalar("rnd/intrinsic_reward_replay_recompute_diff", (data.intrinsic_rewards - data_intrinsic_rewards).mean(), global_step)
            
            # NOTE: ADDED HERE
            if (global_step+1) % 1000 == 0: 
                print("Saving intermediate model")
                save_and_eval_model(global_step, 1)
                if args.is_self_play:
                    writer.add_scalar("self_play/player_elo", player.elo, global_step)
                    opponent_elo = np.mean([env.opponent.elo for env in unwrapped_envs])
                    writer.add_scalar("self_play/opponent_elo", opponent_elo, global_step)
                    n_opponents = np.mean([len(env.opponent_pool) for env in unwrapped_envs])
                    writer.add_scalar("self_play/n_opponents", n_opponents, global_step)
            # END NOTE 

    if args.save_model:
        save_and_eval_model(total_timesteps, 10) # TODO: add evaluation
        # NOTE: removed model upload here

    envs.close()
    writer.close()
    