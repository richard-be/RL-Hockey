# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast, Dict, Tuple

import gymnasium as gym
import hydra.utils
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
from omegaconf import OmegaConf
import imageio
from mbrl.models import OneDTransitionRewardModel

# NOTE: replaced here! 
from .re_implementations import model_env_sample, step_env_and_add_to_buffer, reset_noise
from .custom_sac import SAC
from .seperate_transition_reward_model import create_seperate_transition_reward_model
from tqdm import tqdm
from .hockey_connector import get_hockey_reward

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]
    # self.goal_player_1 = self._create_goal((W / 2 - 245 / SCALE - 10 / SCALE, H / 2), poly)
    # self.goal_player_2 = self._create_goal((W / 2 + 245 / SCALE + 10 / SCALE, H / 2), poly)
    # def _create_goal(self, position, poly):


    # if self.env.goal_player_2 == contact.fixtureA.body or self.env.goal_player_2 == contact.fixtureB.body:
    #   if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
    #     if self.verbose:
    #       print('Player 1 scored')
    #     self.env.done = True
    #     self.env.winner = 1
    # if self.env.goal_player_1 == contact.fixtureA.body or self.env.goal_player_1 == contact.fixtureB.body:
    #   if self.env.puck == contact.fixtureA.body or self.env.puck == contact.fixtureB.body:
    #     if self.verbose:
    #       print('Player 2 scored')
    #     self.env.done = True
    #     self.env.winner = -1

def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    hparams: dict, 
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    print("rollout horizon:", rollout_horizon)

    n_rejected_predictions = 0
    stats_alea_uncertainties = [] 
    stats_epis_uncertainties = [] 

    factor_epist_uncert_reward_bonus = hparams.get("factor_epist_uncert_rw_bonus", 0)
    clip_epist_uncert_min, clip_epist_uncert_max = hparams.get("epist_uncert_clip_range", (0, 5))
    max_alea_uncert = hparams.get("max_alea_uncert", float("inf"))

    for i in range(rollout_horizon):
        # NOTE: could use model's uncertainty here to discard samples 
        # or terminate individual horizons early where there is uncertainty 
        action = agent.act(obs, sample=sac_samples_action, batched=True)


        # NOTE: replaced here!
        # pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
        #     action, model_state, sample=True
        # )
        pred_next_obs, pred_rewards, pred_dones, model_state, alea_uncertainties, epist_uncertainties = model_env_sample(
            model_env, action, model_state, 
        )

        # pred rewards has shape (100000, 1) => unsqueeze uncertainties to add dimension (currently shape (100000))
        # TODO: potentially clamp uncertainties? 
        pred_rewards += factor_epist_uncert_reward_bonus * np.clip(epist_uncertainties.unsqueeze(dim=-1).numpy(), a_min=clip_epist_uncert_min, a_max=clip_epist_uncert_max)

        truncateds = np.zeros_like(pred_dones, dtype=bool)

        # NOTE: added this: exclude predictions with too high aleatoric uncertainty
        pred_too_uncertain = alea_uncertainties.numpy() > max_alea_uncert
        
        # TODO: log
        n_rejected_predictions += pred_too_uncertain.sum()
        stats_alea_uncertainties.append((alea_uncertainties.min(), alea_uncertainties.mean(), alea_uncertainties.max()))
        stats_epis_uncertainties.append((epist_uncertainties.min(), epist_uncertainties.mean(), epist_uncertainties.max()))

        accum_dones |= pred_too_uncertain

        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
            truncateds[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()
    return (n_rejected_predictions, np.mean(stats_alea_uncertainties, axis=0), np.mean(stats_epis_uncertainties, axis=0))

def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
) -> float:
    avg_episode_reward = 0.0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        video_recorder.init(enabled=(episode == 0))
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        (
            obs,
            action,
            next_obs,
            reward,
            terminated,
            truncated,
        ) = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, terminated, truncated)
        return new_buffer
    return sac_buffer

LOGGER_GROUP_UNCERTAINTY = "env_uncertainty"
UNCERT_LOG_FORMAT = [
    ("epoch", "E", "int"),
    ("env_step", "S", "int"),
    ("n_rejected_predictions", "N_REJ", "int"),

    ("alea_uncert_min", "U_ALE_MIN", "float"),
    ("alea_uncert_mean", "U_ALE_MEAN", "float"),
    ("alea_uncert_max", "U_ALE_MAX", "float"),

    ("epis_uncert_min", "U_EPI_MIN", "float"),
    ("epis_uncert_mean", "U_EPI_MEAN", "float"),
    ("epis_uncert_max", "U_EPI_MAX", "float"),
]

def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)

    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    agent = SACAgent(SAC(obs_shape[0], env.action_space, cfg_resolved.algorithm.agent.args))
    
        # cast(SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    # )

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    logger.register_group(
        LOGGER_GROUP_UNCERTAINTY,
        UNCERT_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32

    # TODO: finish creating seperate models
    if cfg.algorithm.learned_rewards: 
        dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    else: 
        cfg.algorithm.learned_rewards = True
        dynamics_model = create_seperate_transition_reward_model(cfg, obs_shape, act_shape)

    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0


    # def create_env(): 
    #     return gym.make("Hockey-One-v0")
    # print("creating fake envs")
    # fake_env = gym.vector.SyncVectorEnv(
    #     [create_env for _ in range(100000)]
    # )
    # print("done")
    
    # fake_env = gym.make("Hockey-One-v0")
    # fake_env.reset()
    
    # def reward_fn(actions, observs): 
    #     def get_reward(i): 
    #         fake_env.reset()
    #         fake_env.set_state(observs[i].numpy().astype(np.float64))
    #         _, reward,  _, _ , _ = fake_env.step(actions[i])
    #         return reward
    #     return torch.tensor([get_reward(i) for i in tqdm(range(len(observs)), "Correcting next rewards")]).unsqueeze(1)

    reward_fn = get_hockey_reward(cfg.device)

    model_env = mbrl.models.ModelEnv(
        # env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0
    sac_buffer = None
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        obs = None
        terminated = False
        truncated = False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or terminated or truncated:
                steps_epoch = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False

                # NOTE: added noise reset here
                noise = reset_noise(cfg.overrides.beta, cfg.overrides.epoch_length, act_shape[0])
                current_env_step = 0 
            # --- Doing env step and adding to model dataset ---
            # NOTE: could explore here? 
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            # ) = mbrl.util.common.step_env_and_add_to_buffer(
            #     env, obs, agent, {}, replay_buffer
            # )
            ) = step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer, noise=noise[:, current_env_step], sigma=cfg.overrides.sigma
            )
            current_env_step += 1 
            # --------------- Model Training -----------------
            # if cfg.algorithm.real_data_ratio == 1: 
            #     print("skipped model training because model data not used")

            if ((env_steps + 1) % cfg.overrides.freq_train_model == 0):
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                n_rejected_predictions, stats_alea_uncertainties, stats_epis_uncertainties  = rollout_model_and_populate_sac_buffer(
                    model_env,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                    cfg.overrides.own_hparams if cfg.overrides.own_hparams else dict(),
                )
                logger.log_data(
                    LOGGER_GROUP_UNCERTAINTY,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "n_rejected_predictions": n_rejected_predictions,
                        "alea_uncert_min": stats_alea_uncertainties[0],
                        "alea_uncert_mean": stats_alea_uncertainties[1],
                        "alea_uncert_max": stats_alea_uncertainties[2],
                        "epis_uncert_min": stats_epis_uncertainties[0],
                        "epis_uncert_mean": stats_epis_uncertainties[1],
                        "epis_uncert_max": stats_epis_uncertainties[2],
                    })

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    which_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                avg_reward = evaluate(
                    test_env, agent, cfg.algorithm.num_eval_episodes, video_recorder
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": avg_reward,
                        "rollout_length": rollout_length,
                    },
                )
                if avg_reward > best_eval_reward:
                    # NOTE replaced save because it needed codec
                    if video_recorder.enabled:
                        video_path = os.path.join(video_recorder.save_dir, f"{epoch}.mp4")
                        imageio.mimsave(video_path, video_recorder.frames, fps=video_recorder.fps, codec="libx264")

                    # video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1

            env_steps += 1
            obs = next_obs
    return np.float32(best_eval_reward)
