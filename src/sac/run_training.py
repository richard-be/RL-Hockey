#based on cleanrl sac_continuous action implementation
import tyro
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from agent.buffers import ReplayBuffer
from agent.sac import Args
from agent.sac import Actor
from agent.sac import SoftQNetwork
import env.custom_hockey as c_env
import hockey.hockey_env as h_env
from env.colored_noise import generate_colored_noise
import copy

def make_env(seed, episode_count, device, weak_opponent, self_play, env_mode="NORMAL", opponent_sampler=None):
    def thunk():
        if not self_play:
            env = c_env.HockeyEnv_Custom_BasicOpponent(env_mode, weak_opponent)
        else:
            env = c_env.HockeyEnv_Custom_CustomOpponent(h_env.BasicOpponent(weak=True), device, mode=h_env.Mode[env_mode]) 
            env = c_env.OpponentResetWrapper(env, opponent_sampler, episode_count)
     
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

def reset_noise(i, noise, beta, samples, action_shape):
    """
    Recreate colored noise arrays
    
    :param i: Description
    :param noise: old noise array
    :param beta: noise exponent
    :param samples: number of samples
    :param action_shape: shape of one action
    """
    noise[i] = np.array([generate_colored_noise(samples, beta) for _ in range(action_shape)])
    return noise

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}_{args.num_q}_{args.update_ratio}_{args.total_timesteps}_{int(time.time())}"

    if args.track:
        writer = SummaryWriter(f"runs/sac/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    opponent_sampler = c_env.OpponentSampler(args.self_play_len)
    episode_count = c_env.EpisodeCounter()
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, episode_count, device, args.weak_opponent, args.self_play, args.env_mode, opponent_sampler) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    q_networks = [SoftQNetwork(envs).to(device) for _ in range(args.num_q)]
    q_targets = [SoftQNetwork(envs).to(device) for _ in range(args.num_q)]
    params = []
    for i in range(args.num_q):
        q_targets[i].load_state_dict(q_networks[i].state_dict())
        params += list(q_networks[i].parameters())
    q_optimizer = optim.Adam(params, lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    #set up colored noise, one series per env and action, gets reset upon episode termination
    samples = 251
    noise = np.zeros((args.num_envs, envs.single_action_space.shape[0], samples))
    for i in range(args.num_envs):
        noise = reset_noise(i, noise, args.beta, samples, envs.single_action_space.shape[0])
    start_time = time.time()

    #setup episode steps for noise indexing
    env_indices = np.arange(args.num_envs)
    episode_steps = np.zeros_like(env_indices)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    frames = []
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.act(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
            actions = actions + args.sigma * noise[env_indices, :, episode_steps]
            actions = np.clip(actions, -1.0, 1.0)

        episode_steps += 1
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        #envs.envs[0].render()

        #count episodes and reset episode steps and noise for finished envs
        for env_index, done in enumerate(terminations):
            if done:
                episode_count.increment()
                #reset episode steps and noise for finished env
                episode_steps[env_index] = 0  
                noise = reset_noise(env_index, noise, args.beta, samples, envs.single_action_space.shape[0])

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for env_index, info in enumerate(infos["final_info"]):
                if info is not None:
                    if episode_count.value % 50 == 0:
                        sps = int(global_step / (time.time() - start_time))
                        if args.self_play:
                            opponent = envs.envs[env_index].get_opponent_name()
                        else:
                            opponent = "weak" if args.weak_opponent else "strong"
                        print(f"episode={episode_count.value}, global_step={global_step}, env={env_index}, winner={info['winner']}, SPS={sps}, opponent={opponent}, episodic_return={info['episode']['r']}, episode_length={info['episode']['l']}")
                    if args.track:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/time_return", info["episode"]["r"], time.time()-start_time)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            for g in range(args.update_ratio):
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.act(data.next_observations)
                    q_targets_sample = random.sample(q_targets, k=args.num_min_q)
                    q_next_targets = torch.stack([q(data.next_observations, next_state_actions) for q in q_targets_sample], dim=0)
                    min_qf_next_target = torch.min(q_next_targets, dim=0)[0] - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf_loss = 0
                q_values_list = []
                for q in q_networks:
                    qf_values = q(data.observations, data.actions).view(-1)
                    qf_loss += F.mse_loss(qf_values, next_q_value)
                    q_values_list.append(qf_values.mean().item())

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for q_network, q_target in zip(q_networks, q_targets):
                        for param, target_param in zip(q_network.parameters(), q_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 100 == 0 and args.track:
                    writer.add_scalar("losses/qf_values", np.mean(q_values_list), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / args.num_q, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.act(data.observations)
                    q_pi = torch.stack([q(data.observations, pi) for q in q_networks], dim=0)
                    actor_loss = ((alpha * log_pi) - torch.mean(q_pi, dim=0)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.act(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step >= args.freeze_start and global_step % args.freeze_freq == 0 and args.self_play:
                frozen_actor = copy.deepcopy(actor)
                frozen_actor.eval()
                for p in frozen_actor.parameters():
                    p.requires_grad = False

                opponent_sampler.add_self_play_opponent(frozen_actor)


    envs.close()
    if args.track:
        writer.close()
        torch.save(actor.state_dict(), f"models/sac/{run_name}.pkl")

