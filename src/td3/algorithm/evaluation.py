import torch 
import numpy as np

def evaluate(
    eval_envs, 
    actor, 
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    render: bool = False, 
):
    actor.eval()
    # note: qf1 and qf2 are not used in this script
    obs, _ = eval_envs.reset()

    n_envs = len(eval_envs.envs)

    accum_rewards = np.zeros(n_envs)
    won = np.zeros(n_envs)
    lost = np.zeros(n_envs)

    n_running = n_envs
    max_steps = 250
    # while (n_running > 0) and (max_steps > 0): 
    #     max_steps -= 1
    #     with torch.no_grad():
    #         actions = actor(torch.Tensor(obs).to(device))
    #         actions = actions.cpu().numpy().clip(eval_envs.single_action_space.low, eval_envs.single_action_space.high)
    #     next_obs, rewards, terms, truncs, infos = eval_envs.step(actions)

    #     running_envs = ~(terms | truncs)
    #     n_running = sum(running_envs) 
    #     accum_rewards[running_envs] += rewards[running_envs]

    #     for i, terminated in enumerate(terms): 
    #         if terminated: 
    #             won[i] = infos["winner"][i] == 1
    #             lost[i] = infos["winner"][i] == -1

    #     obs = next_obs

    # actor.train()
    # return accum_rewards.mean(), won.mean(), lost.mean()

    done = np.zeros(n_envs, dtype=bool) 

    while not done.all() and max_steps > 0:
        max_steps -= 1

        with torch.no_grad():
            actions = actor(torch.as_tensor(obs, device=device))
            actions = actions.cpu().numpy().clip(eval_envs.single_action_space.low, eval_envs.single_action_space.high)


        next_obs, rewards, terms, truncs, infos = eval_envs.step(actions)

        accum_rewards[~done] += rewards[~done]

        just_finished = (terms | truncs) & (~done)

        for i, terminated in enumerate(terms): 
            if terminated and not done[i]: # just finished now
                won[i] = infos["winner"][i] == 1
                lost[i] = infos["winner"][i] == -1

        done |= just_finished
        obs = next_obs

    actor.train()
    return accum_rewards.mean(), won.mean(), lost.mean()

