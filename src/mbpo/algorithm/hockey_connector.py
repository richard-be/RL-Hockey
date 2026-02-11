from hockey.hockey_env import W, H, CENTER_X, CENTER_Y, GOAL_SIZE, SCALE
import torch

PUCK_RADIUS = 13
MAX_TIMESTEPS = 250


def dist_positions(p1, p2):
  return torch.sqrt(torch.sum((p1 - p2) ** 2, axis=-1))


def get_hockey_reward(device = "cpu", include_closeness_to_puck=True):
    # poly = [(-10, GOAL_SIZE), (10, GOAL_SIZE), (10, -GOAL_SIZE), (-10, -GOAL_SIZE)]

    goal_player_1 = (W / 2 - 245 / SCALE - 10 / SCALE, H / 2)
    goal_player_2 = (W / 2 + 245 / SCALE + 10 / SCALE, H / 2)

    offsets = (10 / SCALE, GOAL_SIZE / SCALE)
    # offset_bottom = (10, -GOAL_SIZE)

    g1_x = goal_player_1[0] + offsets[0] + PUCK_RADIUS / SCALE
    g1_y_top = goal_player_1[1] + offsets[1] + PUCK_RADIUS / SCALE
    g1_y_bottom = goal_player_1[1] - offsets[1] - PUCK_RADIUS / SCALE

    g2_x = goal_player_2[0] - offsets[0] - PUCK_RADIUS / SCALE
    # g2_y_top = goal_player_2[1] + offsets[1]  + PUCK_RADIUS / SCALE
    # g2_y_bottom = goal_player_2[1] - offsets[1] - PUCK_RADIUS / SCALE


    # print("goal player 1:", g1_x, f"{g1_y_bottom} <-> {g1_y_top}")
    # print("goal player 2:", g2_x, f"{g2_y_bottom} <-> {g2_y_top}")

    center = torch.tensor([CENTER_X, CENTER_Y])

    def reward_fn(actions, observs):
        assert len(observs.shape) == 2 # BATCH_SIZE, OBS_DIM
        
        batch_size = observs.shape[0]
        center_offset = center.repeat((batch_size, 1)).to(device, dtype=observs.dtype)
        
        puck_pos = observs[:, [12, 13]] + center_offset 
        # print("pred puck pos", puck_pos)
        puck_x, puck_y = puck_pos[:, 0], puck_pos[:, 1]

        reward = torch.zeros((batch_size, 1))
        is_goal_1 = (puck_x <= g1_x) & (g1_y_bottom <= puck_y) & (puck_y <= g1_y_top)
        is_goal_2 = (puck_x >= g2_x) & (g1_y_bottom <= puck_y) & (puck_y <= g1_y_top)

        # print("is goal 1:", is_goal_1.item())
        reward[is_goal_1] = -10 # puck is in player 1's goal => penalty
        reward[is_goal_2] = 10 # puck is in player 2's goal => reward

        # if (is_goal_1 | is_goal_2).any():
        #     print("goal!", reward)

        # add closeness to puck reward 
        if include_closeness_to_puck: 
            # observs[:, 14] is linear velocity in x 
            puck_is_left = (puck_x < CENTER_X) & (observs[:, 14] <= 0)

            player1_pos = observs[:, [0, 1]] + center_offset

            # if self.puck.position[0] < CENTER_X and self.puck.linearVelocity[0] <= 0:
            dist_to_puck = dist_positions(player1_pos, puck_pos)
            assert dist_to_puck.shape[0] == batch_size
            assert len(dist_to_puck.shape) == 1
            assert puck_is_left.shape[0] == batch_size
            assert len(puck_is_left.shape) == 1


            max_dist = 250. / SCALE
            max_reward = -30.  # max (negative) reward through this proxy
            factor = max_reward / (max_dist * MAX_TIMESTEPS / 2)
            
            reward_closeness_to_puck = dist_to_puck * factor  # Proxy reward for being close to puck in the own half
            reward_closeness_to_puck *= puck_is_left # if clause in hockey env computation 
            
            reward += reward_closeness_to_puck.unsqueeze(-1)
        return reward
    return reward_fn

def get_hockey_termination(device): 
   reward_fn = get_hockey_reward(device, include_closeness_to_puck=False)
   def termination_fn(actions, observs): 
      return torch.abs(reward_fn(actions, observs)) == 10
   return termination_fn