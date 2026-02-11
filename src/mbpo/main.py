# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import algorithm.mbpo as mbpo 
import mbrl.util.env
from hockey.hockey_env import HockeyEnv
import gymnasium as gym 
from algorithm.hockey_connector import get_hockey_termination
class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def render(self, **kwargs): 
        return self.env.render(mode=kwargs["mode"])

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    print("Using device", cfg.device)
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)

    term_fn = get_hockey_termination(cfg.device)
    env = RenderWrapper(env) 

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    assert cfg.algorithm.name == "mbpo"
    test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
    test_env = RenderWrapper(test_env)

    return mbpo.train(env, test_env, term_fn, cfg)
    
if __name__ == "__main__":
    run()
