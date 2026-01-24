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

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    print("Using device", cfg.device)
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    assert cfg.algorithm.name == "mbpo"
    test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
    return mbpo.train(env, test_env, term_fn, cfg)
    
if __name__ == "__main__":
    run()
