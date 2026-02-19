import os 
from mbrl.util.common import create_one_dim_tr_model
import pathlib 
import torch 

def get_latest_run_dir(results_dir):
    def get_highest_folder(directory): 
        sub_dirs = sorted([e.name for e in os.scandir(directory) if e.is_dir()], reverse=True)
        return directory+"/"+sub_dirs[0] if len(sub_dirs) > 0 else None   
    
    dir_most_recent_date = get_highest_folder(results_dir) 
    return get_highest_folder(dir_most_recent_date) if dir_most_recent_date is not None else None


def load_dynamics_model(model_dir, env, cfg,):
    if not cfg.algorithm.learned_rewards: 
        from algorithm.seperate_transition_reward_model import create_seperate_transition_reward_model
        dynamics_model = create_seperate_transition_reward_model(cfg, env.observation_space.shape, env.action_space.shape)
        dynamics_model.model.load(model_dir)
        return dynamics_model

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
