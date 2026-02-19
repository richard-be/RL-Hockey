from mbrl.models import Model, Ensemble, GaussianMLP
import torch 
from typing import Optional, Dict
import mbrl.util.common
import mbrl.models
import hydra 

import pathlib
from typing import Sequence, Union

def create_seperate_transition_reward_model(cfg, obs_shape, act_shape): 
    model_cfg = cfg.dynamics_model
    if issubclass(hydra.utils._locate(model_cfg._target_), mbrl.models.BasicEnsemble):
        model_cfg = model_cfg.member_cfg
    if model_cfg.get("in_size", None) is None:
        model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    if model_cfg.get("out_size", None) is None:
        model_cfg.out_size = obs_shape[0] # + int(cfg.algorithm.learned_rewards)

    # Now instantiate the model
    state_model = hydra.utils.instantiate(model_cfg)


    # # name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    # # if name_obs_process_fn:
    #     # obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    # # else:
    # # obs_process_fn = None

    # state_model_in_size = obs_shape[0] + act_shape[0]
    # state_model_out_size = obs_shape[0] #+ int(cfg.algorithm.learned_rewards)
    # # @package _group_
    # _target_: mbrl.models.GaussianMLP
    # num_layers: 4
    # ensemble_size: 7
    # hid_size: 200
    # deterministic: false
    # propagation_method: random_model
    # learn_logvar_bounds: false  # so far this works better
    # activation_fn_cfg:
    # _target_: torch.nn.SiLU

    reward_model = GaussianMLP(
        in_size = obs_shape[0] + act_shape[0], 
        out_size = 1, 
        num_layers = 3, 
        device = cfg.device, 
        ensemble_size = 7, 
        hid_size = 200, 
        deterministic = True, 
        propagation_method = "random_model", 
        learn_logvar_bounds = False, 
        activation_fn_cfg = cfg.dynamics_model.activation_fn_cfg
    )

    # model = SeperateTransitionRewardEnsembleModel(cfg.device, "random_model", state_model, reward_model)

    model = DummyModel(state_model, reward_model)
    dynamics_model = mbrl.models.OneDTransitionRewardModel(
        model,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", False
        ),
        learned_rewards=True,
        # obs_process_fn=obs_process_fn,
        no_delta_list=cfg.overrides.get("no_delta_list", None),
        num_elites=cfg.overrides.get("num_elites", None),
    )

    return dynamics_model

class DummyModel(torch.nn.Module): 
    def __init__(self, state_model: Ensemble, reward_model: Ensemble):
        super().__init__()

        self.in_size = state_model.in_size
        self.out_size = state_model.out_size 
        self.device = state_model.device
        self.deterministic = state_model.deterministic
        # assert state_model.deterministic == reward_model.deterministic
        assert len(state_model) == len(reward_model), "Expected same ensemble size"

        self.state_model = state_model
        self.reward_model = reward_model

        self._STATE_MODEL_FNAME = "state_model.pth"
        self._REWARD_MODEL_FNAME = "reward_model.pth"

    def __len__(self):
        return len(self.state_model) #+ len(self.reward_model)

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor):
        target_next_obs = target[..., :-1]
        target_reward = target[..., -1:]

        score_obs, _ = self.state_model.eval_score(model_in, target_next_obs)
        score_reward, _ = self.reward_model.eval_score(model_in, target_reward)

        return torch.cat((score_obs, score_reward), -1), {} #, {"score_obs": score_obs.item(), "score_reward": score_reward.item()}

    def update(self, model_in, optimizer, target):
        self.train()
        optimizer.zero_grad()
        loss, meta = self.loss(model_in, target)
        loss.backward()
        if meta is not None:
            with torch.no_grad():
                grad_norm = 0.0
                for p in list(filter(lambda p: p.grad is not None, self.parameters())):
                    grad_norm += p.grad.data.norm(2).item() ** 2
                meta["grad_norm"] = grad_norm
        optimizer.step()

        return loss.item(), meta
    
    def _reward_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert self.reward_model.deterministic
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)

        pred_mean, pred_logvar = self.reward_model.forward(model_in, use_propagation=False)
        # if target.shape[0] != self.reward_model.num_members:
            # target = target.repeat(self.reward_model.num_members, 1, 1)
        # return torch.functional.F.binary_cross_entropy(pred_mean, target, reduction="none").sum((1, 2)).sum()
        loss = torch.functional.F.huber_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()
        return loss, {}
        # return torch.functional.focal(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _seperate_forward(self, model_in, target): 
        # assert self.reward_model.deterministic
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)

        state_pred_mean, state_pred_logvar = self.reward_model.forward(model_in, use_propagation=False)
        
        reward_model_in = torch.cat(model_in, state_pred_mean)
        reward_pred_mean, reward_pred_logvar = self.reward_model.forward(reward_model_in, use_propagation=False)

        return (state_pred_mean, state_pred_logvar), (reward_pred_mean, reward_pred_logvar)

    # def loss(self, model_in: torch.Tensor, target: torch.Tensor):
    #     # target: concatenated [next_obs, reward]
    #     target_next_obs = target[..., :-1]
    #     target_reward = target[..., -1:]

    #     loss_obs, meta_obs = self.state_model.loss(model_in, target_next_obs)

    #     # loss_reward, meta_reward = self.reward_model.loss(model_in, target_reward)
    #     loss_reward, meta_reward = self._reward_loss(model_in, target_reward)
    #     total_loss = loss_obs + loss_reward

    #     meta = {"loss_obs": loss_obs.item(), "loss_reward": loss_reward.item()}
    #     return total_loss, meta

    
    def forward(self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True
    ): 
        # x: concatenated obs + action
        (mean_s, logvar_s)  = self.state_model.forward(x, rng=rng, propagation_indices=propagation_indices, use_propagation=use_propagation)
        (mean_r, logvar_r) = self.reward_model.forward(x, rng=rng, propagation_indices=propagation_indices, use_propagation=use_propagation)

        if self.reward_model.deterministic: 
            logvar_r = torch.zeros_like(mean_r)
        mean = torch.cat([mean_s, mean_r], dim=-1)
        logvar = torch.cat([logvar_s, logvar_r], dim=-1)

        return mean, logvar

    def sample_1d(self, model_input: torch.Tensor, model_state: Dict[str, torch.Tensor], deterministic=False, rng=None):
        next_obs, new_model_state = self.state_model.sample_1d(model_input, model_state, deterministic=deterministic, rng=rng)
        reward, _ = self.reward_model.sample_1d(model_input, model_state, deterministic=deterministic, rng=rng)
        # TODO: merge states?
        return torch.cat([next_obs, reward], dim=-1), new_model_state

    def reset_1d(self, obs: torch.Tensor, rng: Optional[torch.Generator] = None):
        state_model_state = self.state_model.reset_1d(obs, rng=rng)
        reward_model_state = self.reward_model.reset_1d(obs, rng=rng)
        # TODO: merge states?
        return state_model_state

    def set_elite(self, elite_indices: Sequence[int]):
        self.state_model.set_elite(elite_indices)
        self.reward_model.set_elite(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        torch.save(self.state_model.state_dict(), pathlib.Path(save_dir) / self._STATE_MODEL_FNAME)
        torch.save(self.reward_model.state_dict(), pathlib.Path(save_dir) / self._REWARD_MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        path_state_model_weights = pathlib.Path(load_dir) / self._STATE_MODEL_FNAME
        path_reward_model_weights = pathlib.Path(load_dir) / self._REWARD_MODEL_FNAME
        
        self.state_model.load_state_dict(torch.load(path_state_model_weights, weights_only=False, map_location=torch.device('cpu')))
        self.reward_model.load_state_dict(torch.load(path_reward_model_weights, weights_only=False, map_location=torch.device('cpu')))
# class SeperateTransitionRewardEnsembleModel():
#     """
#     A model that contains two separate ensembles:
#     - state_model_ensemble: predicts next state
#     - reward_model_ensemble: predicts reward
#     """

#     def __init__(self, device, propagation_method, state_model_ensemble: Ensemble, reward_model_ensemble: Ensemble):
#         super().__init__(self, device, propagation_method)
#         self.state_model_ensemble = state_model_ensemble
#         self.reward_model_ensemble = reward_model_ensemble

#         self.in_size = state_model_ensemble.in_size 
#         self.out_size = state_model_ensemble.out_size 
#     def forward(self, x: torch.Tensor, rng=None, propagation_indices=None):
#         # x: concatenated obs + action
#         (mean_s, logvar_s) = self.state_model_ensemble.forward(x, rng=rng, propagation_indices=propagation_indices)[0]
#         (mean_r, logvar_r) = self.reward_model_ensemble.forward(x, rng=rng, propagation_indices=propagation_indices)[0]
#         # concat next_obs and reward like OneDTransitionRewardModel expects
#         # return torch.cat([next_obs, reward], dim=-1), None  # second element for logvar if needed
    
#         mean = torch.cat([mean_s, mean_r], dim=-1)
#         logvar = torch.cat([logvar_s, logvar_r], dim=-1)

#         return mean, logvar


#     def loss(self, model_in: torch.Tensor, target: torch.Tensor):
#         # target: concatenated [next_obs, reward]
#         target_next_obs = target[:, :-1]
#         target_reward = target[:, -1:]
#         loss_obs, meta_obs = self.state_model_ensemble.loss(model_in, target_next_obs)
#         loss_reward, meta_reward = self.reward_model_ensemble.loss(model_in, target_reward)
#         total_loss = loss_obs + loss_reward
#         meta = {"loss_obs": loss_obs.item(), "loss_reward": loss_reward.item()}
#         return total_loss, meta

#     def eval_score(self, model_in: torch.Tensor, target: torch.Tensor):
#         target_next_obs = target[:, :-1]
#         target_reward = target[:, -1:]
#         score_obs = self.state_model_ensemble.eval_score(model_in, target_next_obs)
#         score_reward = self.reward_model_ensemble.eval_score(model_in, target_reward)
#         return score_obs + score_reward, {"score_obs": score_obs.item(), "score_reward": score_reward.item()}

#     def sample_1d(self, model_input: torch.Tensor, model_state: Dict[str, torch.Tensor], deterministic=False, rng=None):
#         next_obs, _ = self.state_model_ensemble.sample_1d(model_input, model_state, deterministic=deterministic, rng=rng)
#         reward, _ = self.reward_model_ensemble.sample_1d(model_input, model_state, deterministic=deterministic, rng=rng)
#         return torch.cat([next_obs, reward], dim=-1), model_state

#     def reset_1d(self, obs: torch.Tensor, rng: Optional[torch.Generator] = None):
#         state_model_state = self.state_model_ensemble.reset_1d(obs, rng=rng)
#         reward_model_state = self.reward_model_ensemble.reset_1d(obs, rng=rng)
#         # you could merge any states you need; MBPO just needs 'propagation_indices'
#         return state_model_state

#     # def sample_propagation_indices(self, batch_size: int, rng: torch.Generator):
#     #     return self.state_model_ensemble.sample_propagation_indices(batch_size, rng)
