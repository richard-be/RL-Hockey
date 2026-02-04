
# NOTE: self-implemented below

from typing import Optional, Sequence, cast, Dict, Tuple
import torch 
import mbrl.models.util as model_util
import mbrl
import numpy as np 

def sample_1d(
    ensemble_model: mbrl.models.GaussianMLP, # NOTE: just immediately assume G_MLP here => break extendability
    model_input: torch.Tensor,
    model_state: Dict[str, torch.Tensor],
    deterministic: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """Samples an output from the model using .

    This method will be used by :class:`ModelEnv` to simulate a transition of the form.
        outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

        - model_input_t: observation and action at time t, concatenated across axis=1.
        - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
        - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

    The default implementation returns `s_t+1=s_t`.

    Args:
        model_input (tensor): the observation and action at.
        model_state (tensor): the model state st. Must contain a key
            "propagation_indices" to use for uncertainty propagation.
        deterministic (bool): if ``True``, the model returns a deterministic
            "sample" (e.g., the mean prediction). Defaults to ``False``.
        rng (`torch.Generator`, optional): an optional random number generator
            to use.

    Returns:
        (tuple): predicted observation, rewards, terminal indicator and model
            state dictionary. Everything but the observation is optional, and can
            be returned with value ``None``.
    """
    if deterministic or ensemble_model.deterministic:
        raise ValueError("Expected non-deterministic sampling")
    assert model_state["propagation_indices"] is None

    # _default_forward
    assert rng is not None
    # means, logvars = ensemble_model.forward(
    #     model_input, rng=rng, propagation_indices=model_state["propagation_indices"]
    # )

    ensemble_means, ensemble_logvars = ensemble_model._default_forward(model_input, only_elite=False)
    assert len(ensemble_means.shape) == 3 # [ENSEMBLE_SIZE, BATCH_SIZE, OUTPUT_DIM]

    means = ensemble_means.mean(dim=0)
    variances = ensemble_logvars.exp().mean(dim=0)
    
    stds = torch.sqrt(variances)
    preds =  torch.normal(means, stds, generator=rng)
    
    aleatoric_uncertainty = variances.max(dim=-1).values # mean predicted variance of all states
    epistemic_uncertainty = ensemble_means.var(dim=0).max(dim=-1).values # variance of predicted means 


    # NOTE: returned uncertainty here!
    return preds, model_state, aleatoric_uncertainty, epistemic_uncertainty

# taken from mbrl.models.one_dim_tr_model
def dynamics_model_sample(
    dynamics_model: mbrl.models.OneDTransitionRewardModel,
    act: torch.Tensor,
    model_state: Dict[str, torch.Tensor],
    deterministic: bool = False,
    rng: Optional[torch.Generator] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[Dict[str, torch.Tensor]],
]:
    """Samples next observations and rewards from the underlying 1-D model.

    This wrapper assumes that the underlying model's sample method returns a tuple
    with just one tensor, which concatenates next_observation and reward.

    Args:
        act (tensor): the action at.
        model_state (tensor): the model state st.
        deterministic (bool): if ``True``, the model returns a deterministic
            "sample" (e.g., the mean prediction). Defaults to ``False``.
        rng (random number generator): a rng to use for sampling.

    Returns:
        (tuple of two tensors): predicted next_observation (o_{t+1}) and rewards (r_{t+1}).
    """
    obs = model_util.to_tensor(model_state["obs"]).to(dynamics_model.device)
    
    model_in = dynamics_model._get_model_input(model_state["obs"], act)

    # NOTE: replaced here to get uncertainty
    preds, next_model_state, aleatoric_uncertainties, epistemic_uncertainties = sample_1d(
        dynamics_model.model, model_in, model_state, rng=rng, deterministic=deterministic
    )
    
    next_observs = preds[:, :-1] if dynamics_model.learned_rewards else preds
    if dynamics_model.target_is_delta:
        tmp_ = next_observs + obs
        for dim in dynamics_model.no_delta_list:
            tmp_[:, dim] = next_observs[:, dim]
        next_observs = tmp_
    rewards = preds[:, -1:] if dynamics_model.learned_rewards else None
    next_model_state["obs"] = next_observs

    # NOTE: replaced return
    return next_observs, rewards, None, next_model_state, aleatoric_uncertainties, epistemic_uncertainties


# taken from mrbl.models.model_env.py and adapted
def model_env_sample(
    dynamics_model: mbrl.models.OneDTransitionRewardModel, 
    termination_fn, 
    rng, 
    actions: mbrl.types.TensorType,
    model_state: Dict[str, torch.Tensor],
    return_as_np: bool = True,
) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
    """Steps the model environment with the given batch of actions.

    Args:
        actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
            Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
            and ``A`` is the action dimension. Note that ``B`` must correspond to the
            batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
            converted to a torch.Tensor and sent to the model device.
        model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
        sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

    Returns:
        (tuple): contains the predicted next observation, reward, done flag and metadata.
        The done flag is computed using the termination_fn passed in the constructor.
    """
    assert len(actions.shape) == 2  # batch, action_dim
    with torch.no_grad():
        # if actions is tensor, code assumes it's already on self.device
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(dynamics_model.device)

        # TODO: replace here!
        (
            next_observs,
            pred_rewards,
            pred_terminals,
            next_model_state,
            aleatoric_uncertainties, 
            epistemic_uncertainties,
        ) = dynamics_model_sample(
            dynamics_model, 
            actions,
            model_state,
            deterministic=False, # NOTE: was <<not sample>> => here we sample => False
            rng=rng,
        )
        rewards = pred_rewards # NOTE: assumes here env has no reward function! s
        dones = termination_fn(actions, next_observs)

        if pred_terminals is not None:
            raise NotImplementedError(
                "ModelEnv doesn't yet support simulating terminal indicators."
            )

        if return_as_np:
            next_observs = next_observs.cpu().numpy()
            rewards = rewards.cpu().numpy()
            dones = dones.cpu().numpy()
        return next_observs, rewards, dones, next_model_state, aleatoric_uncertainties, epistemic_uncertainties

