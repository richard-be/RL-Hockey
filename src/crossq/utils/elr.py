import torch


@torch.no_grad()
def parameter_snapshot(model: torch.nn.Module, names: set[str] | None = None ) -> dict[str, torch.Tensor]:
    snapshot = {}
    for name, parameter in model.named_parameters():
        if names is None or name in names:
            snapshot[name] = parameter.detach().clone()

    return snapshot


@torch.no_grad()
def measaure_effecitve_learning_rate( model: torch.nn.Module, names: set[str] | None = None, lr: float = 1e-3):
    elrs = {}
    for name, parameter in model.named_parameters():
        if name in names:
            elrs[name] = lr / parameter.norm().clamp_min(1e-12)

    return elrs


