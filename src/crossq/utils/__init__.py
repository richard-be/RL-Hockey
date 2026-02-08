import torch


@torch.no_grad()
def parameter_snapshot(model: torch.nn.Module, names: set[str] | None = None ) -> dict[str, torch.Tensor]:
    snapshot = {}
    for name, parameter in model.named_parameters():
        if names is None or name in names:
            snapshot[name] = parameter.detach().clone()

    return snapshot


@torch.no_grad()
def measaure_effecitve_learning_rate(snapshot: dict[str, torch.Tensor], model: torch.nn.Module):
    elrs = {}
    names = set(snapshot.keys())
    for name, parameter in model.named_parameters():
        if name in names:
            diff_norm = (parameter - snapshot[name]).norm(p=2)
            prev_norm = snapshot[name].norm(p=2).clamp_min(1e-12)
            elrs[name] = (diff_norm / prev_norm).item()

    return elrs


