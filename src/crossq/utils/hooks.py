import torch


def register_relu_hooks(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    activations = {}

    def hook_fn(name: str):
        def inner_hook(module, input, output) -> None:
            activations[name] = output.detach()
        return inner_hook
    
    for name, module in model.named_modules():
        if "act" in name:
            module.register_forward_hook(hook_fn(name))
    return activations


@torch.no_grad()
def compute_dead_relu_metrics(activations: dict[str, torch.Tensor]):
    stats = {}
    for name, act in activations.items():

        act_flat = act.view(act.shape[0], act.shape[1], -1)  # shape should be (batch_size, hidden_dim, 1)

        total = act_flat.shape[1]
        dead = (act_flat == 0).all(dim=(0, 2)).sum().item()

        stats[name] = {"total": total, "dead": dead, "dead_ratio": dead / float(total)}
        