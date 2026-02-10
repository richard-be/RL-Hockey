import torch
from models.feedforward import FeedForward, NNConfig


@torch.no_grad()
def project_weight_to_norm_ball(fc: torch.nn.Module, scale: float = 1.0):
    n = fc.parametrizations.weight.original0.norm()
    if n > scale:
        fc.parametrizations.weight.original0.mul_(1 / (n + 1e-12))  # epsilon to avoid division by zero

class QNetwork(FeedForward):
    def __init__(self, config: NNConfig, *args, **kwargs):
        super().__init__(config=config,
                          *args, **kwargs)

    def q_value(self, s, a):
        x = torch.concat([s, a], dim=1)
        return self.forward(x)
    

    # for Weight Normalization

    def normalize_weights_(self) -> None:

        for mod_name, mod in self.named_modules():
            if ("body" in mod_name) and ("dense" in mod_name) and isinstance(mod, torch.nn.Linear):
                project_weight_to_norm_ball(mod)

        # for name, parameter in self.named_parameters():
        #     # only normalize non-finale dense layer's weights
        #     if "body" in name and "dense" in name and "weight" in name:
        #         project_weight_to_norm_ball(parameter)

    def get_weight_norms(self) -> list[float]:
        norms = []
        for name, parameter in self.named_parameters():
            if "dense" in name and "weight" in name:
                norms.append(torch.norm(parameter, p="fro"))
        return torch.tensor(norms)
