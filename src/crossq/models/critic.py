import torch
import math
from crossq.models.feedforward import FeedForward, NNConfig


@torch.no_grad()
def project_weight_to_norm_ball(module: torch.nn.Linear, scale: float | None  = 1.0):
    weight, bias = module.weight, module.bias
    if not scale:
        scale = math.sqrt(module.weight.shape[0] * 2)  # expected norm under He initialization
    vec = torch.concat([weight.view(-1), bias.view(-1)])
    n = vec.norm(p=2)

    if scale < n:
        module.weight.mul_(scale / (n + 1e-12))  # epsilon to avoid division by zero
        module.bias.mul_(scale / (n + 1e-12))

class QNetwork(FeedForward):
    def __init__(self, config: NNConfig, *args, **kwargs):
        super().__init__(config=config,
                          *args, **kwargs)

    def q_value(self, s, a):
        x = torch.concat([s, a], dim=1)
        return self.forward(x)

    # for Weight Normalization

    def normalize_weights_(self) -> None:
        for name, module in self.named_modules():
            # only normalize non-finale dense layer's weights
            if "dense" in name:
                project_weight_to_norm_ball(module)

    
    