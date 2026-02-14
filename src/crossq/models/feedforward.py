import torch
from torch import nn
from dataclasses import dataclass, field
from copy import deepcopy

from models.batchrenorm import BatchRenorm1d

from collections import OrderedDict



@dataclass
class NormalizationConfig:
    type: str = "BN"
    momentum: float = .01

    warmup_steps: int = 10_000  # for Batch renorm

@dataclass
class NNConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int | list[int]  # if  multiple output layers

    num_hidden_layers: int = 1
    act_func: nn.Module = field(default_factory=nn.ReLU)

    output_act: None | nn.Module | list[None | nn.Module] = None  # can be a list if num_heads > 1
    
    num_heads: int = 1

    use_normalization: bool = True
    normalization_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    

class FeedForward(nn.Module):

    def __init__(self, config: NNConfig,
                        *args, **kwargs):
        super().__init__(*args, **kwargs)

        layers = []
        # Construct Input layer
       

        if config.use_normalization:
            if config.normalization_config.type == "BN":
                input_norm_layer = nn.BatchNorm1d(num_features=config.input_dim,
                                            momentum=config.normalization_config.momentum)
            else:
                input_norm_layer = BatchRenorm1d(num_features=config.input_dim,
                                                           momentum=config.normalization_config.momentum,
                                                           warmup_steps=config.normalization_config.warmup_steps)
            layers.append(("batchnorm0", deepcopy(input_norm_layer)))
        

        layers.extend([("dense0", nn.Linear(config.input_dim, config.hidden_dim)), ("act0", config.act_func)])

        

        if config.use_normalization:
            if config.normalization_config.type == "BN":
                norm_layer = nn.BatchNorm1d(num_features=config.hidden_dim,
                                            momentum=config.normalization_config.momentum)
            else:
                norm_layer = BatchRenorm1d(num_features=config.hidden_dim,
                                                           momentum=config.normalization_config.momentum,
                                                           warmup_steps=config.normalization_config.warmup_steps)
            layers.append(("batchnorm1", deepcopy(norm_layer)))
     
        for idx in range(config.num_hidden_layers):
            layers.append((f"dense{idx + 1}", nn.Linear(config.hidden_dim, config.hidden_dim)))
            layers.append((f"act{idx + 1}", deepcopy(config.act_func)))
            if config.use_normalization:
                layers.append((f"batchnorm{idx + 2}", deepcopy(norm_layer)))

        self.body = nn.Sequential(OrderedDict(layers))


        # add output layer(s)
        output_layers = []
        for idx in range(config.num_heads):
            out_dim = config.output_dim[idx] if isinstance(config.output_dim, list) else config.output_dim
            fc_out = nn.Linear(config.hidden_dim, out_dim)

            if config.output_act:
                out_act = config.output_act[idx] if isinstance(config.output_act, list) else deepcopy(config.output_act)

                if not out_act:
                    output_layers.append(fc_out)
                else:
                    output_layers.append(nn.Sequential(fc_out, out_act))
            else:
                output_layers.append(fc_out)

        self.output_layers = nn.ModuleList(output_layers)
            


    
    def forward(self, x):
        z = self.body(x)
        if len(self.output_layers) > 1:
            return [out_layer(z) for out_layer in self.output_layers]
        else:
            return self.output_layers[0](z)
        

    def get_weight_norms(self) -> dict[float]:
        norms = {}
        for name, parameter in self.named_parameters():
            if ("dense" in name or "output_layers" in name) and "weight" in name:
                norms[name] = torch.norm(parameter, p="fro").item()
        return norms
        

def get_gradient_norm(model: nn.Module, p: int = 2) -> float:
    norm = 0
    for param in model.parameters():
        param_norm = param.grad.detach().data.norm(p).item()
        norm += param_norm ** p
    return norm ** (1.0 / p )