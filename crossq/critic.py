import torch
from feedforward import FeedForward, NNConfig



class QNetwork(FeedForward):
    def __init__(self, config: NNConfig, *args, **kwargs):
        super().__init__(config=config,
                          *args, **kwargs)

    def q_value(self, s, a):
        x = torch.concat([s, a], dim=1)
        return self.forward(x)