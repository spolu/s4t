import math
import torch
import torch.nn as nn


class GeLU(nn.Module):
    def __init__(
            self,
    ):
        super(GeLU, self).__init__()

    def forward(
            self,
            x,
    ):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))
