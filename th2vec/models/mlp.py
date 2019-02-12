import torch
import torch.nn as nn

from generic.gelu import GeLU
from generic.layer_norm import LayerNorm


class P(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(P, self).__init__()

        self.device = torch.device(config.get('device'))

        self.hidden_size = \
            config.get('th2vec_transformer_hidden_size')

        layers = []

        layers += [
            nn.Linear(2*self.hidden_size, self.hidden_size),
            GeLU(),
            nn.Dropout(config.get('th2vec_mlp_dropout')),
            LayerNorm(self.hidden_size),
        ]

        for _ in range(8):
            layers += [
                nn.Linear(self.hidden_size, self.hidden_size),
                GeLU(),
                # nn.Dropout(config.get('th2vec_mlp_dropout')),
                LayerNorm(self.hidden_size),
            ]

        layers += [
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
        ]

        self.layers = nn.Sequential(*layers)

    def parameters_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(
            self,
            conjecture,
            theorem,
    ):
        return self.layers(
            torch.cat((conjecture, theorem), dim=1)
        )
