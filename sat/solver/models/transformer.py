import torch
import torch.nn as nn

from generic.gelu import GeLU
from generic.layer_norm import LayerNorm
from generic.transformer import Transformer


class S(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(S, self).__init__()

        self.device = torch.device(config.get('device'))

        self.variable_count = \
            config.get('sat_dataset_variable_count')
        self.embedding_size = \
            config.get('sat_solver_transformer_embedding_size')
        self.hidden_size = \
            config.get('sat_solver_transformer_hidden_size')
        self.intermediate_size = \
            config.get('sat_solver_transformer_intermediate_size')
        self.attention_head_count = \
            config.get('sat_solver_transformer_attention_head_count')
        self.layer_count = \
            config.get('sat_solver_transformer_layer_count')

        self.embedding = nn.Embedding(
            self.variable_count+1, self.embedding_size,
        )

        layers = []
        layers += [
            nn.Linear(self.embedding_size, self.hidden_size),
        ]

        for _ in range(self.layer_count):
            layers += [
                Transformer(
                    self.hidden_size,
                    self.attention_head_count,
                    self.intermediate_size,
                    dropout=0.1,
                ),
            ]

        head = [
            nn.Linear(self.hidden_size, self.hidden_size),
            GeLU(),
            nn.Dropout(0.1),
            LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        ]

        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(*head)

    def parameters_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(
            self,
            cl_pos,
            cl_neg,
    ):
        embeds = self.embedding(cl_pos).sum(2) - self.embedding(cl_neg).sum(2)
        hiddens = self.layers(embeds)

        pool = torch.tanh(
            # torch.mean(hiddens, 1)
            torch.max(hiddens, 1)[0]
        )

        return self.head(pool)
