import torch
import torch.nn as nn

from generic.transformer import Transformer


class S(nn.Module):
    def __init__(
            self,
            config,
            variable_count,
            clause_count,
    ):
        super(S, self).__init__()

        self.device = torch.device(config.get('device'))
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
            variable_count+1, self.embedding_size,
        )

        layers = []

        layers += [
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        ]

        for _ in range(self.layer_count):
            layers += [
                Transformer(
                    self.hidden_size,
                    self.attention_head_count,
                    self.intermediate_size,
                    dropout=0.00,
                ),
                Transformer(
                    self.hidden_size,
                    self.attention_head_count,
                    self.intermediate_size,
                    dropout=0.00,
                ),
            ]

        head = [
            nn.Linear(self.hidden_size, 1),
            nn.Tanh(),
        ]

        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(*head)

        self.apply(self.init_weights)

    def init_weights(
            self,
            module,
    ):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def parameters_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(
            self,
            cl_pos,
            cl_neg,
    ):
        embeds = self.embedding(cl_pos).sum(2) - self.embedding(cl_neg).sum(2)
        hiddens = self.layers(embeds)

        return 0.5 + 0.5 * self.head(hiddens.mean(1))