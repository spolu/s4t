import torch
import torch.nn as nn

from generic.transformer import Transformer


class P(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(P, self).__init__()

        self.device = torch.device(config.get('device'))

        self.token_count = \
            config.get('th2vec_token_count')
        self.theorem_length = \
            config.get('th2vec_theorem_length')
        self.embedding_size = \
            config.get('th2vec_transformer_embedding_size')
        self.hidden_size = \
            config.get('th2vec_transformer_hidden_size')
        self.intermediate_size = \
            config.get('th2vec_transformer_intermediate_size')
        self.attention_head_count = \
            config.get('th2vec_transformer_attention_head_count')
        self.layer_count = \
            config.get('th2vec_transformer_layer_count')

        self.input_embedding = nn.Embedding(
            self.token_count, self.embedding_size,
        )
        self.position_embedding = nn.Embedding(
            self.theorem_length, self.embedding_size
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

        self.layers = nn.Sequential(*layers)

        self.head = nn.Sequential([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        ])

    def parameters_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(
            self,
            theorem,
    ):
        input_embeds = self.input_embedding(theorem)

        position_embeds = torch.arange(
            input_embeds.size(1), dtype=torch.long
        ).to(self.device)
        position_embeds = position_embeds.unsqueeze(0).expand(
            input_embeds.size(0), input_embeds.size(1),
        )
        import pdb; pdb.set_trace();

        hiddens = self.layers(input_embeds)

        th2vec = torch.tanh(torch.max(hiddens, 1)[0])

        return self.head(th2vec)
