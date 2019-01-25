import torch
import torch.nn as nn

from generic.transformer import Transformer


class E(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(E, self).__init__()

        self.device = torch.device(config.get('device'))

        self.token_count = \
            config.get('th2vec_token_count')
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

        self.embedding = nn.Embedding(
            self.token_count, self.embedding_size,
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

        # self.apply(self.init_weights)

    # def init_weights(
    #         self,
    #         module,
    # ):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         # if module.bias is not None:
    #         #     module.bias.data.zero_()

    def parameters_count(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(
            self,
            theorem,
    ):
        embeds = self.embedding(theorem)
        hiddens = self.layers(embeds)

        return torch.tanh(torch.max(hiddens, 1)[0])
