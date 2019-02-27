import torch
import torch.nn as nn

from dataset.prooftrace import ACTION_TOKENS

from generic.transformer import TransformerBlock

from prooftrace.models.embedder import ActionEmbedder


class P(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(P, self).__init__()

        self.device = torch.device(config.get('device'))

        self.sequence_length = \
            config.get('prooftrace_sequence_length')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')
        self.attention_head_count = \
            config.get('prooftrace_transformer_attention_head_count')
        self.layer_count = \
            config.get('prooftrace_transformer_layer_count')

        self.embedder = ActionEmbedder(config)
        self.embedder.to(self.device)

        self.position_embedding = nn.Embedding(
            self.sequence_length, self.hidden_size
        )

        layers = []

        for _ in range(self.layer_count):
            layers += [
                TransformerBlock(
                    self.sequence_length,
                    self.hidden_size,
                    self.attention_head_count,
                    dropout=0.0,
                ),
            ]

        self.layers = nn.Sequential(*layers)

        # position_decoder = nn.Linear(
        #     self.hidden_size, self.sequence_length, bias=False,
        # )
        # position_decoder.weight = self.position_embedding.weight

        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, len(ACTION_TOKENS)),
            nn.LogSoftmax(dim=1),
        )
        self.left_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )
        self.right_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, 1),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed(
            self,
            actions,
    ):
        return self.embedder(actions)

    def head(
            self,
            predictions,
    ):
        actions = self.action_head(predictions)
        lefts = self.left_head(predictions)
        rights = self.right_head(predictions)
        values = self.value_head(predictions)

        return actions, lefts, rights, values

    def forward(
            self,
            embeds,
    ):
        pos_embeds = torch.arange(
            self.sequence_length, dtype=torch.long
        ).to(self.device)
        pos_embeds = pos_embeds.unsqueeze(0).expand(
           embeds.size(0), self.sequence_length,
        )
        pos_embeds = self.position_embedding(pos_embeds)

        hiddens = self.layers(embeds + pos_embeds)

        return hiddens
