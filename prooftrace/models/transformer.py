import torch
import torch.nn as nn

from generic.transformer import TransformerBlock

from prooftrace.models.embedder import ActionEmbedder


class LM(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(LM, self).__init__()

        self.device = torch.device(config.get('device'))

        self.sequence_length = \
            config.get('prooftrace_sequence_length')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')
        self.attention_head_count = \
            config.get('prooftrace_transformer_attention_head_count')
        self.layer_count = \
            config.get('prooftrace_transformer_layer_count')

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
                    masking=True,
                    dropout=0.1,
                ),
            ]

        self.layers = nn.Sequential(*layers)

        self.embedder = ActionEmbedder(config)
        self.embedder.to(self.device)

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def embed(
            self,
            actions,
    ):
        return self.embedder(actions)

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

        return self.layers(embeds + pos_embeds)
