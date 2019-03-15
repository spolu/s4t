import torch
import torch.nn as nn

from prooftrace.models.embedder import ActionEmbedder


class H(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(H, self).__init__()

        self.device = torch.device(config.get('device'))

        self.sequence_length = \
            config.get('prooftrace_sequence_length')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')
        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')
        self.lstm_layer_count = \
            config.get('prooftrace_lstm_layer_count')

        self.embedder = ActionEmbedder(config)
        self.embedder.to(self.device)

        self.position_embedding = nn.Embedding(
            self.sequence_length, self.hidden_size
        )

        self.lstm = nn.LSTM(
            self.hidden_size, self.lstm_hidden_size,
            num_layers=self.lstm_layer_count, batch_first=True,
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

        hiddens, _ = self.lstm(embeds + pos_embeds)

        return hiddens
