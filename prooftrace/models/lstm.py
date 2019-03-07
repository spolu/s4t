import torch
import torch.nn as nn

from dataset.prooftrace import ACTION_TOKENS

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

        self.action_head = nn.Sequential(
            nn.Linear(2 * self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, len(ACTION_TOKENS)),
            nn.LogSoftmax(dim=1),
        )
        self.left_head = nn.Sequential(
            nn.Linear(2 * self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )
        self.right_head = nn.Sequential(
            nn.Linear(2 * self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
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
            targets,
    ):
        residuals = torch.cat([targets, predictions], dim=1)

        actions = self.action_head(residuals)
        lefts = self.left_head(residuals)
        rights = self.right_head(residuals)

        return actions, lefts, rights

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
