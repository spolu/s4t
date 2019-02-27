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

        self.embedder = ActionEmbedder(config)
        self.embedder.to(self.device)

        self.lstm = nn.LSTM(
            self.hidden_size, self.hidden_size,
            num_layers=2, batch_first=True,
        )

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

        return actions, lefts, rights

    def forward(
            self,
            embeds,
    ):
        hiddens, _ = self.lstm(embeds)

        return hiddens
