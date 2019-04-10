import torch
import torch.nn as nn

from prooftrace.prooftrace import ACTION_TOKENS, PREPARE_TOKENS

from generic.gelu import GeLU
from generic.layer_norm import LayerNorm


class PH(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(PH, self).__init__()

        self.device = torch.device(config.get('device'))

        self.sequence_length = \
            config.get('prooftrace_sequence_length')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')
        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')

        self.adapter = nn.Linear(self.hidden_size, self.lstm_hidden_size)

        self.action_head = nn.Sequential(
            nn.Linear(
                self.lstm_hidden_size,
                self.lstm_hidden_size,
            ),
            GeLU(),
            LayerNorm(self.lstm_hidden_size),
            nn.Linear(
                self.lstm_hidden_size,
                len(ACTION_TOKENS) - len(PREPARE_TOKENS)
            ),
            nn.LogSoftmax(dim=1),
        )
        self.left_head = nn.Sequential(
            nn.Linear(
                self.lstm_hidden_size,
                self.lstm_hidden_size,
            ),
            GeLU(),
            LayerNorm(self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )
        self.right_head = nn.Sequential(
            nn.Linear(
                self.lstm_hidden_size,
                self.lstm_hidden_size,
            ),
            GeLU(),
            LayerNorm(self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            heads,
            targets,
    ):
        residuals = self.adapter(targets) + heads

        actions = self.action_head(residuals)
        lefts = self.left_head(residuals)
        rights = self.right_head(residuals)

        return actions, lefts, rights


class VH(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(VH, self).__init__()

        self.device = torch.device(config.get('device'))

        self.hidden_size = \
            config.get('prooftrace_hidden_size')
        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')

        self.adapter = nn.Linear(self.hidden_size, self.lstm_hidden_size)

        self.value_head = nn.Sequential(
            nn.Linear(
                self.lstm_hidden_size,
                self.lstm_hidden_size,
            ),
            GeLU(),
            LayerNorm(self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, 1),
            nn.Softplus(),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            heads,
            targets,
    ):
        residuals = self.adapter(targets) + heads

        value = self.value_head(residuals)

        return value
