import torch
import torch.nn as nn

from dataset.prooftrace import ACTION_TOKENS


class PH(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(PH, self).__init__()

        self.device = torch.device(config.get('device'))

        self.sequence_length = \
            config.get('prooftrace_sequence_length')
        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')

        self.action_head = nn.Sequential(
            nn.Linear(2*self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, len(ACTION_TOKENS)),
            nn.LogSoftmax(dim=1),
        )
        self.left_head = nn.Sequential(
            nn.Linear(2*self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )
        self.right_head = nn.Sequential(
            nn.Linear(2*self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, self.sequence_length),
            nn.LogSoftmax(dim=1),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            predictions,
            targets,
    ):
        residuals = torch.cat([targets, predictions], dim=1)

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

        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')

        self.value_head = nn.Sequential(
            nn.Linear(2*self.lstm_hidden_size, self.lstm_hidden_size),
            nn.Linear(self.lstm_hidden_size, 1),
        )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            predictions,
            targets,
    ):
        residuals = torch.cat([targets, predictions], dim=1)

        value = self.value_head(residuals)

        return value
