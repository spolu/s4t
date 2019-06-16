import torch
import torch.nn as nn

from prooftrace.prooftrace import ACTION_TOKENS, PREPARE_TOKENS

from generic.gelu import GeLU


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

        # self.adapter = nn.Linear(self.hidden_size, self.lstm_hidden_size)

        self.action_head = nn.Sequential(
            nn.Linear(
                self.lstm_hidden_size,
                self.lstm_hidden_size,
            ),
            GeLU(),
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Linear(
                self.lstm_hidden_size,
                len(ACTION_TOKENS) - len(PREPARE_TOKENS)
            ),
            nn.LogSoftmax(dim=1),
        )

        # self.left_head = nn.Sequential(
        #     nn.Linear(
        #         self.lstm_hidden_size,
        #         self.lstm_hidden_size,
        #     ),
        #     GeLU(),
        #     nn.LayerNorm(self.lstm_hidden_size),
        #     nn.Linear(self.lstm_hidden_size, self.sequence_length),
        #     nn.LogSoftmax(dim=1),
        # )

        # self.right_head = nn.Sequential(
        #     nn.Linear(
        #         self.lstm_hidden_size,
        #         self.lstm_hidden_size,
        #     ),
        #     GeLU(),
        #     nn.LayerNorm(self.lstm_hidden_size),
        #     nn.Linear(self.lstm_hidden_size, self.sequence_length),
        #     nn.LogSoftmax(dim=1),
        # )

        # self.left_ptr_heads = nn.Linear(
        #     self.lstm_hidden_size,
        #     self.lstm_hidden_size,
        # )
        # self.left_ptr_targets = nn.Linear(
        #     self.lstm_hidden_size,
        #     self.lstm_hidden_size,
        # )
        self.left_ptr_hiddens = nn.Linear(
            self.lstm_hidden_size,
            self.lstm_hidden_size,
        )
        self.left_ptr_proj = nn.Sequential(
            GeLU(),
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Linear(
                self.lstm_hidden_size,
                1,
            ),
        )

        # self.right_ptr_heads = nn.Linear(
        #     self.lstm_hidden_size,
        #     self.lstm_hidden_size,
        # )
        # self.right_ptr_targets = nn.Linear(
        #     self.lstm_hidden_size,
        #     self.lstm_hidden_size,
        # )
        self.right_ptr_hiddens = nn.Linear(
            self.lstm_hidden_size,
            self.lstm_hidden_size,
        )
        self.right_ptr_proj = nn.Sequential(
            GeLU(),
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Linear(
                self.lstm_hidden_size,
                1,
            ),
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            heads,
            hiddens,
            targets,
    ):
        # targets = self.adapter(targets)
        # actions = self.action_head(targets + heads)
        actions = self.action_head(heads)

        # lefts = self.left_head(targets + heads)
        # rights = self.right_head(targets + heads)

        lefts = self.left_ptr_proj(
            # self.left_ptr_heads(
            #     heads
            # ).unsqueeze(1).expand(hiddens.size()) +
            # self.left_ptr_targets(
            #     targets
            # ).unsqueeze(1).expand(hiddens.size()) +
            self.left_ptr_hiddens(hiddens)
        ).squeeze(2)

        rights = self.right_ptr_proj(
            # self.right_ptr_heads(
            #     heads
            # ).unsqueeze(1).expand(hiddens.size()) +
            # self.right_ptr_targets(
            #     targets
            # ).unsqueeze(1).expand(hiddens.size()) +
            self.right_ptr_hiddens(hiddens)
        ).squeeze(2)

        lefts = self.log_softmax(lefts)
        rights = self.log_softmax(rights)

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
            nn.LayerNorm(self.lstm_hidden_size),
            nn.Linear(
                self.lstm_hidden_size,
                4*self.lstm_hidden_size,
            ),
            GeLU(),
            nn.Linear(4*self.lstm_hidden_size, 1),
            nn.ReLU(),
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
