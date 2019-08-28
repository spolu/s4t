import torch
import torch.nn as nn

from prooftrace.prooftrace import PROOFTRACE_TOKENS, PREPARE_TOKENS

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
        self.head_hidden_size = \
            config.get('prooftrace_head_hidden_size')

        self.action_head = nn.Sequential(
            nn.Linear(
                self.head_hidden_size,
                self.head_hidden_size,
            ),
            GeLU(),
            nn.LayerNorm(self.head_hidden_size),
            nn.Linear(
                self.head_hidden_size,
                len(PROOFTRACE_TOKENS) - len(PREPARE_TOKENS)
            ),
            nn.LogSoftmax(dim=2),
        )

        self.left_ptr_heads = nn.Linear(
            self.head_hidden_size,
            self.head_hidden_size,
        )
        self.left_ptr_hiddens = nn.Linear(
            self.head_hidden_size,
            self.head_hidden_size,
        )
        self.left_ptr_proj = nn.Sequential(
            GeLU(),
            nn.LayerNorm(self.head_hidden_size),
            nn.Linear(
                self.head_hidden_size,
                1,
            ),
        )

        self.right_ptr_heads = nn.Linear(
            self.head_hidden_size,
            self.head_hidden_size,
        )
        self.right_ptr_hiddens = nn.Linear(
            self.head_hidden_size,
            self.head_hidden_size,
        )
        self.right_ptr_proj = nn.Sequential(
            GeLU(),
            nn.LayerNorm(self.head_hidden_size),
            nn.Linear(
                self.head_hidden_size,
                1,
            ),
        )

        self.log_softmax = nn.LogSoftmax(dim=2)

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            hiddens,
            heads,
    ):
        actions = self.action_head(heads)

        left_hiddens = self.left_ptr_hiddens(hiddens).unsqueeze(1).expand(
            hiddens.size(0),
            heads.size(1),
            hiddens.size(1),
            hiddens.size(2),
        )
        left_heads = self.left_ptr_heads(heads).unsqueeze(2).expand(
            heads.size(0),
            heads.size(1),
            hiddens.size(1),
            heads.size(2),
        )
        lefts = self.left_ptr_proj(left_hiddens + left_heads).squeeze(-1)

        right_hiddens = self.right_ptr_hiddens(hiddens).unsqueeze(1).expand(
            hiddens.size(0),
            heads.size(1),
            hiddens.size(1),
            hiddens.size(2),
        )
        right_heads = self.right_ptr_heads(heads).unsqueeze(2).expand(
            heads.size(0),
            heads.size(1),
            hiddens.size(1),
            heads.size(2),
        )
        rights = self.right_ptr_proj(right_hiddens + right_heads).squeeze(-1)

        lefts = self.log_softmax(lefts)
        rights = self.log_softmax(rights)

        return actions, lefts, rights
