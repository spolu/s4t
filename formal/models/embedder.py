import torch
import torch.nn as nn
import typing

from generic.tree_lstm import TreeLSTM
from dataset.prooftrace import Term, Action


class TermEmbedder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(TermEmbedder, self).__init__()

        self.device = torch.device(config.get('device'))

        self.term_token_count = \
            config.get('formal_term_token_count')
        self.hidden_size = \
            config.get('formal_hidden_size')

        self.tree_lstm = TreeLSTM(self.term_token_count, self.hidden_size)

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            terms: typing.List[Term],
    ):
        h, _ = self.batch(terms)

        return h


class ActionEmbedder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(ActionEmbedder, self).__init__()

        self.term_token_count = \
            config.get('formal_action_token_count')
        self.hidden_size = \
            config.get('formal_hidden_size')

        self.term_embedder = TermEmbedder(config)

        self.action_token_embedding = nn.Embedding(
            self.action_token_count,
            self.hidden_size,
        )
        self.lstm = nn.LSTM(
            self.hidden_size, self.hidden_size,
            num_layers=1, bias=True, batch_first=True, dropout=0.0,
        )

    def forward(
            self,
            actions: typing.List[
                typing.List[Action],
            ]
    ):
        pass
