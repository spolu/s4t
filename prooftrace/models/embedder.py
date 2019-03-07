import argparse
import os
import torch
import torch.nn as nn
import typing

from dataset.prooftrace import \
    Term, Type, Action, \
    ACTION_TOKENS, ProofTraceLMDataset

from generic.tree_lstm import BinaryTreeLSTM

from utils.config import Config


class TypeEmbedder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(TypeEmbedder, self).__init__()

        self.device = torch.device(config.get('device'))

        self.type_token_count = \
            config.get('prooftrace_type_token_count')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')

        self.type_token_embedder = nn.Embedding(
            self.type_token_count, self.hidden_size,
        )

        self.tree_lstm = BinaryTreeLSTM(self.hidden_size)
        self.tree_lstm.to(self.device)

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            types: typing.List[Type],
    ):
        def embedder(values):
            return self.type_token_embedder(
                torch.tensor(values, dtype=torch.int64).to(self.device),
            )

        h, _ = self.tree_lstm.batch(types, embedder)
        return h


class TermEmbedder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(TermEmbedder, self).__init__()

        self.device = torch.device(config.get('device'))

        self.term_token_count = \
            config.get('prooftrace_term_token_count')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')

        self.type_embedder = TypeEmbedder(config)

        self.term_token_embedder = nn.Embedding(
            self.term_token_count, self.hidden_size,
        )

        self.tree_lstm = BinaryTreeLSTM(self.hidden_size)
        self.tree_lstm.to(self.device)

    def extract_types(
            self,
            terms: typing.List[Term],
    ) -> typing.List[Type]:
        seen = {}

        def dfs(term):
            if term.hash() in seen:
                return []
            seen[term.hash()] = True

            if type(term.value) is Type:
                if term.value.hash() not in seen:
                    seen[term.value.hash()] = True
                    return [term.value]
                else:
                    return []
            else:
                left = []
                if term.left is not None:
                    left = dfs(term.left)
                right = []
                if term.right is not None:
                    right = dfs(term.right)
                return left + right

        types = []
        for t in terms:
            types += dfs(t)

        return types

    def extract_tokens(
            self,
            terms: typing.List[Term],
    ) -> typing.List[int]:
        seen = {}

        def dfs(term):
            if term.hash() in seen:
                return []
            seen[term.hash()] = True

            if type(term.value) is not Type:
                left = []
                if term.left is not None:
                    left = dfs(term.left)
                right = []
                if term.right is not None:
                    right = dfs(term.right)

                if term.value not in seen:
                    seen[term.value] = True
                    return left + right + [term.value]
                else:
                    return left + right + []
            else:
                return []

        tokens = []
        for t in terms:
            tokens += dfs(t)

        return tokens

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            terms: typing.List[Term],
    ):
        cache = {}

        # TODO(stan): optmize by sharing extraction
        types = self.extract_types(terms)
        tokens = self.extract_tokens(terms)

        if len(types) > 0:
            types_embeds = self.type_embedder(types)
            for i, ty in enumerate(types):
                cache[ty.hash()] = types_embeds[i].unsqueeze(0)

        if len(tokens) > 0:
            tokens_embeds = self.term_token_embedder(
                torch.tensor(tokens, dtype=torch.int64).to(self.device),
            )
            for i, v in enumerate(tokens):
                cache[v] = tokens_embeds[i].unsqueeze(0)

        def embedder(values):
            embeds = [[]] * len(values)
            for idx, v in enumerate(values):
                if type(v) is Type:
                    embeds[idx] = cache[v.hash()]
                else:
                    embeds[idx] = cache[v]
            return torch.cat(embeds, dim=0)

        h, _ = self.tree_lstm.batch(terms, embedder)
        return h


class ActionEmbedder(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(ActionEmbedder, self).__init__()

        self.device = torch.device(config.get('device'))

        self.hidden_size = \
            config.get('prooftrace_hidden_size')

        self.term_embedder = TermEmbedder(config)

        self.action_token_embedder = nn.Embedding(
            len(ACTION_TOKENS), self.hidden_size,
        )

        self.tree_lstm = BinaryTreeLSTM(self.hidden_size)
        self.tree_lstm.to(self.device)

    def extract_terms(
            self,
            actions: typing.List[Action],
    ) -> typing.List[Term]:
        seen = {}

        def dfs(action):
            if action.hash() in seen:
                return []
            seen[action.hash()] = True
            if type(action.value) is Term:
                if action.value.hash() not in seen:
                    seen[action.value.hash()] = True
                    return [action.value]
                else:
                    return []
            else:
                left = []
                if action.left is not None:
                    left = dfs(action.left)
                right = []
                if action.right is not None:
                    right = dfs(action.right)

                return left + right

        terms = []
        for a in actions:
            terms += dfs(a)

        return terms

    def extract_types(
            self,
            actions: typing.List[Action],
    ) -> typing.List[Type]:
        seen = {}

        def dfs(action):
            if action.hash() in seen:
                return []
            seen[action.hash()] = True
            if type(action.value) is Type:
                if action.value.hash() not in seen:
                    seen[action.value.hash()] = True
                    return [action.value]
                else:
                    return []
            else:
                left = []
                if action.left is not None:
                    left = dfs(action.left)
                right = []
                if action.right is not None:
                    right = dfs(action.right)

                return left + right

        types = []
        for a in actions:
            types += dfs(a)

        return types

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            actions: typing.List[
                typing.List[Action],
            ]
    ):
        flat = []
        for a in actions:
            flat += a

        cache = {}

        # TODO(stan): optmize by sharing extraction
        terms = self.extract_terms(flat)
        types = self.extract_types(flat)

        if len(terms) > 0:
            terms_embeds = self.term_embedder(terms)
            for i, tm in enumerate(terms):
                cache[tm.hash()] = terms_embeds[i].unsqueeze(0)

        if len(types) > 0:
            types_embeds = self.term_embedder.type_embedder(types)
            for i, ty in enumerate(types):
                cache[ty.hash()] = types_embeds[i].unsqueeze(0)

        tokens_embeds = self.action_token_embedder(
            torch.tensor(
                list(ACTION_TOKENS.values()),
                dtype=torch.int64
            ).to(self.device),
        )
        for i, tk in enumerate(list(ACTION_TOKENS.values())):
            cache[tk] = tokens_embeds[i].unsqueeze(0)

        def embedder(values):
            embeds = [[]] * len(values)
            for idx, v in enumerate(values):
                if type(v) is Term:
                    embeds[idx] = cache[v.hash()]
                elif type(v) is Type:
                    embeds[idx] = cache[v.hash()]
                else:
                    embeds[idx] = cache[v]
            return torch.cat(embeds, dim=0)

        h, _ = self.tree_lstm.batch(flat, embedder)

        # This assumes that all received action lists have equal size.
        return torch.cat(
            [t.unsqueeze(0) for t in torch.chunk(h, len(actions), dim=0)],
            dim=0,
        )


def test():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="congif override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    # train_set = ProofTraceLMDataset(
    #     os.path.expanduser(config.get('prooftrace_dataset_dir')),
    #     config.get('prooftrace_dataset_size'),
    #     False,
    #     config.get('prooftrace_sequence_length'),
    # )
    test_set = ProofTraceLMDataset(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        True,
        config.get('prooftrace_sequence_length'),
    )

    embedder = ActionEmbedder(config)

    device = torch.device(config.get('device'))
    embedder.to(device)

    indices = []
    traces = []

    for i in range(test_set.__len__()):
        idx, tr = test_set.__getitem__(i)
        indices.append(idx)
        traces.append(tr)

    embeds = embedder(traces[0:32])
