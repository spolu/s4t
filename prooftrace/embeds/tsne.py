import argparse
import base64
import gzip
import numpy as np
import os
import pickle
import torch
import typing
import re

from prooftrace.prooftrace import \
    INV_ACTION_TOKENS, ProofTraceActions, Action

from prooftrace.models.embedder import E

from sklearn.manifold import TSNE

from utils.config import Config
from utils.log import Log


class Embed():
    def __init__(
            self,
            action: Action,
            embed: typing.List[float],
            ptras: typing.List[ProofTraceActions],
    ) -> None:
        self._embed = embed
        self._prooftraces = [ptra.name() for ptra in ptras]
        self._action = action.value

    def embed(
            self,
    ) -> typing.List[float]:
        return self._embed

    def prooftraces(
            self,
    ) -> typing.List[str]:
        return self._prooftraces

    def action(
            self,
    ) -> str:
        return INV_ACTION_TOKENS[self._action]

    def __iter__(
            self,
    ):
        yield 'embed', self._embed
        yield 'action', self.action()
        yield 'prooftraces', self.prooftraces()


class ProofTraceEmbeds():
    def __init__(
            self,
    ) -> None:
        self._embeds = {}

    def size(
            self,
    ) -> int:
        return len(self._embeds)

    def add(
            self,
            action: Action,
            argument: Action,
            embed: typing.List[float],
            ptras: typing.List[ProofTraceActions],
    ) -> None:
        self._embeds[argument.hash()] = Embed(action, embed, ptras)

    def get(
            self,
            action: Action,
    ) -> Embed:
        assert action.hash() in self._embeds
        return self._embeds[action.hash()]

    def __iter__(
            self,
    ):
        for h in self._embeds:
            yield base64.b64encode(h).decode('utf-8'), dict(self._embeds[h])


class TreeLSTMEmbedder:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._load_dir = config.get('prooftrace_load_dir')

        self._model_E = E(config).to(self._device)

        Log.out("Initializing prooftrace Embedder", {
            'parameter_count_E': self._model_E.parameters_count(),
        })

    def load(
            self,
    ):
        if self._load_dir:
            if os.path.isfile(self._load_dir + "/model_E.pt"):
                Log.out('Loading E', {
                    'load_dir': self._load_dir,
                })
                self._model_E.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_E.pt",
                        map_location=self._device,
                    ),
                )

        return self

    def embed(
            self,
            batch: typing.List[ProofTraceActions],
    ) -> ProofTraceEmbeds:
        return self._model_E([ptra.arguments() for ptra in batch])


def extract():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )
    parser.add_argument(
        '--device',
        type=str, help="config override",
    )
    parser.add_argument(
        '--load_dir',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )
    if args.device is not None:
        config.override('device', args.device)
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )

    # with open(
    #         os.path.join(
    #             os.path.expanduser(config.get('prooftrace_dataset_dir')),
    #             config.get('prooftrace_dataset_size'),
    #             'traces.tokenizer',
    #         ), 'rb') as f:
    #     tokenizer = pickle.load(f)

    embedder = TreeLSTMEmbedder(config).load()
    tSNE = TSNE(n_components=2)

    def embed_dataset(dataset_dir):
        all_arguments = []
        all_actions = []
        all_embeds = []
        all_ptras = {}

        files = sorted(
            [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        )
        for p in files:
            if re.search("\\.actions$", p) is None:
                continue
            Log.out("Loading ProofTraceActions", {
                'path': p,
            })
            with gzip.open(p, 'rb') as f:
                ptra = pickle.load(f)
            Log.out("Embedding ProofTraceActions", {
                'name': ptra.name(),
            })
            embeds = embedder.embed([ptra]).cpu().data.numpy()
            for i in range(ptra.len()):
                action = ptra.actions()[i]
                argument = ptra.arguments()[i]
                if action.hash() not in all_ptras:
                    all_ptras[action.hash()] = []
                    all_arguments.append(argument)
                    all_actions.append(action)
                    all_embeds.append(embeds[0][i])
                all_ptras[action.hash()].append(ptra)

        Log.out("Running t-SNE on all embeds", {
            'embed_count': len(all_embeds),
        })
        tsne = tSNE.fit_transform(np.array(all_embeds))

        ptre = ProofTraceEmbeds()
        for i in range(len(all_actions)):
            action = all_actions[i]
            argument = all_arguments[i]
            ptre.add(
                action, argument,
                tsne[i].tolist(),
                all_ptras[action.hash()],
            )

        ptre_path = os.path.join(dataset_dir, 'traces.embeds')
        Log.out("Writing ProofTraceEmbeds", {
            'path': ptre_path,
        })
        with gzip.open(ptre_path, 'wb') as f:
            pickle.dump(ptre, f)

    test_dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        "test_traces",
    )
    embed_dataset(test_dataset_dir)

    train_dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        "train_traces",
    )
    embed_dataset(train_dataset_dir)
