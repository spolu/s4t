import argparse
import os
import pickle
import torch
import typing
import re

from dataset.prooftrace import \
    ProofTraceActions, Action

from prooftrace.models.embedder import E

from sklearn.manifold import TSNE

from utils.config import Config
from utils.log import Log


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
            embed: typing.List[float],
    ) -> None:
        self._embeds[action.hash()] = embed

    def get(
            self,
            action: Action,
    ) -> typing.List[float]:
        assert action.hash() in self._embeds
        return self._embeds[action.hash()]


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
        return self._model_E([ptra.actions() for ptra in batch])


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
        all_actions = []
        all_embeds = []
        all_index = {}

        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        for p in files:
            if re.search("\\.actions$", p) is None:
                continue
            Log.out("Loading ProofTraceActions", {
                'path': p,
            })
            with open(p, 'rb') as f:
                ptra = pickle.load(f)
            Log.out("Embedding ProofTraceActions", {
                'name': ptra.name(),
            })
            embeds = embedder.embed([ptra]).cpu().data.numpy()
            for i, a in enumerate(ptra.actions()):
                if a.hash() not in all_index:
                    all_index[a.hash()] = len(all_embeds)
                    all_actions.append(a)
                    all_embeds.append(embeds[0][i])

        Log.out("Running t-SNE on all embeds", {
            'embed_count': len(all_embeds),
        })
        tsne = tSNE.fit_transform(all_embeds)

        ptre = ProofTraceEmbeds()
        for i, a in enumerate(all_actions):
            ptre.add(a, tsne[i])

        ptre_path = os.path.join(dataset_dir, 'tarces.embeds')
        Log.out("Writing ProofTraceEmbeds", {
            'path': ptre_path,
        })
        with open(ptre_path, 'wb') as f:
            pickle.dump(ptre, f)

    train_dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        "train_traces",
    )
    embed_dataset(train_dataset_dir)

    test_dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        "test_traces",
    )
    embed_dataset(test_dataset_dir)
