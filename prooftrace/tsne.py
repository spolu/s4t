import argparse
import os
import pickle
import torch
import re

from dataset.prooftrace import \
    ACTION_TOKENS, INV_ACTION_TOKENS, \
    ProofTraceTokenizer, Action, ProofTraceActions

from prooftrace.models.embedder import E

from sklearn.manifold import TSNE

from utils.config import Config
from utils.log import Log


def embed_targets():
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

    seen = {}

    premises = []
    targets = []
    names = []

    def add_dataset(dataset_dir):
        files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        for p in files:
            if re.search("\\.actions$", p) is None:
                continue
            with open(p, 'rb') as f:
                ptra = pickle.load(f)

            for a in ptra.actions():
                if a.value == ACTION_TOKENS['TARGET']:
                    if a.hash() not in seen:
                        seen[a.hash()] = True
                        targets.append([a])
                        names.append(ptra.name())
                if a.value == ACTION_TOKENS['PREMISE']:
                    if a.hash() not in premises:
                        seen[a.hash()] = True
                        premises.append([a])

    dataset_dir = "./data/prooftrace/{}/train_traces".format(
        config.get("prooftrace_dataset_size"),
    )
    Log.out('Loading train traces', {
        'dataset_dir': dataset_dir,
    })
    add_dataset(dataset_dir)

    dataset_dir = "./data/prooftrace/{}/test_traces".format(
        config.get("prooftrace_dataset_size"),
    )
    Log.out('Loading test traces', {
        'dataset_dir': dataset_dir,
    })
    add_dataset(dataset_dir)

    device = torch.device(config.get('device'))

    model_E = E(config).to(device)

    model_E.load_state_dict(torch.load(
        config.get('prooftrace_load_dir') + "/model_E_0.pt",
        map_location=device,
    ))
    Log.out('Loaded embedder', {
        'parameter_count': model_E.parameters_count(),
    })

    Log.out('Embedding premises', {
        'premise_count': len(premises),
    })
    premise_embeds = model_E(premises).squeeze(1)

    Log.out('Embedding targets', {
        'target_count': len(targets),
    })
    target_embeds = model_E(targets).squeeze(1)

    embeds = torch.cat((premise_embeds, target_embeds), dim=0)

    Log.out('Applying tSNE', {
        'embed_count': len(embeds),
    })

    tSNE = TSNE(n_components=2)
    Y = tSNE.fit_transform(embeds.cpu().data.numpy())

    Log.out('DONE')

    for i in range(len(premises)):
        print("PREMISE_{},{},{}".format(i, Y[i][0], Y[i][1]))
    for i in range(len(targets)):
        j = i + len(premises)
        print("{},{},{}".format(names[i], Y[j][0], Y[j][1]))
