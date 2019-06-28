import argparse
import gzip
import pickle
import os
import random
import sys
import torch
import typing

from prooftrace.prooftrace import ACTION_TOKENS, ProofTraceActions

from utils.config import Config
from utils.log import Log


def search_target(
        config: Config,
        target: str,
        query: str,
) -> typing.Optional[str]:
    dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        target,
    )

    assert os.path.isdir(dataset_dir)
    files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]

    for p in files:
        if "/"+query in p:
            return p

    return None


def search_path(
        config: Config,
        query: str,
) -> typing.Optional[str]:
    path = search_target(config, 'test_traces', query)
    if path is not None:
        return path
    path = search_target(config, 'train_traces', query)
    if path is not None:
        return path

    return None


def search_ptra(
        config: Config,
        query: str,
) -> typing.Tuple[
    str,
    typing.Optional[ProofTraceActions]
]:
    path = None

    path = search_path(config, query)
    if path is None:
        return None, None

    with gzip.open(path, 'rb') as f:
        ptra = pickle.load(f)

    return path, ptra


def dump():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'query',
        type=str, help="query on name of the prooftrace to dump",
    )
    parser.add_argument(
        'action',
        type=str, help="dump|premises",
    )

    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    torch.manual_seed(0)

    path, ptra = search_ptra(config, args.query)

    if ptra is None:
        Log.out("Prooftrace not found", {
            'dataset_size': config.get('prooftrace_dataset_size'),
            'query': args.query,
        })
        sys.exit()

    Log.out("Prooftrace found", {
        'dataset_size': config.get('prooftrace_dataset_size'),
        'name': ptra.name(),
        'len': ptra.len(),
        'prepare_len': ptra.prepare_len(),
        'path': path,
    })

    if args.action == 'dump':
        for a in ptra.actions():
            dump = dict(a)
            print(dump)
    elif args.action == 'premises':
        for i, a in enumerate(ptra.actions()):
            if a.value == ACTION_TOKENS['THEOREM'] and i > 0:
                dump = dict(a)
                path = search_path(config, "{}_".format(dump['index']))
                print(
                    "{} [{}] {}".format(
                        dump['type'],
                        dump['index'],
                        path,
                    ),
                )
                for h in dump['hyp']:
                    print("  {}".format(h))
                print("  |- {}".format(dump['ccl']))
    else:
        Log.out("Unknown action", {
            'action': args.action
        })
        sys.exit()


def generate_testset():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'count',
        type=int, help="",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    torch.manual_seed(0)
    random.seed(0)

    train_dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        'train_traces',
    )
    test_dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        'test_traces',
    )

    assert os.path.isdir(train_dataset_dir)
    files = [
        os.path.join(train_dataset_dir, f)
        for f in os.listdir(train_dataset_dir)
        if os.path.isfile(os.path.join(train_dataset_dir, f))
    ]

    testset = random.sample(files, args.count)
    for p in testset:
        print("mv {} {}/".format(p, test_dataset_dir))
