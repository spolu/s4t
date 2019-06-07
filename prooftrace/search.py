import argparse
import datetime
import gzip
import os
import pickle
import random
import re

from prooftrace.prooftrace import ProofTraceActions, INV_PREPARE_TOKENS

from prooftrace.repl.repl import REPL
from prooftrace.search_base import SearchModel
from prooftrace.beam import Beam
from prooftrace.mcts import MCTS

from utils.config import Config
from utils.log import Log


def search():
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

    parser.add_argument(
        '--device',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)

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

    dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        'train_traces'
    )

    assert os.path.isdir(dataset_dir)
    files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]
    cases = []

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    for p in files:
        match = re.search("_(\\d+)_(\\d+)\\.actions$", p)
        if match is None:
            continue
        ptra_len = int(match.group(1))
        cases.append((p, ptra_len))

    Log.out(
        "Loaded ProofTraceActions", {
            'cases': len(cases),
        })

    model = SearchModel(config).load()

    cases = sorted(cases, key=lambda c: c[1])

    for i in range(len(cases)):
        c = cases[i][0]
        with gzip.open(c, 'rb') as f:
            ground = pickle.load(f)

        ptra = ProofTraceActions(
            'BEAM-{}-{}'.format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
                random.randint(0, 9999),
            ),
            [
                ground.actions()[i] for i in range(ground.len())
                if ground.actions()[i].value in INV_PREPARE_TOKENS
            ],
            [
                ground.arguments()[i] for i in range(ground.len())
                if ground.actions()[i].value in INV_PREPARE_TOKENS
            ],
        )
        repl = REPL(tokenizer)
        target = repl.prepare(ptra)

        offset = 0
        fixed_gamma = 4
        if fixed_gamma > 0:
            gamma_len = max(ground.action_len() - fixed_gamma, 0)
            offset = ground.prepare_len() + gamma_len

            for i in range(gamma_len):
                assert ground.prepare_len() + i < ground.len() - 1
                pos = ground.prepare_len() + i

                action = ground.actions()[pos]
                argument = ground.arguments()[pos]

                thm = repl.apply(action)

                action._index = thm.index()
                argument._index = thm.index()

                ptra.append(action, argument)

        Log.out("TARGET", {
            'name': ground.name(),
            'prepare_length': ground.prepare_len(),
            'length': ground.action_len(),
            'summary': ground.summary(offset),
        })

        search = None
        if config.get('prooftrace_search_type') == 'beam':
            search = Beam(config, model, ptra, repl, target)
        if config.get('prooftrace_search_type') == 'mcts':
            search = MCTS(config, model, ptra, repl, target)
        assert search is not None

        depth = config.get('prooftrace_search_depth')
        if config.get('prooftrace_search_type') == 'beam':
            depth = fixed_gamma * 2

        for i in range(depth):
            done, ptra, proved = search.step(False, offset)
            if done:
                break
