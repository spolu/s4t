import argparse
import datetime
import gzip
import pickle
import os
import random
import re
import sys
import time
import torch
import typing

from prooftrace.prooftrace import INV_PREPARE_TOKENS, ACTION_TOKENS, \
    ProofTraceActions

from prooftrace.models.model import Model
from prooftrace.repl.repl import REPL
from prooftrace.search.beam import Beam
from prooftrace.search.particle_filter import ParticleFilter
from prooftrace.search.policy_sample import PolicySample

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
        'test_traces'
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

    model = Model(config).load()

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
        fixed_gamma = config.get('prooftrace_search_fixed_gamma')
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
        if config.get('prooftrace_search_type') == 'particle_filter':
            search = ParticleFilter(config, model, ptra, repl, target)
        if config.get('prooftrace_search_type') == 'policy_sample':
            search = PolicySample(config, model, ptra, repl, target)
        assert search is not None

        depth = fixed_gamma * 4

        for i in range(depth):
            step_start = time.time()
            done, ptra, proved = search.step(offset)
            step_end = time.time()

            Log.out('STEP', {
                'i': i,
                'done': done,
                'proved': proved,
                'time': "{:.2f}".format(step_end - step_start),
            })
            if done:
                if proved:
                    Log.out("DEMONSTRATED", {
                        'theorem': thm.thm_string(True),
                    })
                break

            if (step_end - step_start) > \
                    config.get('prooftrace_search_step_timeout'):
                break

        Log.out("FINISH", {
            'summary': ptra.summary(offset),
        })
