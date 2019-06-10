import argparse
import datetime
import concurrent.futures
import gzip
import os
import pickle
import random
import re
import typing

from prooftrace.prooftrace import ProofTraceActions, INV_PREPARE_TOKENS

from prooftrace.models.model import Model
from prooftrace.repl.repl import REPL
from prooftrace.beam import Beam
from prooftrace.mcts import MCTS

from utils.config import Config
from utils.log import Log
from utils.str2bool import str2bool


class Rollout():
    def __init__(
            self,
            name: str,
            positives: typing.List[ProofTraceActions],
            negatives: typing.List[ProofTraceActions],
            capacity: int = 5,
    ) -> None:
        self._name = name
        self._positives = positives
        self._negatives = negatives
        self._capacity = capacity

    def name(
            self,
    ) -> str:
        return self._name

    def merge(
            self,
            other,
    ):
        assert other.name() == self.name()
        assert other._capacity == self._capacity

        # We keep the shortest positives
        for i in range(len(other._positives)):
            self._positives = [other._positives[i]] + self._positives
        self._positives = sorted(
            self._positives, key=lambda p: p.len(),
        )[:self._capacity]

        # We keep the most recent negatives
        for i in range(len(other._negatives)):
            self._negatives = [other._negatives[i]] + self._negatives
        self._negatives = self._negatives[:self._capacity]

        return self

    def positive(
            self,
    ) -> ProofTraceActions:
        assert len(self._positives) > 0
        return random.choice(self._positives).copy()

    def random(
            self,
    ) -> typing.Tuple[ProofTraceActions, bool]:
        choices = []
        for p in self._positives:
            choices += [(p, True)]
        for p in self._negatives:
            choices += [(p, False)]

        assert len(choices) > 0

        ptra, outcome = random.choice(choices)
        return (ptra.copy(), outcome)


###############################################################################
# Rollout bootstrapping.
###############################################################################

def translate(
        args,
):
    config, test, path, idx = args

    with gzip.open(path, 'rb') as f:
        ptra = pickle.load(f)

    rollout = Rollout(ptra.name(), [ptra], [])

    if test:
        rollout_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'test_rollouts',
        )
    else:
        rollout_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'train_rollouts',
        )

    rdir = os.path.join(rollout_dir, rollout.name())
    if not os.path.exists(rdir):
        os.mkdir(rdir)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f")
    rnd = random.randint(0, 10e9)

    tmp_path = os.path.join(rdir, "{}_{}.tmp".format(now, rnd))
    fnl_path = os.path.join(rdir, "{}_{}.rollout".format(now, rnd))

    with gzip.open(tmp_path, 'wb') as f:
        pickle.dump(
            rollout, f, protocol=pickle.HIGHEST_PROTOCOL
        )
    os.rename(tmp_path, fnl_path)

    Log.out("Writing Rollout", {
        'path': fnl_path,
        'index': idx,
    })


def bootstrap():
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
        '--rollout_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--test',
        type=str2bool, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.rollout_dir is not None:
        config.override(
            'prooftrace_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )
    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    test = False
    if args.test is not None:
        test = args.test

    if test:
        dataset_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            config.get('prooftrace_dataset_size'),
            'test_traces'
        )
    else:
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

    executor = concurrent.futures.ProcessPoolExecutor()

    map_args = []
    for i, path in enumerate(files):
        map_args.append([config, test, path, i])

    executor.map(translate, map_args)


###############################################################################
# Rollout inspection.
###############################################################################

def read(
        args,
):
    config, rdir, idx = args

    rfiles = sorted([
        os.path.join(rdir, f)
        for f in os.listdir(rdir) if re.search(".rollout$", f)
    ], reverse=True)

    with gzip.open(rfiles[0], 'rb') as f:
        rollout = pickle.load(f)

    Log.out("Rollout", {
        'rdir': rdir,
        'positives': len(rollout._positives),
        'negatives': len(rollout._negatives),
        'index': idx,
    })

    return (len(rollout._positives), len(rollout._negatives))


def inspect():
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
        '--rollout_dir',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.rollout_dir is not None:
        config.override(
            'prooftrace_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )
    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    rollout_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_rollout_dir')),
        config.get('prooftrace_dataset_size'),
    )

    assert os.path.isdir(rollout_dir)
    rdirs = [
        os.path.join(rollout_dir, f)
        for f in os.listdir(rollout_dir)
        if os.path.isdir(os.path.join(rollout_dir, f))
    ]

    executor = concurrent.futures.ProcessPoolExecutor()

    map_args = []
    for i, rdir in enumerate(rdirs):
        map_args.append([config, rdir, i])

    values = executor.map(read, map_args)

    positives = 0
    negatives = 0
    for p, n in values:
        if p > 0:
            positives += 1
        if n > 0:
            negatives += 1

    Log.out("Summary", {
        'positives': positives,
        'negatives': negatives,
    })


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
