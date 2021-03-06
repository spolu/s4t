import argparse
import datetime
import concurrent.futures
import gzip
import os
import pickle
import random
import re
import typing

from prooftrace.prooftrace import ProofTraceActions

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

    length = ptra.action_len()

    del ptra
    del rollout

    return length


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
        type=str2bool, help="bootstrap test set",
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

    Log.out('Processing prooftraces', {
        'count': len(files),
    })

    map_args = []
    for i, path in enumerate(files):
        map_args.append([config, test, path, i])

    total_length = 0
    STEP = 1000

    for i in range(0, len(map_args), STEP):
        args = map_args[i:i+STEP]

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=48,
        ) as executor:
            for l in executor.map(translate, args):
                total_length += l

        Log.out('Checkpoint', {
            'i': i,
            'len': len(map_args),
            'total_length': total_length,
        })

    # executor = concurrent.futures.ProcessPoolExecutor()
    # for l in executor.map(translate, map_args, chunksize=32):
    #     total_length += l

    Log.out('Processed all profotraces', {
        'total_length': total_length,
    })


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
        'train_rollouts',
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
