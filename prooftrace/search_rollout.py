import argparse
import concurrent.futures
import datetime
import gzip
import os
import pickle
import random
import re
import time
import torch
import typing

from generic.iota import IOTACtl, IOTAWrk

from prooftrace.prooftrace import ProofTraceActions, INV_PREPARE_TOKENS

from prooftrace.search_base import SearchModel
from prooftrace.beam import Beam
from prooftrace.mcts import MCTS

from prooftrace.repl.repl import REPL

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


GAMMAS = [8, 16, 32, 64, 128, 256, 512, 1024]


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


class WRK():
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._model = SearchModel(config)

        self._rollout_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_search_rollout_dir')),
            config.get('prooftrace_dataset_size'),
        )
        with gzip.open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._wrk = IOTAWrk(
            config.get('prooftrace_search_iota_sync_dir'),
            'rollout',
            self._model.modules(),
        )

        self._type = config.get('prooftrace_search_type')
        self._depth = config.get('prooftrace_search_depth')

        Log.out('WRK initialization', {})

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

        depth = self._config.get('prooftrace_search_depth')
        if depth != self._depth:
            self._depth = depth
            Log.out("Updated", {
                "prooftrace_search_depth": depth,
            })
        t = self._config.get('prooftrace_search_type')
        if t != self._type:
            self._type = t
            Log.out("Updated", {
                "prooftrace_search_type": t,
            })

    def run_once(
            self,
    ):
        info = self._wrk.fetch(self._device, False)
        if info is not None:
            self.update(info['config'])

        for m in self._model.modules():
            self._model.modules()[m].eval()

        assert os.path.isdir(self._rollout_dir)

        rdirs = [
            os.path.join(self._rollout_dir, d)
            for d in os.listdir(self._rollout_dir)
            if os.path.isdir(os.path.join(self._rollout_dir, d))
        ]

        rdir = random.choice(rdirs)
        rfiles = sorted([
            os.path.join(rdir, f)
            for f in os.listdir(rdir) if re.search(".rollout$", f)
        ], reverse=True)

        if len(rfiles) == 0:
            return

        path = rfiles[0]
        with gzip.open(path, 'rb') as f:
            base = pickle.load(f)

        gamma = random.choice(GAMMAS)

        ground = base.positive()
        name = base.name()

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
        repl = REPL(self._tokenizer)
        target = repl.prepare(ptra)

        gamma = min(ground.action_len(), gamma)
        gamma_len = ground.action_len() - gamma
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

        search = None
        if self._config.get('prooftrace_search_type') == 'beam':
            search = Beam(self._config, self._model, ptra, repl, target)
        if self._config.get('prooftrace_search_type') == 'mcts':
            search = MCTS(self._config, self._model, ptra, repl, target)
        assert search is not None

        Log.out("ROLLOUT START", {
            'name': name,
            'gamma': gamma,
            'prepare_length': ground.prepare_len(),
            'action_length': ground.action_len(),
            'length': ground.len(),
        })

        Log.out("TARGET", {
            'name': name,
            'summary': ground.summary(offset),
        })

        rollout = None
        proven = False
        ptra = None

        depth = self._config.get('prooftrace_search_depth')
        if self._config.get('prooftrace_search_type') == 'beam':
            depth = gamma * 2

        for i in range(depth):
            step_start = time.time()
            done, ptra, proven = search.step(i == (gamma-1), offset)
            step_end = time.time()
            Log.out('STEP', {
                'i': i,
                'done': done,
                'proven': proven,
                'gamma': gamma,
                'time': "{:.2f}".format(step_end - step_start),
            })
            if done:
                if proven:
                    rollout = Rollout(name, [ptra], [])
                else:
                    rollout = Rollout(name, [], [ptra])
                break
            if (step_end - step_start) > \
                    self._config.get('prooftrace_search_step_timeout'):
                rollout = Rollout(name, [], [ptra])
                break

        demo_length = (ptra.len() - (ground.prepare_len() + gamma_len))

        Log.out("ROLLOUT END", {
            'name': name,
            'proven': proven,
            'gamma': gamma,
            'demo_length': demo_length,
        })

        Log.out("PTRA", {
            'name': name,
            'summary': ptra.summary(offset),
        })

        if demo_length > 0:
            info = {
                'rll_cnt': 1,
                'pos_cnt': 1 if proven else 0,
                'neg_cnt': 0 if proven else 1,
            }
            if proven:
                info['demo_len'] = demo_length

            # Publish the statistics.
            self._wrk.publish(info)

            # Finally merge and store the new rollout
            base.merge(rollout)

            now = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f")
            rnd = random.randint(0, 10e9)

            tmp_path = os.path.join(rdir, "{}_{}.tmp".format(now, rnd))
            fnl_path = os.path.join(rdir, "{}_{}.rollout".format(now, rnd))

            with gzip.open(tmp_path, 'wb') as f:
                pickle.dump(
                    base, f, protocol=pickle.HIGHEST_PROTOCOL
                )
            os.rename(tmp_path, fnl_path)

            del base
            del rollout

            if len(rfiles) > 1:
                for p in rfiles[1:]:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass

            Log.out("MERGE WRITE", {
                'name': name,
                'path': fnl_path,
            })


class CTL():
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._epoch = 0

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        Log.out("CTL Initializing", {})

        self._ctl = IOTACtl(
            config.get('prooftrace_search_iota_sync_dir'),
            'rollout',
        )

        self._executor = concurrent.futures.ThreadPoolExecutor()

    def update(
            self,
    ) -> None:
        update = self._config.update()

        if self._tb_writer is not None:
            for k in update:
                if k in [
                        'prooftrace_search_type',
                        'prooftrace_search_depth',
                ]:
                    self._tb_writer.add_scalar(
                        "prooftrace_search_rollout_run/{}".format(k),
                        update[k], self._epoch,
                    )

    def run_once(
            self,
    ):
        run_start = time.time()

        infos = self._ctl.aggregate()

        if len(infos) == 0:
            time.sleep(10)
            return

        rll_cnt_meter = Meter()
        pos_cnt_meter = Meter()
        neg_cnt_meter = Meter()
        demo_len_meter = Meter()

        for info in infos:
            rll_cnt_meter.update(info['rll_cnt'])
            pos_cnt_meter.update(info['pos_cnt'])
            neg_cnt_meter.update(info['neg_cnt'])
            if 'demo_len' in info:
                demo_len_meter.update(info['demo_len'])

        Log.out("PROOFTRACE BEAM ROLLOUT CTL RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'rll_cnt': "{:.4f}".format(rll_cnt_meter.sum or 0.0),
            'pos_cnt': "{:.4f}".format(pos_cnt_meter.avg or 0.0),
            'neg_cnt': "{:.4f}".format(neg_cnt_meter.avg or 0.0),
            'demo_len': "{:.4f}".format(demo_len_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            if rll_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_rollout/rll_cnt",
                    rll_cnt_meter.sum, self._epoch,
                )
            if pos_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_rollout/pos_cnt",
                    pos_cnt_meter.avg, self._epoch,
                )
            if neg_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_rollout/neg_cnt",
                    neg_cnt_meter.avg, self._epoch,
                )
            if demo_len_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_rollout/demo_len",
                    demo_len_meter.avg, self._epoch,
                )

        self._epoch += 1


###############################################################################
# WRK run.
###############################################################################

# def wrk_run():
#     import cProfile
#     cProfile.runctx(
#         'wrk_run_profile()', globals(), locals(), 'wrk_run.profile'
#     )


def wrk_run():
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
        '--sync_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--rollout_dir',
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
    if args.sync_dir is not None:
        config.override(
            'prooftrace_search_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_search_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    wrk = WRK(config)

    while True:
        wrk.run_once()


###############################################################################
# CTL run.
###############################################################################

# def ctl_run():
#     import cProfile
#     cProfile.runctx(
#         'ctl_run_profile()', globals(), locals(), 'ctl_run.profile'
#     )


def ctl_run():
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
        '--tensorboard_log_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--sync_dir',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)
    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.sync_dir is not None:
        config.override(
            'prooftrace_search_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    ctl = CTL(config)

    while True:
        ctl.run_once()


###############################################################################
# Rollout bootstrapping.
###############################################################################

def translate(
        args,
):
    config, path, idx = args

    with gzip.open(path, 'rb') as f:
        ptra = pickle.load(f)

    rollout = Rollout(ptra.name(), [ptra], [])

    rollout_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_search_rollout_dir')),
        config.get('prooftrace_dataset_size'),
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

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.rollout_dir is not None:
        config.override(
            'prooftrace_search_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )
    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
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

    executor = concurrent.futures.ProcessPoolExecutor()

    map_args = []
    for i, path in enumerate(files):
        map_args.append([config, path, i])

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
            'prooftrace_search_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )
    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    rollout_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_search_rollout_dir')),
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
