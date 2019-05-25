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

from generic.iota import IOTARollout, IOTAAgg, IOTARll

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

from prooftrace.prooftrace import ProofTraceActions, INV_PREPARE_TOKENS

from prooftrace.beam import Beam, BeamModel

from prooftrace.repl.repl import REPL

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


GAMMAS = [8, 16, 32, 64, 128, 256, 512, 1024]


class Rollout(IOTARollout):
    def __init__(
            self,
            name: str,
            positives: typing.List[ProofTraceActions],
            negatives: typing.List[ProofTraceActions],
            capacity: int = 5,
    ) -> None:
        super(IOTARollout, self).__init__()

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


class RLL():
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._modules = {
            'E': E(self._config).to(self._device),
            'T': T(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

        self._rollout_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_beam_rollout_dir')),
            config.get('prooftrace_dataset_size'),
        )
        with gzip.open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._rll = IOTARll(
            config.get('prooftrace_beam_iota_sync_dir'),
            self._modules,
        )

        Log.out('RLL initialization', {})

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

    def run_once(
            self,
    ):
        info = self._rll.fetch(self._device, False)
        if info is not None:
            self.update(info['config'])

        self._modules['E'].eval()
        self._modules['T'].eval()
        self._modules['PH'].eval()
        self._modules['VH'].eval()

        assert os.path.isdir(self._rollout_dir)

        rollout_dirs = [
            os.path.join(self._rollout_dir, d)
            for d in os.listdir(self._rollout_dir)
            if os.path.isdir(os.path.join(self._rollout_dir, d))
        ]

        rdir = random.choice(rollout_dirs)
        rfiles = sorted([
            os.path.join(rdir, f)
            for f in os.listdir(rdir) if re.search(".rollout$", f)
        ], reverse=True)

        if len(rfiles) == 0:
            return

        path = rfiles[0]
        with gzip.open(path, 'rb') as f:
            rollout = pickle.load(f)

        gamma = random.choice(GAMMAS)

        ground = rollout.positive()
        name = rollout.name()
        del rollout

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

        model = BeamModel(self._config, self._modules)
        beam = Beam(self._config, model, ptra, repl, target)

        Log.out("ROLLOUT START", {
            'name': ground.name(),
            'gamma': gamma,
            'prepare_length': ground.prepare_len(),
            'action_length': ground.action_len(),
            'length': ground.len(),
        })

        rollout = None
        proven = False
        ptra = None

        for i in range(gamma):
            ptra, proven = beam.step(i == (gamma-1), offset)
            if ptra is not None:
                if proven:
                    rollout = Rollout(name, [ptra], [])
                else:
                    rollout = Rollout(name, [], [ptra])
                break

        demo_length = (ptra.len() - (ground.prepare_len() + gamma_len))

        Log.out("ROLLOUT END", {
            'name': ground.name(),
            'proven': proven,
            'gamma': gamma,
            'demo_length': demo_length,
        })

        info = {
            'rll_cnt': 1,
            'pos_cnt': 1 if proven else 0,
            'neg_cnt': 0 if proven else 1,
        }
        if proven:
            info['demo_len'] = demo_length

        self._rll.publish(info, rollout)


class AGG():
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._rollout_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_beam_rollout_dir')),
            config.get('prooftrace_dataset_size'),
        )

        self._epoch = 0

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        Log.out("AGG Initializing", {})

        self._agg = IOTAAgg(
            config.get('prooftrace_beam_iota_sync_dir'),
        )

    def update(
            self,
    ) -> None:
        self._config.update()

    def run_once(
            self,
    ):
        run_start = time.time()

        rollouts, infos = self._agg.aggregate()

        if len(infos) == 0:
            time.sleep(60)
            return

        # merge rollouts and atomic_write to new name
        for r in rollouts:
            rdir = os.path.join(self._rollout_dir, r.name())

            rfiles = sorted([
                os.path.join(rdir, f)
                for f in os.listdir(rdir) if re.search(".rollout$", f)
            ], reverse=True)

            assert len(rfiles) > 0

            path = rfiles[0]
            with gzip.open(path, 'rb') as f:
                rollout = pickle.load(f)

            rollout.merge(r)

            now = datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f")
            rnd = random.randint(0, 10e9)

            tmp_path = os.path.join(rdir, "{}_{}.tmp".format(now, rnd))
            fnl_path = os.path.join(rdir, "{}_{}.rollout".format(now, rnd))

            with gzip.open(tmp_path, 'wb') as f:
                pickle.dump(
                    rollout, f, protocol=pickle.HIGHEST_PROTOCOL
                )
            os.rename(tmp_path, fnl_path)

            if len(rfiles) > 1:
                for p in rfiles[1:]:
                    os.remove(p)

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

        Log.out("PROOFTRACE BEAM AGG RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'rll_cnt': "{:.4f}".format(rll_cnt_meter.sum or 0.0),
            'pos_cnt': "{:.4f}".format(pos_cnt_meter.avg or 0.0),
            'demo_len': "{:.4f}".format(demo_len_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            if rll_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_agg/rll_cnt",
                    rll_cnt_meter.sum, self._epoch,
                )
            if pos_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_agg/pos_cnt",
                    pos_cnt_meter.avg, self._epoch,
                )
            if neg_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_agg/neg_cnt",
                    neg_cnt_meter.avg, self._epoch,
                )
            if demo_len_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_agg/demo_len",
                    demo_len_meter.avg, self._epoch,
                )

        self._epoch += 1


def rll_run():
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
    if args.sync_dir is not None:
        config.override(
            'prooftrace_beam_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_beam_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    rll = RLL(config)

    while True:
        rll.run_once()


def agg_run():
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
    parser.add_argument(
        '--rollout_dir',
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
            'prooftrace_beam_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_beam_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    agg = AGG(config)

    while True:
        agg.run_once()


def translate(
        args,
):
    config, path, idx = args

    with gzip.open(path, 'rb') as f:
        ptra = pickle.load(f)

    rollout = Rollout(ptra.name(), [ptra], [])

    rollout_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_beam_rollout_dir')),
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
            'prooftrace_beam_rollout_dir',
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
