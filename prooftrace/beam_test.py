import argparse
import datetime
import gzip
import os
import pickle
import random
import re
import time
import torch

from generic.iota import IOTATst, IOTACtl

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


GAMMAS = [2, 4, 8, 16]


class TST():
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

        self._dataset_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            config.get('prooftrace_dataset_size'),
            'test_traces',
        )
        with gzip.open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._tst = IOTATst(
            config.get('prooftrace_beam_iota_sync_dir'),
            self._modules,
        )

        self._test_gamma_size = config.get('prooftrace_beam_test_gamma_size')

        Log.out('TST initialization', {})

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

    def run_once(
            self,
    ):
        info = self._tst.fetch(self._device, False)
        if info is not None:
            self.update(info['config'])

        self._modules['E'].eval()
        self._modules['T'].eval()
        self._modules['PH'].eval()
        self._modules['VH'].eval()

        model = BeamModel(self._config, self._modules)

        assert os.path.isdir(self._dataset_dir)
        files = [
            os.path.join(self._dataset_dir, f)
            for f in os.listdir(self._dataset_dir)
            if os.path.isfile(os.path.join(self._dataset_dir, f))
        ]

        cases = {}
        for gamma in GAMMAS:
            cases[gamma] = []

        for p in files:
            match = re.search("_(\\d+)_(\\d+)\\.actions$", p)
            if match is None:
                continue
            for gamma in GAMMAS:
                cases[gamma].append(p)

        info = {
            'demo_len': 0.0,
        }
        for gamma in GAMMAS:
            cases[gamma] = random.sample(cases[gamma], self._test_gamma_size)
            info['gamma_{}'.format(gamma)] = 0.0

        for gamma in GAMMAS:
            for i in range(len(cases)):
                c = cases[i]
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
                repl = REPL(self._tokenizer)
                target = repl.prepare(ptra)

                offset = 0
                gamma_len = max(ground.action_len() - gamma, 0)
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

                beam = Beam(self._config, model, ptra, repl, target)

                proven = False
                ptra = None

                for i in range(gamma):
                    step_start = time.time()
                    done, ptra, proven = beam.step(i == (gamma-1), offset)
                    step_end = time.time()
                    if done:
                        break
                    if (step_end - step_start) > \
                            self._config.get('prooftrace_beam_step_timeout'):
                        break

                demo_length = (ptra.len() - (ground.prepare_len() + gamma_len))

                Log.out("DONE", {
                    'name': ground.name(),
                    'proven': proven,
                    'gamma': gamma,
                    'demo_length': demo_length,
                })

                if proven:
                    info['gamma_{}'.format(gamma)] += \
                        1.0 / self._test_gamma_size
                info['demo_len'] += \
                    demo_length / (self._test_gamma_size * len(GAMMAS))

        self._tst.publish(info)


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

        infos = self._ctl.aggregate()

        if len(infos) == 0:
            time.sleep(10)
            return

        demo_len_meter = Meter()
        gamma_meters = {}
        for gamma in GAMMAS:
            gamma_meters['gamma_{}'.format(gamma)] = Meter()

        for info in infos:
            demo_len_meter.update(info['demo_len'])
            for gamma in GAMMAS:
                key = 'gamma_{}'.format(gamma)
                gamma_meters[key].update(info[key])

        out = {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'demo_len': "{:.4f}".format(demo_len_meter.avg or 0.0),
        }
        for gamma in GAMMAS:
            key = 'gamma_{}'.format(gamma)
            out[key] = "{:.4f}".format(gamma_meters[key].avg or 0.0)
        Log.out("PROOFTRACE BEAM CTL RUN", out)

        if self._tb_writer is not None:
            if demo_len_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_agg/demo_len",
                    demo_len_meter.avg, self._epoch,
                )
            for gamma in GAMMAS:
                key = 'gamma_{}'.format(gamma)
                if gamma_meters[key].avg is not None:
                    self._tb_writer.add_scalar(
                        "prooftrace_beam_ctl/{}".format(key),
                        gamma_meters[key].avg, self._epoch,
                    )

        self._epoch += 1


###############################################################################
# TST Run.
###############################################################################

def tst_run():
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
            'prooftrace_beam_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    tst = TST(config)

    while True:
        tst.run_once()


###############################################################################
# CTL Run.
###############################################################################

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
            'prooftrace_beam_iota_sync_dir',
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
