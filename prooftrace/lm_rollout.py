import argparse
import datetime
import gzip
import os
import pickle
import random
import re
import time
import torch

from generic.iota import IOTACtl, IOTAWrk

from prooftrace.models.model import LModel
from prooftrace.prooftrace import ProofTraceActions, INV_PREPARE_TOKENS
from prooftrace.rollout import Rollout
from prooftrace.search.beam import Beam
from prooftrace.search.particle_filter import ParticleFilter
from prooftrace.search.policy_sample import PolicySample

from prooftrace.repl.repl import REPL

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


class WRK():
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._model = LModel(config)

        self._rollout_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'train_rollouts',
        )
        with gzip.open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._wrk = IOTAWrk(
            config.get('prooftrace_lm_iota_sync_dir'),
            'rollout',
            self._model.modules(),
        )

        self._type = config.get('prooftrace_search_type')

        Log.out('WRK initialization', {})

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

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

        ground = base.positive()
        name = base.name()

        ptra = ProofTraceActions(
            'ROLLOUT-{}-{}'.format(
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

        search = None
        if self._config.get('prooftrace_search_type') == 'beam':
            search = Beam(
                self._config, self._model, ptra, repl, target,
            )
        if self._config.get('prooftrace_search_type') == 'particle_filter':
            search = ParticleFilter(
                self._config, self._model, ptra, repl, target,
            )
        if self._config.get('prooftrace_search_type') == 'policy_sample':
            search = PolicySample(
                self._config, self._model, ptra, repl, target,
            )
        assert search is not None

        depth = self._config.get('prooftrace_sequence_length') - \
            ground.prepare_len()

        if 2 * ground.action_len() < depth:
            depth = 2 * ground.action_len()

        Log.out("ROLLOUT START", {
            'name': name,
            'prepare_length': ground.prepare_len(),
            'action_length': ground.action_len(),
            'depth': depth,
        })

        rollout = None
        proved = False
        ptra = None

        for i in range(depth):
            step_start = time.time()
            done, ptra, proved = search.step()
            step_end = time.time()
            Log.out('STEP', {
                'i': i,
                'done': done,
                'proved': proved,
                'time': "{:.2f}".format(step_end - step_start),
            })
            if done:
                break
            if (step_end - step_start) > 20:
                    # self._config.get('prooftrace_search_step_timeout'):
                break

        if proved:
            rollout = Rollout(name, [ptra], [])
        else:
            rollout = Rollout(name, [], [ptra])

        demo_length = ptra.action_len()
        demo_delta = ptra.action_len() - ground.action_len()

        Log.out("ROLLOUT END", {
            'name': name,
            'proved': proved,
            'demo_length': demo_length,
            'demo_delta': demo_delta
        })

        if proved:
            Log.out("PTRA", {
                'name': name,
                'summary': ptra.summary(),
            })

        if demo_length > 0:
            info = {
                'rll_cnt': 1,
                'pos_cnt': 1 if proved else 0,
                'neg_cnt': 0 if proved else 1,
            }
            if proved:
                info['demo_len'] = demo_length
                info['demo_dlt'] = demo_delta

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
            config.get('prooftrace_lm_iota_sync_dir'),
            'rollout',
        )

    def update(
            self,
    ) -> None:
        update = self._config.update()

        if self._tb_writer is not None:
            for k in update:
                if k in []:
                    self._tb_writer.add_scalar(
                        "prooftrace_lm_rollout_run/{}".format(k),
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
        demo_dlt_meter = Meter()

        for info in infos:
            rll_cnt_meter.update(info['rll_cnt'])
            pos_cnt_meter.update(info['pos_cnt'])
            neg_cnt_meter.update(info['neg_cnt'])
            if 'demo_len' in info:
                demo_len_meter.update(info['demo_len'])
            if 'demo_dlt' in info:
                demo_dlt_meter.update(info['demo_dlt'])

        Log.out("PROOFTRACE LM ROLLOUT CTL RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'rll_cnt': "{:.4f}".format(rll_cnt_meter.sum or 0.0),
            'pos_cnt': "{:.4f}".format(pos_cnt_meter.avg or 0.0),
            'neg_cnt': "{:.4f}".format(neg_cnt_meter.avg or 0.0),
            'demo_len': "{:.4f}".format(demo_len_meter.avg or 0.0),
            'demo_dlt': "{:.4f}".format(demo_dlt_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            if rll_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_rollout/rll_cnt",
                    rll_cnt_meter.sum, self._epoch,
                )
            if pos_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_rollout/pos_cnt",
                    pos_cnt_meter.avg, self._epoch,
                )
            if neg_cnt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_rollout/neg_cnt",
                    neg_cnt_meter.avg, self._epoch,
                )
            if demo_len_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_rollout/demo_len",
                    demo_len_meter.avg, self._epoch,
                )
            if demo_dlt_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_rollout/demo_dlt",
                    demo_dlt_meter.avg, self._epoch,
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
            'prooftrace_lm_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_rollout_dir',
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
            'prooftrace_lm_iota_sync_dir',
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
