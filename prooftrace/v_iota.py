import argparse
import gzip
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

from generic.iota import IOTAAck, IOTASyn

from prooftrace.dataset import ProofTraceLMDataset, lm_collate
from prooftrace.models.model import VModel

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


class ACK:
    def __init__(
            self,
            config: Config,
            train_dataset: ProofTraceLMDataset,
    ):
        self._config = config

        self._grad_norm_max = config.get('prooftrace_v_grad_norm_max')

        self._device = torch.device(config.get('device'))

        self._model = VModel(config)
        self._ack = IOTAAck(
            config.get('prooftrace_v_iota_sync_dir'),
            self._model.modules(),
        )

        self._mse_loss = nn.MSELoss()

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('prooftrace_v_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('ACK initialization', {
            "batch_size": self._config.get('prooftrace_v_batch_size'),
        })

        self._train_batch = 0

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

    def run_once(
            self,
            epoch,
    ):
        for it, (idx, act, arg, trh, val) in enumerate(self._train_loader):
            info = self._ack.fetch(self._device)
            if info is not None:
                self.update(info['config'])
            self._model.train()

            prd_values = self._model.infer(idx, act, arg)
            values = torch.tensor(val, dtype=torch.float).to(self._device)

            val_loss = self._mse_loss(prd_values, values)

            # Backward pass.
            for m in self._model.modules():
                self._model.modules()[m].zero_grad()

            val_loss.backward()

            if self._grad_norm_max > 0.0:
                for m in self._model.modules():
                    torch.nn.utils.clip_grad_norm_(
                        self._model.modules()[m].parameters(),
                        self._grad_norm_max,
                    )

            info = {
                'val_loss': val_loss.item(),
            }

            self._ack.push(info, None)

            Log.out("PROOFTRACE V ACK RUN", {
                'epoch': epoch,
                'train_batch': self._train_batch,
                'val_loss_avg': "{:.4f}".format(val_loss.item()),
            })

            self._train_batch += 1

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })


class TST:
    def __init__(
            self,
            config: Config,
            test_dataset: ProofTraceLMDataset,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._model = VModel(config)
        self._ack = IOTAAck(
            config.get('prooftrace_v_iota_sync_dir'),
            self._model.modules(),
        )

        self._mse_loss = nn.MSELoss()

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._config.get('prooftrace_v_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('TST initialization', {
            "batch_size": self._config.get('prooftrace_v_batch_size'),
        })

        self._train_batch = 0

    def run_once(
            self,
            epoch,
    ):
        val_loss_meter = Meter()

        with torch.no_grad():
            for it, (idx, act, arg, trh, val) in enumerate(self._test_loader):
                self._ack.fetch(self._device, blocking=False)
                self._model.eval()

                prd_values = self._model.infer(idx, act, arg)
                values = torch.tensor(val, dtype=torch.float).to(self._device)

                val_loss = self._mse_loss(prd_values, values)

                val_loss_meter.update(val_loss.item())

                info = {
                    'test_val_loss': val_loss_meter.avg,
                }

                self._ack.push(info, None, True)

                Log.out("PROOFTRACE V TST RUN", {
                    'epoch': epoch,
                    'val_loss_avg': "{:.4f}".format(val_loss.item()),
                })

                self._train_batch += 1

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })


class SYN:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._learning_rate = config.get('prooftrace_v_learning_rate')
        self._min_update_count = \
            config.get('prooftrace_v_iota_min_update_count')

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._epoch = 0
        self._last_update = None

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._model = VModel(config)

        Log.out(
            "SYN Initializing", {
                'parameters_count_vE':
                self._model.modules()['vE'].parameters_count(),
                'parameters_count_vT':
                self._model.modules()['vT'].parameters_count(),
                'parameters_count_vH':
                self._model.modules()['vH'].parameters_count(),
            },
        )

        self._syn = IOTASyn(
            config.get('prooftrace_v_iota_sync_dir'),
            self._model.modules(),
        )

        self._value_optimizer = optim.Adam(
            [
                {'params': self._model.modules()['vE'].parameters()},
                {'params': self._model.modules()['vT'].parameters()},
                {'params': self._model.modules()['vH'].parameters()},
            ],
            lr=self._learning_rate,
        )

        self._syn.broadcast({'config': self._config})

    def load(
            self,
            training=True,
    ):

        if self._load_dir:
            Log.out(
                "Loading prooftrace search models", {
                    'load_dir': self._load_dir,
                })

            self._model.load()

            if training and os.path.isfile(self._load_dir + "/optimizer.pt"):
                self._value_optimizer.load_state_dict(
                    torch.load(
                        self._load_dir + "/value_optimizer.pt",
                        map_location=self._device,
                    ),
                )

        self._syn.broadcast({'config': self._config})

        return self

    def save(
            self,
    ):
        if self._save_dir:
            Log.out(
                "Saving prooftrace search models", {
                    'save_dir': self._save_dir,
                })

            self._model.save()

            torch.save(
                self._value_optimizer.state_dict(),
                self._save_dir + "/value_optimizer.pt",
            )

    def update(
            self,
    ) -> None:
        update = self._config.update()
        if update:
            if 'prooftrace_v_learning_rate' in update:
                lr = self._config.get('prooftrace_v_learning_rate')
                if lr != self._learning_rate:
                    self._learning_rate = lr
                    for group in self._value_optimizer.param_groups:
                        group['lr'] = lr
                    Log.out("Updated", {
                        "prooftrace_v_learning_rate": lr,
                    })
            if 'prooftrace_v_iota_min_update_count' in update:
                cnt = \
                    self._config.get('prooftrace_v_iota_min_update_count')
                if cnt != self._min_update_count:
                    self._min_update_count = cnt
                    Log.out("Updated", {
                        "prooftrace_v_iota_min_update_count": cnt,
                    })

            if self._tb_writer is not None:
                for k in update:
                    if k in [
                            'prooftrace_v_learning_rate',
                            'prooftrace_v_iota_min_update_count',
                    ]:
                        self._tb_writer.add_scalar(
                            "prooftrace_v_train_run/{}".format(k),
                            update[k], self._epoch,
                        )

    def run_once(
            self,
    ):
        for m in self._model.modules():
            self._model.modules()[m].train()

        run_start = time.time()

        self._value_optimizer.zero_grad()

        infos = self._syn.reduce(self._device, self._min_update_count)

        if len(infos) == 0:
            time.sleep(1)
            return

        self._value_optimizer.step()

        self._syn.broadcast({'config': self._config})

        if self._last_update is not None:
            update_delta = time.time() - self._last_update
        else:
            update_delta = 0.0
        self._last_update = time.time()

        val_loss_meter = Meter()
        test_val_loss_meter = Meter()

        for info in infos:
            if 'val_loss' in info:
                val_loss_meter.update(info['val_loss'])
            if 'test_val_loss' in info:
                test_val_loss_meter.update(info['test_val_loss'])

        Log.out("PROOFTRACE BEAM SYN RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'update_delta': "{:.2f}".format(update_delta),
            'val_loss': "{:.4f}".format(val_loss_meter.avg or 0.0),
            'test_val_loss': "{:.4f}".format(test_val_loss_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "prooftrace_v_train/update_delta",
                update_delta, self._epoch,
            )
            self._tb_writer.add_scalar(
                "prooftrace_v_train/update_count",
                len(infos), self._epoch,
            )
            if val_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_v_train/val_loss",
                    val_loss_meter.avg, self._epoch,
                )
            if test_val_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_v_test/val_loss",
                    test_val_loss_meter.avg, self._epoch,
                )

        self._epoch += 1

        if self._epoch % 100 == 0:
            self.save()


def ack_run():
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
            'prooftrace_v_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    train_dataset = ProofTraceLMDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'train_rollouts',
        ),
        config.get('prooftrace_sequence_length'),
        tokenizer,
        config.get('prooftrace_v_iota_augment'),
        config.get('prooftrace_v_iota_augment_period'),
    )

    ack = ACK(config, train_dataset)

    epoch = 0
    while True:
        ack.run_once(epoch)
        epoch += 1


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
            'prooftrace_v_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    test_dataset = ProofTraceLMDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'test_rollouts',
        ),
        config.get('prooftrace_sequence_length'),
        tokenizer,
        config.get('prooftrace_v_iota_augment'),
        config.get('prooftrace_v_iota_augment_period'),
    )

    tst = TST(config, test_dataset)

    epoch = 0
    while True:
        tst.run_once(epoch)
        epoch += 1


def syn_run():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--save_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--load_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--tensorboard_log_dir',
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
    if args.sync_dir is not None:
        config.override(
            'prooftrace_v_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )

    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )
    if args.save_dir is not None:
        config.override(
            'prooftrace_save_dir',
            os.path.expanduser(args.save_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    syn = SYN(config).load(True)

    while True:
        syn.update()
        syn.run_once()
