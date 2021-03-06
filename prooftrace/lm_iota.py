import argparse
import gzip
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from generic.iota import IOTAAck, IOTASyn

from prooftrace.dataset import ProofTraceLMDataset, lm_collate, trh_extract
from prooftrace.models.model import LModel

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

        self._action_coeff = config.get('prooftrace_lm_action_coeff')
        self._grad_norm_max = config.get('prooftrace_lm_grad_norm_max')

        self._device = torch.device(config.get('device'))

        self._sequence_length = config.get('prooftrace_sequence_length')

        self._model = LModel(config)
        self._ack = IOTAAck(
            config.get('prooftrace_lm_iota_sync_dir'),
            self._model.modules(),
        )

        self._nll_loss = nn.NLLLoss()

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('prooftrace_lm_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('ACK initialization', {
            "batch_size": self._config.get('prooftrace_lm_batch_size'),
        })

        self._train_batch = 0

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

        coeff = self._config.get('prooftrace_lm_action_coeff')
        if coeff != self._action_coeff:
            self._action_coeff = coeff
            Log.out("Updated", {
                "prooftrace_lm_action_coeff": coeff,
            })

    def run_once(
            self,
            epoch,
    ):
        for it, (act, arg, trh) in enumerate(self._train_loader):
            info = self._ack.fetch(self._device)
            if info is not None:
                self.update(info['config'])
            self._model.train()

            trh_actions, trh_lefts, trh_rights = trh_extract(trh, arg)

            # Because we can't run a pointer network on the full length
            # (memory), we extract indices to focus loss on.
            idx = random.sample(range(self._sequence_length), 64)

            actions = torch.index_select(
                torch.tensor(trh_actions, dtype=torch.int64),
                1,
                torch.tensor(idx, dtype=torch.int64),
            ).to(self._device)
            lefts = torch.index_select(
                torch.tensor(trh_lefts, dtype=torch.int64),
                1,
                torch.tensor(idx, dtype=torch.int64),
            ).to(self._device)
            rights = torch.index_select(
                torch.tensor(trh_rights, dtype=torch.int64),
                1,
                torch.tensor(idx, dtype=torch.int64),
            ).to(self._device)

            prd_actions, prd_lefts, prd_rights = \
                self._model.infer(idx, act, arg)

            act_loss = self._nll_loss(
                prd_actions.view(-1, prd_actions.size(-1)), actions.view(-1),
            )
            lft_loss = self._nll_loss(
                prd_lefts.view(-1, prd_lefts.size(-1)), lefts.view(-1),
            )
            rgt_loss = self._nll_loss(
                prd_rights.view(-1, prd_rights.size(-1)), rights.view(-1),
            )

            # Backward pass.
            for m in self._model.modules():
                self._model.modules()[m].zero_grad()

            (self._action_coeff * act_loss + lft_loss + rgt_loss).backward()

            if self._grad_norm_max > 0.0:
                for m in self._model.modules():
                    torch.nn.utils.clip_grad_norm_(
                        self._model.modules()[m].parameters(),
                        self._grad_norm_max,
                    )

            info = {
                'act_loss': act_loss.item(),
                'lft_loss': lft_loss.item(),
                'rgt_loss': rgt_loss.item(),
            }

            self._ack.push(info, None)

            Log.out("PROOFTRACE LM ACK RUN", {
                'epoch': epoch,
                'train_batch': self._train_batch,
                'act_loss_avg': "{:.4f}".format(act_loss.item()),
                'lft_loss_avg': "{:.4f}".format(lft_loss.item()),
                'rgt_loss_avg': "{:.4f}".format(rgt_loss.item()),
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

        self._sequence_length = config.get('prooftrace_sequence_length')

        self._model = LModel(config)
        self._ack = IOTAAck(
            config.get('prooftrace_lm_iota_sync_dir'),
            self._model.modules(),
        )

        self._nll_loss = nn.NLLLoss()

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._config.get('prooftrace_lm_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('TST initialization', {
            "batch_size": self._config.get('prooftrace_lm_batch_size'),
        })

        self._train_batch = 0

    def run_once(
            self,
            epoch,
    ):
        act_loss_meter = Meter()
        lft_loss_meter = Meter()
        rgt_loss_meter = Meter()

        with torch.no_grad():
            for it, (act, arg, trh) in enumerate(self._test_loader):
                self._ack.fetch(self._device, blocking=False)
                self._model.eval()

                trh_actions, trh_lefts, trh_rights = trh_extract(trh, arg)

                # Because we can't run a pointer network on the full length
                # (memory), we extract indices to focus loss on.
                idx = random.sample(range(self._sequence_length), 64)

                actions = torch.index_select(
                    torch.tensor(trh_actions, dtype=torch.int64),
                    1,
                    torch.tensor(idx, dtype=torch.int64),
                ).to(self._device)
                lefts = torch.index_select(
                    torch.tensor(trh_lefts, dtype=torch.int64),
                    1,
                    torch.tensor(idx, dtype=torch.int64),
                ).to(self._device)
                rights = torch.index_select(
                    torch.tensor(trh_rights, dtype=torch.int64),
                    1,
                    torch.tensor(idx, dtype=torch.int64),
                ).to(self._device)

                prd_actions, prd_lefts, prd_rights = \
                    self._model.infer(idx, act, arg)

                act_loss = self._nll_loss(
                    prd_actions.view(-1, prd_actions.size(-1)),
                    actions.view(-1),
                )
                lft_loss = self._nll_loss(
                    prd_lefts.view(-1, prd_lefts.size(-1)), lefts.view(-1),
                )
                rgt_loss = self._nll_loss(
                    prd_rights.view(-1, prd_rights.size(-1)), rights.view(-1),
                )

                act_loss_meter.update(act_loss.item())
                lft_loss_meter.update(lft_loss.item())
                rgt_loss_meter.update(rgt_loss.item())

                info = {
                    'test_act_loss': act_loss_meter.avg,
                    'test_lft_loss': lft_loss_meter.avg,
                    'test_rgt_loss': rgt_loss_meter.avg,
                }

                self._ack.push(info, None, True)

                Log.out("PROOFTRACE LM TST RUN", {
                    'epoch': epoch,
                    'act_loss_avg': "{:.4f}".format(act_loss.item()),
                    'lft_loss_avg': "{:.4f}".format(lft_loss.item()),
                    'rgt_loss_avg': "{:.4f}".format(rgt_loss.item()),
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

        self._learning_rate = config.get('prooftrace_lm_learning_rate')
        self._min_update_count = \
            config.get('prooftrace_lm_iota_min_update_count')

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

        self._model = LModel(config)

        Log.out(
            "SYN Initializing", {
                'parameters_count_pE':
                self._model.modules()['pE'].parameters_count(),
                'parameters_count_pT':
                self._model.modules()['pT'].parameters_count(),
                'parameters_count_pH':
                self._model.modules()['pH'].parameters_count(),
            },
        )

        self._syn = IOTASyn(
            config.get('prooftrace_lm_iota_sync_dir'),
            self._model.modules(),
        )

        self._policy_optimizer = optim.Adam(
            [
                {'params': self._model.modules()['pE'].parameters()},
                {'params': self._model.modules()['pT'].parameters()},
                {'params': self._model.modules()['pH'].parameters()},
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
                "Loading prooftrace models", {
                    'load_dir': self._load_dir,
                })

            self._model.load()

            if training and os.path.isfile(self._load_dir + "/optimizer.pt"):
                self._policy_optimizer.load_state_dict(
                    torch.load(
                        self._load_dir + "/policy_optimizer.pt",
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
                "Saving prooftrace models", {
                    'save_dir': self._save_dir,
                })

            self._model.save()

            torch.save(
                self._policy_optimizer.state_dict(),
                self._save_dir + "/policy_optimizer.pt",
            )

    def update(
            self,
    ) -> None:
        update = self._config.update()
        if update:
            if 'prooftrace_lm_learning_rate' in update:
                lr = self._config.get('prooftrace_lm_learning_rate')
                if lr != self._learning_rate:
                    self._learning_rate = lr
                    for group in self._policy_optimizer.param_groups:
                        group['lr'] = lr
                    Log.out("Updated", {
                        "prooftrace_lm_learning_rate": lr,
                    })
            if 'prooftrace_lm_iota_min_update_count' in update:
                cnt = \
                    self._config.get('prooftrace_lm_iota_min_update_count')
                if cnt != self._min_update_count:
                    self._min_update_count = cnt
                    Log.out("Updated", {
                        "prooftrace_lm_iota_min_update_count": cnt,
                    })

            if self._tb_writer is not None:
                for k in update:
                    if k in [
                            'prooftrace_lm_learning_rate',
                            'prooftrace_lm_iota_min_update_count',
                            'prooftrace_lm_action_coeff',
                    ]:
                        self._tb_writer.add_scalar(
                            "prooftrace_lm_train_run/{}".format(k),
                            update[k], self._epoch,
                        )

    def run_once(
            self,
    ):
        for m in self._model.modules():
            self._model.modules()[m].train()

        run_start = time.time()

        self._policy_optimizer.zero_grad()

        infos = self._syn.reduce(self._device, self._min_update_count)

        if len(infos) == 0:
            time.sleep(1)
            return

        self._policy_optimizer.step()

        self._syn.broadcast({'config': self._config})

        if self._last_update is not None:
            update_delta = time.time() - self._last_update
        else:
            update_delta = 0.0
        self._last_update = time.time()

        act_loss_meter = Meter()
        lft_loss_meter = Meter()
        rgt_loss_meter = Meter()
        test_act_loss_meter = Meter()
        test_lft_loss_meter = Meter()
        test_rgt_loss_meter = Meter()

        for info in infos:
            if 'act_loss' in info:
                act_loss_meter.update(info['act_loss'])
            if 'lft_loss' in info:
                lft_loss_meter.update(info['lft_loss'])
            if 'rgt_loss' in info:
                rgt_loss_meter.update(info['rgt_loss'])
            if 'test_act_loss' in info:
                test_act_loss_meter.update(info['test_act_loss'])
            if 'test_lft_loss' in info:
                test_lft_loss_meter.update(info['test_lft_loss'])
            if 'test_rgt_loss' in info:
                test_rgt_loss_meter.update(info['test_rgt_loss'])

        Log.out("PROOFTRACE SYN RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'update_delta': "{:.2f}".format(update_delta),
            'act_loss': "{:.4f}".format(act_loss_meter.avg or 0.0),
            'lft_loss': "{:.4f}".format(lft_loss_meter.avg or 0.0),
            'rgt_loss': "{:.4f}".format(rgt_loss_meter.avg or 0.0),
            'test_act_loss': "{:.4f}".format(test_act_loss_meter.avg or 0.0),
            'test_lft_loss': "{:.4f}".format(test_lft_loss_meter.avg or 0.0),
            'test_rgt_loss': "{:.4f}".format(test_rgt_loss_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "prooftrace_lm_train/update_delta",
                update_delta, self._epoch,
            )
            self._tb_writer.add_scalar(
                "prooftrace_lm_train/update_count",
                len(infos), self._epoch,
            )
            if act_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_train/act_loss",
                    act_loss_meter.avg, self._epoch,
                )
            if lft_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_train/lft_loss",
                    lft_loss_meter.avg, self._epoch,
                )
            if rgt_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_train/rgt_loss",
                    rgt_loss_meter.avg, self._epoch,
                )

            if test_act_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_test/act_loss",
                    test_act_loss_meter.avg, self._epoch,
                )
            if test_lft_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_test/lft_loss",
                    test_lft_loss_meter.avg, self._epoch,
                )
            if test_rgt_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_test/rgt_loss",
                    test_rgt_loss_meter.avg, self._epoch,
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
            'prooftrace_lm_iota_sync_dir',
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
