import argparse
import gzip
import os
import pickle
import random
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
import typing

from prooftrace.prooftrace import PREPARE_TOKENS, Action

from generic.iota import IOTAAck, IOTASyn

from prooftrace.models.model import Model

from tensorboardX import SummaryWriter

from torch.utils.data import Dataset

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


def lm_collate(
        batch
) -> typing.Tuple[
    typing.List[int],
    typing.List[typing.List[Action]],
    typing.List[typing.List[Action]],
    typing.List[Action],
]:
    indices = []
    actions = []
    arguments = []
    truths = []

    for (idx, act, arg, trh) in batch:
        indices.append(idx)
        actions.append(act)
        arguments.append(arg)
        truths.append(trh)

    return (indices, actions, arguments, truths)


class ProofTraceLMDataset(Dataset):
    def __init__(
            self,
            rollout_dir: str,
            sequence_length: int,
    ) -> None:
        self._sequence_length = sequence_length

        self._rdirs = []

        assert os.path.isdir(rollout_dir)
        self._rdirs = [
            os.path.join(rollout_dir, f)
            for f in os.listdir(rollout_dir)
            if os.path.isdir(os.path.join(rollout_dir, f))
        ]

        Log.out(
            "Loaded extracted ProofTraces Rollout Dataset", {
                'cases': len(self._rdirs),
            })

    def __len__(
            self,
    ) -> int:
        return len(self._rdirs)

    def __getitem__(
            self,
            idx: int,
    ) -> typing.Tuple[
        int,
        typing.List[Action],
        typing.List[Action],
        Action,
    ]:
        rdir = self._rdirs[idx]

        rfiles = sorted([
            os.path.join(rdir, f)
            for f in os.listdir(rdir) if re.search(".rollout$", f)
        ], reverse=True)

        with gzip.open(rfiles[0], 'rb') as f:
            rollout = pickle.load(f)

        ptra = rollout.positive()

        index = random.randrange(ptra.prepare_len(), ptra.len())

        assert index <= self._sequence_length

        truth = ptra.actions()[index]
        actions = ptra.actions()[:index]
        arguments = ptra.arguments()[:index]

        actions.append(Action.from_action('EXTRACT', None, None))

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < self._sequence_length:
            actions.append(empty)
        while len(arguments) < self._sequence_length:
            arguments.append(empty)

        return (index, actions, arguments, truth)


class ACK:
    def __init__(
            self,
            config: Config,
            train_dataset: ProofTraceLMDataset,
            test_dataset: ProofTraceLMDataset,
    ):
        self._config = config

        self._action_coeff = config.get('prooftrace_lm_action_coeff')

        self._device = torch.device(config.get('device'))

        self._model = Model(config)
        self._ack = IOTAAck(
            config.get('prooftrace_lm_iota_sync_dir'),
            self._model.modules(),
        )

        self._nll_loss = nn.NLLLoss()
        self._mse_loss = nn.MSELoss()

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('prooftrace_lm_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )
        self._test_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('prooftrace_lm_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('ACK initialization', {
            "batch_size": self._config.get('prooftrace_lm_batch_size'),
        })

        self._train_batch = 0
        self._test_info = None

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
        for it, (idx, act, arg, trh) in enumerate(self._train_loader):
            info = self._ack.fetch(self._device)
            if info is not None:
                self.update(info['config'])
            self._model.train()

            prd_actions, prd_lefts, prd_rights = \
                self._model.infer(idx, act, arg)

            actions = torch.tensor([
                trh[i].value - len(PREPARE_TOKENS) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            lefts = torch.tensor([
                arg[i].index(trh[i].left) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            rights = torch.tensor([
                arg[i].index(trh[i].right) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)

            act_loss = self._nll_loss(prd_actions, actions)
            lft_loss = self._nll_loss(prd_lefts, lefts)
            rgt_loss = self._nll_loss(prd_rights, rights)

            # Backward pass.
            for m in self._model.modules():
                self._model.modules()[m].zero_grad()

            (self._action_coeff * act_loss + lft_loss + rgt_loss).backward()

            info = {
                'act_loss': act_loss.item(),
                'lft_loss': lft_loss.item(),
                'rgt_loss': rgt_loss.item(),
            }
            if self._test_info is not None:
                info.update(self._test_info)
                self._test_info = None

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

    def test(
            self,
            epoch,
    ):
        self._model.eval()

        act_loss_meter = Meter()
        lft_loss_meter = Meter()
        rgt_loss_meter = Meter()

        with torch.no_grad():
            for it, (idx, act, arg, trh) in enumerate(self._train_loader):
                prd_actions, prd_lefts, prd_rights = \
                    self._model.infer(idx, act, arg)

                actions = torch.tensor([
                    trh[i].value - len(PREPARE_TOKENS) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                lefts = torch.tensor([
                    arg[i].index(trh[i].left) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                rights = torch.tensor([
                    arg[i].index(trh[i].right) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)

                act_loss = self._nll_loss(prd_actions, actions)
                lft_loss = self._nll_loss(prd_lefts, lefts)
                rgt_loss = self._nll_loss(prd_rights, rights)

                act_loss_meter.update(act_loss.item())
                lft_loss_meter.update(lft_loss.item())
                rgt_loss_meter.update(rgt_loss.item())

        Log.out("PROOFTRACE LM ACK TEST", {
            'epoch': epoch,
            'act_loss_avg': "{:.4f}".format(act_loss.item()),
            'lft_loss_avg': "{:.4f}".format(lft_loss.item()),
            'rgt_loss_avg': "{:.4f}".format(rgt_loss.item()),
        })

        self._test_info = {
            'test_act_loss': act_loss_meter.avg,
            'test_lft_loss': lft_loss_meter.avg,
            'test_rgt_loss': rgt_loss_meter.avg,
        }


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

        self._model = Model(config)

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
                "Loading prooftrace search models", {
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
                "Saving prooftrace search models", {
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
            act_loss_meter.update(info['act_loss'])
            lft_loss_meter.update(info['lft_loss'])
            rgt_loss_meter.update(info['rgt_loss'])
            if 'test_act_loss' in info:
                test_act_loss_meter.update(info['test_act_loss'])
            if 'test_lft_loss' in info:
                test_lft_loss_meter.update(info['test_lft_loss'])
            if 'test_rgt_loss' in info:
                test_rgt_loss_meter.update(info['test_rgt_loss'])

        Log.out("PROOFTRACE BEAM SYN RUN", {
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
            'prooftrace_lm_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    train_dataset = ProofTraceLMDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_lm_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'train_rollouts',
        ),
        config.get('prooftrace_sequence_length'),
    )
    test_dataset = ProofTraceLMDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_lm_rollout_dir')),
            config.get('prooftrace_dataset_size'),
            'test_rollouts',
        ),
        config.get('prooftrace_sequence_length'),
    )

    ack = ACK(config, train_dataset, test_dataset)

    epoch = 0
    while True:
        if epoch % 2 == 0 and epoch > 0:
            ack.test(epoch)
        ack.run_once(epoch)
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
