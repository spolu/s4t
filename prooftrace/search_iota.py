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

from prooftrace.prooftrace import PREPARE_TOKENS, Action, lm_collate

from generic.iota import IOTAAck, IOTASyn

from prooftrace.seach_base import SearchModel

from tensorboardX import SummaryWriter

from torch.utils.data import Dataset

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


class ProofTraceRLLDataset(Dataset):
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
        float,
    ]:
        rdir = self._rdirs[idx]

        rfiles = sorted([
            os.path.join(rdir, f)
            for f in os.listdir(rdir) if re.search(".rollout$", f)
        ], reverse=True)

        with gzip.open(rfiles[0], 'rb') as f:
            rollout = pickle.load(f)

        ptra, outcome = rollout.random()

        index = random.randrange(ptra.prepare_len(), ptra.len())

        assert idx <= self._sequence_length

        truth = ptra.actions()[index]
        actions = ptra.actions()[:index]
        arguments = ptra.arguments()[:index]

        value = 0.0
        if outcome:
            value = 1.0

        actions.append(Action.from_action('EXTRACT', None, None))

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < self._sequence_length:
            actions.append(empty)
        while len(arguments) < self._sequence_length:
            arguments.append(empty)

        return (index, actions, arguments, truth, value)


class ACK:
    def __init__(
            self,
            config: Config,
            train_dataset: ProofTraceRLLDataset,
    ):
        self._config = config

        self._action_coeff = config.get('prooftrace_search_action_coeff')
        self._value_coeff = config.get('prooftrace_search_value_coeff')

        self._device = torch.device(config.get('device'))
        self._type = config.get('prooftrace_search_model_type')

        self._model = SearchModel(config)

        self._ack = IOTAAck(
            config.get('prooftrace_search_iota_sync_dir'),
            self._model.modules(),
        )

        self._nll_loss = nn.NLLLoss()
        self._mse_loss = nn.MSELoss()

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('prooftrace_search_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('ACK initialization', {
            "batch_size": self._config.get('prooftrace_search_batch_size'),
        })

        self._train_batch = 0

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

        coeff = self._config.get('prooftrace_search_action_coeff')
        if coeff != self._action_coeff:
            self._action_coeff = coeff
            Log.out("Updated", {
                "prooftrace_search_action_coeff": coeff,
            })
        coeff = self._config.get('prooftrace_search_value_coeff')
        if coeff != self._value_coeff:
            self._value_coeff = coeff
            Log.out("Updated", {
                "prooftrace_search_value_coeff": coeff,
            })

    def run_once(
            self,
            epoch,
    ):
        for it, (idx, act, arg, trh, val) in enumerate(self._train_loader):
            info = self._ack.fetch(self._device)
            if info is not None:
                self.update(info['config'])

                for m in self._modules:
                    self._modules[m].train()

            assert len(trh) == len(val)

            act_loss = torch.tensor(0.0).to(self._device)
            lft_loss = torch.tensor(0.0).to(self._device)
            rgt_loss = torch.tensor(0.0).to(self._device)
            val_loss = torch.tensor(0.0).to(self._device)

            if self._type == 'value' or self._policy == 'both':
                prd_actions, prd_lefts, prd_rights, prd_values = \
                    self._model.infer(idx, act, arg)

                values = torch.tensor(val).unsqueeze(1).to(self._device)

                val_loss = self._mse_loss(prd_values, values)

            if self._type == 'policy' or self._policy == 'both':
                prd_actions, prd_lefts, prd_rights = \
                    self._model.infer_actions(idx, act, arg)

                actions = torch.tensor([
                    trh[i].value - len(PREPARE_TOKENS) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                lefts = torch.tensor([
                    arg[i].index(trh[i].left) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                rights = torch.tensor([
                    arg[i].index(trh[i].right) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)

                act_cnt = 0

                for i in range(len(val)):
                    if val[i] > 0:
                        act_loss += self._nll_loss(prd_actions, actions)
                        lft_loss += self._nll_loss(prd_lefts, lefts)
                        rgt_loss += self._nll_loss(prd_rights, rights)
                        act_cnt += 1

                if act_cnt > 0:
                    act_loss /= act_cnt
                    lft_loss /= act_cnt
                    rgt_loss /= act_cnt

            # Backward pass.
            for m in self._modules:
                self._modules[m].zero_grad()

            (self._action_coeff * act_loss + lft_loss + rgt_loss +
             self._value_coeff * val_loss).backward()

            self._ack.push({
                'act_loss': act_loss.item(),
                'lft_loss': lft_loss.item(),
                'rgt_loss': rgt_loss.item(),
                'val_loss': val_loss.item(),
            }, None)

            Log.out("PROOFTRACE BEAM ACK RUN", {
                'epoch': epoch,
                'train_batch': self._train_batch,
                'act_loss_avg': "{:.4f}".format(act_loss.item()),
                'lft_loss_avg': "{:.4f}".format(lft_loss.item()),
                'rgt_loss_avg': "{:.4f}".format(rgt_loss.item()),
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

        self._learning_rate = config.get('prooftrace_search_learning_rate')
        self._min_update_count = \
            config.get('prooftrace_search_iota_min_update_count')

        self._device = torch.device(config.get('device'))
        self._type = config.get('prooftrace_search_model_type')

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._epoch = 0

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._model = SearchModel(config)

        Log.out(
            "SYN Initializing", {
                'parameters_count_pE':
                self._model.modules()['pE'].parameters_count(),
                'parameters_count_pT':
                self._model.modules()['pT'].parameters_count(),
                'parameters_count_pH':
                self._model.modules()['pH'].parameters_count(),
                'parameters_count_vE':
                self._model.modules()['vE'].parameters_count(),
                'parameters_count_vT':
                self._model.modules()['vT'].parameters_count(),
                'parameters_count_vH':
                self._model.modules()['vH'].parameters_count(),
            },
        )

        self._syn = IOTASyn(
            config.get('prooftrace_search_iota_sync_dir'),
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
                    'type': self._type,
                })

            self._model.load()

            if training and os.path.isfile(self._load_dir + "/optimizer.pt"):
                self._value_optimizer.load_state_dict(
                    torch.load(
                        self._load_dir + "/value_optimizer.pt",
                        map_location=self._device,
                    ),
                )
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
                    'type': self._type,
                })

            self._model.save()

            torch.save(
                self._value_optimizer.state_dict(),
                self._save_dir + "/value_optimizer.pt",
            )
            torch.save(
                self._policy_optimizer.state_dict(),
                self._save_dir + "/policy_optimizer.pt",
            )

    def update(
            self,
    ) -> None:
        update = self._config.update()
        if update:
            if 'prooftrace_search_learning_rate' in update:
                lr = self._config.get('prooftrace_search_learning_rate')
                if lr != self._learning_rate:
                    self._learning_rate = lr
                    for group in self._optimizer.param_groups:
                        group['lr'] = lr
                    Log.out("Updated", {
                        "prooftrace_search_learning_rate": lr,
                    })
            if 'prooftrace_search_iota_min_update_count' in update:
                cnt = \
                    self._config.get('prooftrace_search_iota_min_update_count')
                if cnt != self._min_update_count:
                    self._min_update_count = cnt
                    Log.out("Updated", {
                        "prooftrace_search_iota_min_update_count": cnt,
                    })

            if self._tb_writer is not None:
                for k in update:
                    if k in [
                            'prooftrace_search_learning_rate',
                            'prooftrace_search_iota_min_update_count',
                            'prooftrace_search_action_coeff',
                            'prooftrace_search_value_coeff',
                    ]:
                        self._tb_writer.add_scalar(
                            "prooftrace_search_train_run/{}".format(k),
                            update[k], self._epoch,
                        )

    def run_once(
            self,
    ):
        for m in self._modules:
            self._modules[m].train()

        run_start = time.time()

        self._optimizer.zero_grad()
        infos = self._syn.reduce(self._device, self._min_update_count)

        if len(infos) == 0:
            time.sleep(1)
            return

        if self._type == 'policy' or self._policy == 'both':
            self._policy_optimizer.step()
        if self._type == 'value' or self._policy == 'both':
            self._value_optimizer.step()

        self._syn.broadcast({'config': self._config})

        act_loss_meter = Meter()
        lft_loss_meter = Meter()
        rgt_loss_meter = Meter()
        val_loss_meter = Meter()

        for info in infos:
            act_loss_meter.update(info['act_loss'])
            lft_loss_meter.update(info['lft_loss'])
            rgt_loss_meter.update(info['rgt_loss'])
            val_loss_meter.update(info['val_loss'])

        Log.out("PROOFTRACE BEAM SYN RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'act_loss': "{:.4f}".format(act_loss_meter.avg or 0.0),
            'lft_loss': "{:.4f}".format(lft_loss_meter.avg or 0.0),
            'rgt_loss': "{:.4f}".format(rgt_loss_meter.avg or 0.0),
            'val_loss': "{:.4f}".format(val_loss_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            if act_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_train/act_loss",
                    act_loss_meter.avg, self._epoch,
                )
            if lft_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_train/lft_loss",
                    lft_loss_meter.avg, self._epoch,
                )
            if rgt_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_train/rgt_loss",
                    rgt_loss_meter.avg, self._epoch,
                )
            if val_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_search_train/val_loss",
                    val_loss_meter.avg, self._epoch,
                )
            self._tb_writer.add_scalar(
                "prooftrace_search_train/update_count",
                len(infos), self._epoch,
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

    train_dataset = ProofTraceRLLDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_search_rollout_dir')),
            config.get('prooftrace_dataset_size'),
        ),
        config.get('prooftrace_sequence_length'),
    )

    ack = ACK(config, train_dataset)

    epoch = 0
    while True:
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
            'prooftrace_search_iota_sync_dir',
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
