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

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

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

        self._action_coeff = config.get('prooftrace_beam_action_coeff')
        self._value_coeff = config.get('prooftrace_beam_value_coeff')

        self._device = torch.device(config.get('device'))

        self._modules = {
            'E': E(self._config).to(self._device),
            'T': T(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

        self._ack = IOTAAck(
            config.get('prooftrace_beam_iota_sync_dir'),
            self._modules,
        )

        self._nll_loss = nn.NLLLoss()
        self._mse_loss = nn.MSELoss()

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('prooftrace_beam_batch_size'),
            shuffle=True,
            collate_fn=lm_collate,
        )

        Log.out('ACK initialization', {
            "batch_size": self._config.get('prooftrace_beam_batch_size'),
        })

        self._train_batch = 0

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

        coeff = self._config.get('prooftrace_beam_action_coeff')
        if coeff != self._action_coeff:
            self._action_coeff = coeff
            Log.out("Updated", {
                "prooftrace_beam_action_coeff": coeff,
            })
        coeff = self._config.get('prooftrace_beam_value_coeff')
        if coeff != self._value_coeff:
            self._value_coeff = coeff
            Log.out("Updated", {
                "prooftrace_beam_value_coeff": coeff,
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

            action_embeds = self._modules['E'](act)
            argument_embeds = self._modules['E'](arg)

            hiddens = self._modules['T'](action_embeds, argument_embeds)
            heads = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            prd_actions, prd_lefts, prd_rights = \
                self._modules['PH'](heads, hiddens, targets)
            prd_values = self._modules['VH'](heads, targets)

            actions = torch.tensor([
                trh[i].value - len(PREPARE_TOKENS) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            lefts = torch.tensor([
                arg[i].index(trh[i].left) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            rights = torch.tensor([
                arg[i].index(trh[i].right) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            values = torch.tensor(val).unsqueeze(1).to(self._device)

            assert len(trh) == len(val)

            act_loss = torch.tensor(0.0).to(self.device)
            lft_loss = torch.tensor(0.0).to(self.device)
            rgt_loss = torch.tensor(0.0).to(self.device)

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

            val_loss = self._mse_loss(prd_values, values)

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

        self._learning_rate = config.get('prooftrace_beam_learning_rate')
        self._min_update_count = \
            config.get('prooftrace_beam_iota_min_update_count')
        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._epoch = 0

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._modules = {
            'E': E(self._config).to(self._device),
            'T': T(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

        Log.out(
            "SYN Initializing", {
                'parameter_count_E': self._modules['E'].parameters_count(),
                'parameter_count_T': self._modules['T'].parameters_count(),
                'parameter_count_PH': self._modules['PH'].parameters_count(),
                'parameter_count_VH': self._modules['VH'].parameters_count(),
            },
        )

        self._syn = IOTASyn(
            config.get('prooftrace_beam_iota_sync_dir'),
            self._modules,
        )

        self._optimizer = optim.Adam(
            [
                {'params': self._modules['E'].parameters()},
                {'params': self._modules['T'].parameters()},
                {'params': self._modules['PH'].parameters()},
                {'params': self._modules['VH'].parameters()},
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
                "Loading prooftrace BEAM", {
                    'load_dir': self._load_dir,
                })
            if os.path.isfile(self._load_dir + "/model_E.pt"):
                self._modules['E'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_E.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_T.pt"):
                self._modules['T'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_T.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_PH.pt"):
                self._modules['PH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_PH.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_VH.pt"):
                self._modules['VH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_VH.pt",
                        map_location=self._device,
                    ),
                )

            if training and os.path.isfile(self._load_dir + "/optimizer.pt"):
                self._optimizer.load_state_dict(
                    torch.load(
                        self._load_dir + "/optimizer.pt",
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

            torch.save(
                self._modules['E'].state_dict(),
                self._save_dir + "/model_E.pt",
            )
            torch.save(
                self._modules['T'].state_dict(),
                self._save_dir + "/model_T.pt",
            )
            torch.save(
                self._modules['PH'].state_dict(),
                self._save_dir + "/model_PH.pt",
            )
            torch.save(
                self._modules['VH'].state_dict(),
                self._save_dir + "/model_VH.pt",
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer.pt",
            )

    def update(
            self,
    ) -> None:
        update = self._config.update()
        if update:
            if 'prooftrace_beam_learning_rate' in update:
                lr = self._config.get('prooftrace_beam_learning_rate')
                if lr != self._learning_rate:
                    self._learning_rate = lr
                    for group in self._optimizer.param_groups:
                        group['lr'] = lr
                    Log.out("Updated", {
                        "prooftrace_beam_learning_rate": lr,
                    })
            if 'prooftrace_beam_iota_min_update_count' in update:
                cnt = self._config.get('prooftrace_beam_iota_min_update_count')
                if cnt != self._min_update_count:
                    self._min_update_count = cnt
                    Log.out("Updated", {
                        "prooftrace_beam_iota_min_update_count": cnt,
                    })

            if self._tb_writer is not None:
                for k in update:
                    if k in [
                            'prooftrace_beam_learning_rate',
                            'prooftrace_beam_iota_min_update_count',
                            'prooftrace_beam_action_coeff',
                            'prooftrace_beam_value_coeff',
                    ]:
                        self._tb_writer.add_scalar(
                            "prooftrace_beam_train_run/{}".format(k),
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

        self._optimizer.step()
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
                    "prooftrace_beam_train/act_loss",
                    act_loss_meter.avg, self._epoch,
                )
            if lft_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_train/lft_loss",
                    lft_loss_meter.avg, self._epoch,
                )
            if rgt_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_train/rgt_loss",
                    rgt_loss_meter.avg, self._epoch,
                )
            if val_loss_meter.avg is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_beam_train/val_loss",
                    val_loss_meter.avg, self._epoch,
                )
            self._tb_writer.add_scalar(
                "prooftrace_beam_train/update_count",
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
            'prooftrace_beam_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )
    if args.rollout_dir is not None:
        config.override(
            'prooftrace_beam_rollout_dir',
            os.path.expanduser(args.rollout_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    train_dataset = ProofTraceRLLDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_beam_rollout_dir')),
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
            'prooftrace_beam_iota_sync_dir',
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
