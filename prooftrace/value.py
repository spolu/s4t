import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim

from dataset.prooftrace import ProofTraceVDataset, v_collate

from tensorboardX import SummaryWriter

from prooftrace.models.embedder import E
from prooftrace.models.heads import VH
from prooftrace.models.lstm import H

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class LanguageModel:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._accumulation_step_count = \
            config.get('prooftrace_accumulation_step_count')

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model_E = E(self._config).to(self._device)
        self._inner_model_H = H(self._config).to(self._device)
        self._inner_model_VH = VH(self._config).to(self._device)

        Log.out(
            "Initializing prooftrace LanguageModel", {
                'parameter_count_E': self._inner_model_E.parameters_count(),
                'parameter_count_H': self._inner_model_H.parameters_count(),
                'parameter_count_VH': self._inner_model_VH.parameters_count(),
            },
        )

        self._model_E = self._inner_model_E
        self._model_H = self._inner_model_H
        self._model_VH = self._inner_model_VH

        self._loss = nn.NLLLoss()

        self._train_batch = 0

    def init_training(
            self,
            train_dataset,
    ):
        if self._config.get('distributed_training'):
            self._model_E = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_E,
                device_ids=[self._device],
            )
            self._model_H = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_H,
                device_ids=[self._device],
            )
            self._model_VH = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_VH,
                device_ids=[self._device],
            )

        self._optimizer = optim.Adam(
            [
                {'params': self._model_E.parameters()},
                {'params': self._model_H.parameters()},
                {'params': self._model_VH.parameters()},
            ],
            lr=self._config.get('prooftrace_learning_rate'),
        )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                )

        batch_size = self._config.get('prooftrace_batch_size') // \
            self._accumulation_step_count

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self._train_sampler is None),
            sampler=self._train_sampler,
            collate_fn=v_collate,
            num_workers=16,
        )

        Log.out('Training initialization', {
            "accumulation_step_count": self._accumulation_step_count,
            "world_size": self._config.get('distributed_world_size'),
            "batch_size": self._config.get('prooftrace_batch_size'),
            "dataloader_batch_size": batch_size,
            "effective_batch_size": (
                self._config.get('prooftrace_batch_size') *
                self._config.get('distributed_world_size')
            ),
        })

    def init_testing(
            self,
            test_dataset,
    ):
        batch_size = self._config.get('prooftrace_batch_size') // \
            self._accumulation_step_count

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=v_collate,
            num_workers=8,
        )

    def load(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model_H_{}.pt".format(rank)
            ):
                Log.out(
                    "Loading prooftrace", {
                        'load_dir': self._load_dir,
                    })
                self._inner_model_E.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_E_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                self._inner_model_H.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_H_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                self._inner_model_VH.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_VH_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                if training:
                    self._optimizer.load_state_dict(
                        torch.load(
                            self._load_dir +
                            "/optimizer_{}.pt".format(rank),
                            map_location=self._device,
                        ),
                    )

        return self

    def save(
            self,
    ):
        rank = self._config.get('distributed_rank')

        if self._save_dir:
            Log.out(
                "Saving prooftrace models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._inner_model_E.state_dict(),
                self._save_dir + "/model_E_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_H.state_dict(),
                self._save_dir + "/model_H_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_VH.state_dict(),
                self._save_dir + "/model_VH_{}.pt".format(rank),
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer_{}.pt".format(rank),
            )

    def batch_train(
            self,
            epoch,
    ):
        assert self._train_loader is not None

        self._model_E.train()
        self._model_H.train()
        self._model_VH.train()

        val_loss_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(epoch)

        for it, (idx, trc, val) in enumerate(self._train_loader):
            embeds = self._model_E(trc)
            hiddens = self._model_H(embeds)

            head = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                hiddens[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            values = torch.tensor(val).unsqueeze(1).to(self._device)
            prd_values = self._model_VH(head, targets)
            val_loss = F.mse_loss(prd_values, values)

            (val_loss).backward()

            if it % self._accumulation_step_count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()

            val_loss_meter.update(val_loss.item())

            if self._train_batch % 10 == 0:
                Log.out("PROOFTRACE TRAIN", {
                    'train_batch': self._train_batch,
                    'val_loss_avg': "{:.4f}".format(val_loss_meter.avg),
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "prooftrace_value_train/val_loss",
                        val_loss_meter.avg, self._train_batch,
                    )

                val_loss_meter = Meter()

            if self._train_batch % 100 == 0:
                self.save()

            self._train_batch += 1

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })

    def batch_test(
            self,
    ):
        assert self._test_loader is not None

        self._model_E.eval()
        self._model_H.eval()
        self._model_VH.eval()

        val_loss_meter = Meter()

        with torch.no_grad():
            for it, (idx, trc, val) in enumerate(self._test_loader):
                embeds = self._model_E(trc)
                hiddens = self._model_H(embeds)

                head = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    hiddens[i][0].unsqueeze(0) for i in range(len(idx))
                ], dim=0)

                values = torch.tensor(val).to(self._device)
                prd_values = self._model_VH(head, targets)
                val_loss = F.mse_loss(prd_values, values)

                val_loss_meter.update(val_loss.item())

                Log.out("PROOFTRACE TEST", {
                    'batch': it,
                    'val_loss_avg': "{:.4f}".format(val_loss_meter.avg),
                })


def train():
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
        '--distributed_training',
        type=str2bool, help="confg override",
    )
    parser.add_argument(
        '--distributed_world_size',
        type=int, help="config override",
    )
    parser.add_argument(
        '--distributed_rank',
        type=int, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)

    if args.distributed_training is not None:
        config.override('distributed_training', args.distributed_training)
    if args.distributed_rank is not None:
        config.override('distributed_rank', args.distributed_rank)
    if args.distributed_world_size is not None:
        config.override('distributed_world_size', args.distributed_world_size)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
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

    if config.get('distributed_training'):
        distributed.init_process_group(
            backend=config.get('distributed_backend'),
            init_method=config.get('distributed_init_method'),
            rank=config.get('distributed_rank'),
            world_size=config.get('distributed_world_size'),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    train_dataset = ProofTraceVDataset(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        False,
        config.get('prooftrace_sequence_length'),
    )

    lm = LanguageModel(config)

    lm.init_training(train_dataset)
    lm.load(True)

    epoch = 0
    while True:
        lm.batch_train(epoch)
        lm.save()
        epoch += 1


def test():
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
        '--load_dir',
        type=str, help="config override",
    )

    parser.add_argument(
        '--device',
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
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    test_dataset = ProofTraceVDataset(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        True,
        config.get('prooftrace_sequence_length'),
    )

    lm = LanguageModel(config)

    lm.init_testing(test_dataset)
    lm.load(False)

    lm.batch_test()
