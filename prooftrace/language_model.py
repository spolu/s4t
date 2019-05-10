import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim

from prooftrace.prooftrace import \
    ProofTraceLMDataset, lm_collate, PREPARE_TOKENS

from tensorboardX import SummaryWriter

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import H

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
            config.get('prooftrace_lm_accumulation_step_count')
        self._learning_rate = \
            config.get('prooftrace_lm_learning_rate')
        self._value_coeff = config.get('prooftrace_lm_value_coeff')

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
        self._inner_model_PH = PH(self._config).to(self._device)
        self._inner_model_VH = VH(self._config).to(self._device)

        Log.out(
            "Initializing prooftrace LanguageModel", {
                'parameter_count_E': self._inner_model_E.parameters_count(),
                'parameter_count_H': self._inner_model_H.parameters_count(),
                'parameter_count_PH': self._inner_model_PH.parameters_count(),
                'parameter_count_VH': self._inner_model_VH.parameters_count(),
            },
        )

        self._model_E = self._inner_model_E
        self._model_H = self._inner_model_H
        self._model_PH = self._inner_model_PH
        self._model_VH = self._inner_model_VH

        self._nll_loss = nn.NLLLoss()
        self._mse_loss = nn.MSELoss()

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
            self._model_PH = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_PH,
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
                {'params': self._model_PH.parameters()},
                {'params': self._model_VH.parameters()},
            ],
            lr=self._learning_rate,
        )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                )

        batch_size = \
            self._config.get('prooftrace_lm_batch_size') // \
            self._accumulation_step_count

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self._train_sampler is None),
            sampler=self._train_sampler,
            collate_fn=lm_collate,
            num_workers=8,
        )

        Log.out('Training initialization', {
            "accumulation_step_count": self._accumulation_step_count,
            "world_size": self._config.get('distributed_world_size'),
            "batch_size": self._config.get('prooftrace_lm_batch_size'),
            "dataloader_batch_size": batch_size,
            "effective_batch_size": (
                self._config.get('prooftrace_lm_batch_size') *
                self._config.get('distributed_world_size')
            ),
        })

    def init_testing(
            self,
            test_dataset,
    ):
        batch_size = \
            self._config.get('prooftrace_lm_batch_size') // \
            self._accumulation_step_count

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lm_collate,
            num_workers=8,
        )

    def load(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model_E_{}.pt".format(rank)
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
                self._inner_model_PH.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_PH_{}.pt".format(rank),
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
                    if os.path.isfile(
                            self._load_dir + "/optimizer_{}.pt".format(rank)
                    ):
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
                self._inner_model_PH.state_dict(),
                self._save_dir + "/model_PH_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_VH.state_dict(),
                self._save_dir + "/model_VH_{}.pt".format(rank),
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer_{}.pt".format(rank),
            )

    def update(
            self,
    ) -> None:
        update = self._config.update()
        if update:
            if 'prooftrace_lm_learning_rate' in update:
                lr = \
                    self._config.get('prooftrace_lm_learning_rate')
                if lr != self._learning_rate:
                    self._learning_rate = lr
                    for group in self._optimizer.param_groups:
                        group['lr'] = lr
                    Log.out("Updated", {
                        "prooftrace_learning_rate": lr,
                    })
            if 'prooftrace_lm_value_coeff' in update:
                coeff = self._config.get('prooftrace_lm_value_coeff')
                if coeff != self._value_coeff:
                    self._value_coeff = coeff
                    Log.out("Updated", {
                        "prooftrace_lm_value_coeff": coeff,
                    })

            if self._tb_writer is not None:
                for k in update:
                    if k in [
                            'prooftrace_lm_learning_rate',
                            'prooftrace_lm_value_coeff',
                    ]:
                        self._tb_writer.add_scalar(
                            "prooftrace_lm_train_run/{}".format(k),
                            update[k], self._train_batch,
                        )

    def batch_train(
            self,
            epoch,
    ):
        assert self._train_loader is not None

        self._model_E.train()
        self._model_H.train()
        self._model_PH.train()
        self._model_VH.train()

        act_loss_meter = Meter()
        lft_loss_meter = Meter()
        rgt_loss_meter = Meter()
        # val_loss_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(epoch)

        for it, (idx, act, arg, trh, val) in enumerate(self._train_loader):
            action_embeds = self._model_E(act)
            argument_embeds = self._model_E(arg)

            # action_embeds = \
            #     torch.zeros(action_embeds.size()).to(self._device)
            # argument_embeds = \
            #     torch.zeros(argument_embeds.size()).to(self._device)

            hiddens = self._model_H(action_embeds, argument_embeds)

            heads = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            actions = torch.tensor([
                trh[i].value - len(PREPARE_TOKENS) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            lefts = torch.tensor([
                arg[i].index(trh[i].left) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            rights = torch.tensor([
                arg[i].index(trh[i].right) for i in range(len(trh))
            ], dtype=torch.int64).to(self._device)
            # values = torch.tensor(val).unsqueeze(1).to(self._device)

            prd_actions, prd_lefts, prd_rights = \
                self._model_PH(heads, hiddens, targets)
            # prd_values = self._model_VH(heads, targets)

            act_loss = self._nll_loss(prd_actions, actions)
            lft_loss = self._nll_loss(prd_lefts, lefts)
            rgt_loss = self._nll_loss(prd_rights, rights)
            # val_loss = self._mse_loss(prd_values, values)

            # (act_loss + lft_loss + rgt_loss +
            #  self._value_coeff * val_loss).backward()
            (act_loss + lft_loss + rgt_loss).backward()

            if it % self._accumulation_step_count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()

            act_loss_meter.update(act_loss.item())
            lft_loss_meter.update(lft_loss.item())
            rgt_loss_meter.update(rgt_loss.item())
            # val_loss_meter.update(val_loss.item())

            Log.out("TRAIN BATCH", {
                'train_batch': self._train_batch,
                'act_loss_avg': "{:.4f}".format(act_loss.item()),
                'lft_loss_avg': "{:.4f}".format(lft_loss.item()),
                'rgt_loss_avg': "{:.4f}".format(rgt_loss.item()),
                # 'val_loss_avg': "{:.4f}".format(val_loss.item()),
            })

            if self._train_batch % 10 == 0 and self._train_batch != 0:
                Log.out("PROOFTRACE TRAIN", {
                    'epoch': epoch,
                    'train_batch': self._train_batch,
                    'act_loss_avg': "{:.4f}".format(act_loss_meter.avg),
                    'lft_loss_avg': "{:.4f}".format(lft_loss_meter.avg),
                    'rgt_loss_avg': "{:.4f}".format(rgt_loss_meter.avg),
                    # 'val_loss_avg': "{:.4f}".format(val_loss_meter.avg),
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "prooftrace_lm_train/act_loss",
                        act_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "prooftrace_lm_train/lft_loss",
                        lft_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "prooftrace_lm_train/rgt_loss",
                        rgt_loss_meter.avg, self._train_batch,
                    )
                    # self._tb_writer.add_scalar(
                    #     "prooftrace_lm_train/val_loss",
                    #     val_loss_meter.avg, self._train_batch,
                    # )

                act_loss_meter = Meter()
                lft_loss_meter = Meter()
                rgt_loss_meter = Meter()
                # val_loss_meter = Meter()

            if self._train_batch % 100 == 0:
                self.save()

                self.test()
                self._model_E.train()
                self._model_H.train()
                self._model_PH.train()
                self._model_VH.train()

                self.update()

            self._train_batch += 1

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })

    def test(
            self,
    ):
        assert self._test_loader is not None

        self._model_E.eval()
        self._model_H.eval()
        self._model_PH.eval()
        self._model_VH.eval()

        act_loss_meter = Meter()
        lft_loss_meter = Meter()
        rgt_loss_meter = Meter()
        # val_loss_meter = Meter()

        test_batch = 0

        with torch.no_grad():
            for it, (idx, act, arg, trh, val) in enumerate(self._test_loader):
                action_embeds = self._model_E(act)
                argument_embeds = self._model_E(arg)

                hiddens = self._model_H(action_embeds, argument_embeds)

                heads = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
                ], dim=0)

                actions = torch.tensor([
                    trh[i].value - len(PREPARE_TOKENS) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                lefts = torch.tensor([
                    arg[i].index(trh[i].left) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                rights = torch.tensor([
                    arg[i].index(trh[i].right) for i in range(len(trh))
                ], dtype=torch.int64).to(self._device)
                # values = torch.tensor(val).unsqueeze(1).to(self._device)

                prd_actions, prd_lefts, prd_rights = \
                    self._model_PH(heads, hiddens, targets)
                # prd_values = self._model_VH(heads, targets)

                act_loss = self._nll_loss(prd_actions, actions)
                lft_loss = self._nll_loss(prd_lefts, lefts)
                rgt_loss = self._nll_loss(prd_rights, rights)
                # val_loss = self._mse_loss(prd_values, values)

                act_loss_meter.update(act_loss.item())
                lft_loss_meter.update(lft_loss.item())
                rgt_loss_meter.update(rgt_loss.item())
                # val_loss_meter.update(val_loss.item())

                Log.out("TEST BATCH", {
                    'train_batch': self._train_batch,
                    'test_batch': test_batch,
                    'act_loss_avg': "{:.4f}".format(act_loss.item()),
                    'lft_loss_avg': "{:.4f}".format(lft_loss.item()),
                    'rgt_loss_avg': "{:.4f}".format(rgt_loss.item()),
                    # 'val_loss_avg': "{:.4f}".format(val_loss.item()),
                })

                test_batch += 1

            Log.out("PROOFTRACE TEST", {
                'train_batch': self._train_batch,
                'act_loss_avg': "{:.4f}".format(act_loss_meter.avg),
                'lft_loss_avg': "{:.4f}".format(lft_loss_meter.avg),
                'rgt_loss_avg': "{:.4f}".format(rgt_loss_meter.avg),
                # 'val_loss_avg': "{:.4f}".format(val_loss_meter.avg),
            })

            if self._tb_writer is not None:
                self._tb_writer.add_scalar(
                    "prooftrace_lm_test/act_loss",
                    act_loss_meter.avg, self._train_batch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_lm_test/lft_loss",
                    lft_loss_meter.avg, self._train_batch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_lm_test/rgt_loss",
                    rgt_loss_meter.avg, self._train_batch,
                )
                # self._tb_writer.add_scalar(
                #     "prooftrace_lm_test/val_loss",
                #     val_loss_meter.avg, self._train_batch,
                # )


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

    torch.manual_seed(0)

    train_dataset = ProofTraceLMDataset(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        False,
        config.get('prooftrace_sequence_length'),
    )
    test_dataset = ProofTraceLMDataset(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        True,
        config.get('prooftrace_sequence_length'),
    )

    lm = LanguageModel(config)

    lm.init_training(train_dataset)
    lm.init_testing(test_dataset)
    lm.load(True)

    epoch = 0
    while True:
        lm.update()
        lm.batch_train(epoch)
        lm.save()
        epoch += 1
