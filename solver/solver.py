import argparse
import os
import torch
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F

from dataset.dataset import SATDataset

from tensorboardX import SummaryWriter

from solver.models.transformer import S

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class Solver:
    def __init__(
            self,
            config: Config,
            train_dataset: SATDataset,
            test_dataset: SATDataset,
    ):
        self._config = config
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('solver_save_dir')
        self._load_dir = config.get('solver_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_sat_policy = S(
            self._config,
            self._train_dataset.variable_count(),
            self._train_dataset.clause_count(),
        ).to(self._device)

        Log.out(
            "Initializing solver", {
                'parameter_count': self._inner_sat_policy.parameters_count()
            },
        )

        if self._config.get('distributed_training'):
            self._sat_policy = torch.nn.parallel.DistributedDataParallel(
                self._inner_sat_policy,
                device_ids=[self._device],
            )
        else:
            self._sat_policy = self._inner_sat_policy

        self._sat_optimizer = optim.Adam(
            self._sat_policy.parameters(),
            lr=self._config.get('solver_learning_rate'),
            betas=(
                self._config.get('solver_adam_beta_1'),
                self._config.get('solver_adam_beta_2'),
            ),
        )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    self._train_dataset,
                )

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._config.get('solver_batch_size'),
            shuffle=(self._train_sampler is None),
            pin_memory=True,
            num_workers=8,
            sampler=self._train_sampler,
        )
        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._config.get('solver_batch_size'),
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        self._sat_batch_count = 0

    def load_sat(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            if os.path.isfile(self._load_dir + "/sat_policy.pt"):
                self._inner_sat_policy.load_state_dict(
                    torch.load(
                        self._load_dir + "/sat_policy_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                if training:
                    self._sat_optimizer.load_state_dict(
                        torch.load(
                            self._load_dir +
                            "/sat_optimizer_{}.pt".format(rank),
                            map_location=self._device,
                        ),
                    )

    def save_sat(
            self,
    ):
        rank = self._config.get('distributed_rank')

        if self._save_dir:
            Log.out(
                "Saving sat models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._inner_sat_policy.state_dict(),
                self._save_dir + "/sat_policy_{}.pt".format(rank),
            )
            torch.save(
                self._sat_optimizer.state_dict(),
                self._save_dir + "/sat_optimizer_{}.pt".format(rank),
            )

    def batch_train_sat(self):
        self._sat_policy.train()
        loss_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(self._sat_batch_count)

        for it, (cl_pos, cl_neg, sats) in enumerate(self._train_loader):
            generated = self._sat_policy(
                cl_pos.to(self._device),
                cl_neg.to(self._device),
            )
            loss = F.mse_loss(generated, sats.to(self._device))

            self._sat_optimizer.zero_grad()
            (10 * loss).backward()
            self._sat_optimizer.step()

            loss_meter.update(loss.item())

            self._sat_batch_count += 1

            if self._sat_batch_count % 10 == 0:

                hit = 0
                total = 0

                for i in range(generated.size(0)):
                    if generated[i].item() >= 0.5 and sats[i].item() >= 0.5:
                        hit += 1
                    if generated[i].item() < 0.5 and sats[i].item() < 0.5:
                        hit += 1
                    total += 1

                Log.out("SAT TRAIN", {
                    'batch_count': self._sat_batch_count,
                    'loss_avg': loss_meter.avg,
                    'hit_rate': "{:.2f}".format(hit / total),
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/sat/loss",
                        loss_meter.avg, self._sat_batch_count,
                    )
                    self._tb_writer.add_scalar(
                        "train/sat/hit_rate",
                        hit / total, self._sat_batch_count,
                    )

                loss_meter = Meter()

            if self._sat_batch_count % 320 == 0:
                self._sat_policy.eval()
                self.batch_test_sat()
                self._sat_policy.train()
                self.save_sat()

    def batch_test_sat(
            self,
    ):
        self._sat_policy.eval()
        loss_meter = Meter()

        hit = 0
        total = 0

        with torch.no_grad():
            for it, (cl_pos, cl_neg, sats) in enumerate(self._test_loader):
                generated = self._sat_policy(
                    cl_pos.to(self._device),
                    cl_neg.to(self._device),
                )
                loss = F.mse_loss(generated, sats.to(self._device))

                loss_meter.update(loss.item())

                for i in range(generated.size(0)):
                    if generated[i].item() >= 0.5 and sats[i].item() >= 0.5:
                        hit += 1
                    if generated[i].item() < 0.5 and sats[i].item() < 0.5:
                        hit += 1
                    total += 1

        Log.out("SAT TEST", {
            'batch_count': self._sat_batch_count,
            'loss_avg': loss_meter.avg,
            'hit_rate': "{:.2f}".format(hit / total),
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/sat/loss",
                loss_meter.avg, self._sat_batch_count,
            )
            self._tb_writer.add_scalar(
                "test/sat/hit_rate",
                hit / total, self._sat_batch_count,
            )

        return loss_meter.avg


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--train_dataset_dir',
        type=str, help="train dataset directory",
    )
    parser.add_argument(
        '--test_dataset_dir',
        type=str, help="test dataset directory",
    )
    parser.add_argument(
        '--solver_save_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--solver_load_dir',
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

    if args.train_dataset_dir is not None:
        config.override(
            'solver_train_dataset_dir',
            os.path.expanduser(args.train_dataset_dir),
        )
    if args.test_dataset_dir is not None:
        config.override(
            'solver_test_dataset_dir',
            os.path.expanduser(args.test_dataset_dir),
        )
    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.solver_load_dir is not None:
        config.override(
            'solver_load_dir',
            os.path.expanduser(args.solver_load_dir),
        )
    if args.solver_save_dir is not None:
        config.override(
            'solver_save_dir',
            os.path.expanduser(args.solver_save_dir),
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

    train_dataset = SATDataset(
        config,
        os.path.expanduser(config.get('solver_train_dataset_dir')),
    )
    test_dataset = SATDataset(
        config,
        os.path.expanduser(config.get('solver_test_dataset_dir')),
    )

    solver = Solver(config, train_dataset, test_dataset)
    solver.load_sat(True)

    while True:
        solver.batch_train_sat()


def test():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'test_dataset_dir',
        type=str, help="test dataset directory",
    )
    parser.add_argument(
        '--solver_load_dir',
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

    if args.distributed_training is not None:
        config.override('distributed_training', args.distributed_training)
    if args.distributed_rank is not None:
        config.override('distributed_rank', args.distributed_rank)
    if args.distributed_world_size is not None:
        config.override('distributed_world_size', args.distributed_world_size)

    if args.solver_load_dir is not None:
        config.override(
            'solver_load_dir',
            os.path.expanduser(args.solver_load_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    test_dataset = SATDataset(
        config,
        os.path.expanduser(args.test_dataset_dir),
    )

    solver = Solver(config, test_dataset, test_dataset)
    solver.load_sat(False)

    solver.batch_test_sat()
