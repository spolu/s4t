import argparse
import os
import torch
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F

from dataset.dataset import SATDataset

from generic.lr_scheduler import LRScheduler

from tensorboardX import SummaryWriter

from sat.solver.models.transformer import S

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class Solver:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('sat_solver_save_dir')
        self._load_dir = config.get('sat_solver_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model = S(
            self._config,
            self._train_dataset.variable_count(),
            self._train_dataset.clause_count(),
        ).to(self._device)

        Log.out(
            "Initializing solver", {
                'parameter_count': self._inner_model.parameters_count()
            },
        )

        self._model = self._inner_model

    def init_training(
            self,
            train_dataset,
    ):
        if self._config.get('distributed_training'):
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._inner_model,
                device_ids=[self._device],
            )
        else:
            self._model = self._inner_model

        self._sat_optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._config.get('sat_solver_learning_rate'),
        )
        self._scheduler = LRScheduler(
            self._optimizer,
            40,
            10,
            self._config.get('sat_solver_learning_rate_annealing'),
        )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                )

        pin_memory = False
        if self._config.get('device') != 'cpu':
            pin_memory = True

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._config.get('sat_solver_batch_size'),
            shuffle=(self._train_sampler is None),
            pin_memory=pin_memory,
            num_workers=8,
            sampler=self._train_sampler,
        )

        self._train_batch = 0

    def init_testing(
            self,
            test_dataset,
    ):
        pin_memory = False
        if self._config.get('device') != 'cpu':
            pin_memory = True

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._config.get('sat_solver_batch_size'),
            shuffle=False,
            num_workers=8,
            pin_memory=pin_memory,
        )

    def load(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model_{}.pt".format(rank)
            ):
                Log.out(
                    "Loading sat models", {
                        'save_dir': self._load_dir,
                    })
                self._inner_model.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                if training:
                    self._sat_optimizer.load_state_dict(
                        torch.load(
                            self._load_dir +
                            "/optimizer_{}.pt".format(rank),
                            map_location=self._device,
                        ),
                    )
                    self._scheduler.load_state_dict(
                        torch.load(
                            self._load_dir +
                            "/scheduler_{}.pt".format(rank),
                            map_location=self._device,
                        ),
                    )

    def save(
            self,
    ):
        rank = self._config.get('distributed_rank')

        if self._save_dir:
            Log.out(
                "Saving sat models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._inner_model.state_dict(),
                self._save_dir + "/model_{}.pt".format(rank),
            )
            torch.save(
                self._sat_optimizer.state_dict(),
                self._save_dir + "/sat_optimizer_{}.pt".format(rank),
            )
            torch.save(
                self._scheduler.state_dict(),
                self._save_dir + "/scheduler_{}.pt".format(rank),
            )

    def batch_train(
            self,
            epoch,
    ):
        assert self._train_loader is not None

        self._model.train()
        loss_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(epoch)
        self._scheduler.step()

        for it, (cl_pos, cl_neg, sats) in enumerate(self._train_loader):
            generated = self._model(
                cl_pos.to(self._device),
                cl_neg.to(self._device),
            )

            loss = F.binary_cross_entropy(generated, sats.to(self._device))

            self._sat_optimizer.zero_grad()
            loss.backward()
            self._sat_optimizer.step()

            loss_meter.update(loss.item())

            self._train_batch += 1

            if self._train_batch % 10 == 0:

                hit = 0
                total = 0

                for i in range(generated.size(0)):
                    if generated[i].item() >= 0.5 and sats[i].item() >= 0.5:
                        hit += 1
                    if generated[i].item() < 0.5 and sats[i].item() < 0.5:
                        hit += 1
                    total += 1

                Log.out("SAT TRAIN", {
                    'train_batch': self._train_batch,
                    'loss_avg': loss_meter.avg,
                    'hit_rate': "{:.2f}".format(hit / total),
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/sat/loss",
                        loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/sat/hit_rate",
                        hit / total, self._train_batch,
                    )

                loss_meter = Meter()

        Log.out("EPOCH DONE", {
            'epoch': epoch,
            'learning_rate': self._scheduler.get_lr(),
        })

    def batch_test(
            self,
    ):
        assert self._test_loader is not None

        self._model.eval()
        loss_meter = Meter()

        hit = 0
        total = 0

        with torch.no_grad():
            for it, (cl_pos, cl_neg, sats) in enumerate(self._test_loader):
                generated = self._model(
                    cl_pos.to(self._device),
                    cl_neg.to(self._device),
                )

                loss = F.binary_cross_entropy(generated, sats.to(self._device))

                loss_meter.update(loss.item())

                for i in range(generated.size(0)):
                    if generated[i].item() >= 0.5 and sats[i].item() >= 0.5:
                        hit += 1
                    if generated[i].item() < 0.5 and sats[i].item() < 0.5:
                        hit += 1
                    total += 1

        Log.out("SAT TEST", {
            'batch_count': self._batch_count,
            'loss_avg': loss_meter.avg,
            'hit_rate': "{:.2f}".format(hit / total),
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/sat/loss",
                loss_meter.avg, self._batch_count,
            )
            self._tb_writer.add_scalar(
                "test/sat/hit_rate",
                hit / total, self._batch_count,
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

    if args.train_dataset_dir is not None:
        config.override(
            'sat_solver_train_dataset_dir',
            os.path.expanduser(args.train_dataset_dir),
        )
    if args.test_dataset_dir is not None:
        config.override(
            'sat_solver_test_dataset_dir',
            os.path.expanduser(args.test_dataset_dir),
        )
    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.load_dir is not None:
        config.override(
            'sat_solver_load_dir',
            os.path.expanduser(args.load_dir),
        )
    if args.save_dir is not None:
        config.override(
            'sat_solver_save_dir',
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

    train_dataset = SATDataset(
        config,
        os.path.expanduser(config.get('sat_solver_train_dataset_dir')),
    )
    test_dataset = SATDataset(
        config,
        os.path.expanduser(config.get('sat_solver_test_dataset_dir')),
    )

    solver = Solver(config)

    solver.init_training(train_dataset)
    solver.init_testing(test_dataset)

    solver.load(True)

    epoch = 0
    while True:
        solver.batch_train()
        solver.batch_test()
        solver.save()
        epoch += 1


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
        '--load_dir',
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

    if args.load_dir is not None:
        config.override(
            'solver_load_dir',
            os.path.expanduser(args.load_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    test_dataset = SATDataset(
        config,
        os.path.expanduser(args.test_dataset_dir),
    )

    solver = Solver(config)

    solver.init_testing(test_dataset)

    solver.load(True)

    solver.batch_test()
