import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataset.dataset import SATDataset

from tensorboardX import SummaryWriter

from solver.models.transformer import SATTransformer

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


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
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._sat_policy = SATTransformer(
            self._config,
            self._train_dataset.variable_count(),
            self._train_dataset.clause_count(),
        ).to(self._device)

        self._sat_optimizer = optim.Adam(
            self._sat_policy.parameters(),
            lr=self._config.get('solver_learning_rate'),
            betas=(
                self._config.get('solver_adam_beta_1'),
                self._config.get('solver_adam_beta_2'),
            ),
        )

        Log.out(
            "Initializing solver", {
                'parameter_count': self._sat_policy.parameters_count()
            },
        )

        if self._load_dir:
            if os.path.isfile(self._load_dir + "/sat_policy.pt"):
                self._sat_policy.load_state_dict(
                    torch.load(
                        self._load_dir + "/sat_policy.pt",
                        map_location=self._device,
                    ),
                )
                self._sat_optimizer.load_state_dict(
                    torch.load(
                        self._load_dir + "/sat_optimizer.pt",
                        map_location=self._device,
                    ),
                )

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._config.get('solver_batch_size'),
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )
        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._config.get('solver_batch_size'),
            shuffle=False,
            pin_memory=True,
        )

        self._sat_batch_count = 0

    def save_sat(
            self,
    ):
        if self._save_dir:
            Log.out(
                "Saving sat models", {
                    'save_dir': self._save_dir,
                })
            torch.save(
                self._sat_policy.state_dict(),
                self._save_dir + "/sat_policy.pt",
            )
            torch.save(
                self._sat_optimizer.state_dict(),
                self._save_dir + "/sat_optimizer.pt",
            )

    def batch_train_sat(self):
        self._sat_policy.train()
        loss_meter = Meter()

        for it, (clauses, sats) in enumerate(self._train_loader):
            generated = self._sat_policy(clauses.to(self._device))
            loss = F.mse_loss(generated, sats.to(self._device))

            # if it == 0:
            #     print("{}".format(generated[0:2]))

            self._sat_optimizer.zero_grad()
            (10 * loss).backward()
            self._sat_optimizer.step()

            loss_meter.update(loss.item())

            self._sat_batch_count += 1

            if self._sat_batch_count % 10 == 0:
                Log.out("SAT TRAIN", {
                    'batch_count': self._sat_batch_count,
                    'loss_avg': loss_meter.avg,
                    # 'loss_min': loss_meter.min,
                    # 'loss_max': loss_meter.max,
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/sat/loss",
                        loss_meter.avg, self._sat_batch_count,
                    )

                loss_meter = Meter()

    def batch_test_sat(
            self,
    ):
        self._sat_policy.eval()
        loss_meter = Meter()

        hit = 0
        total = 0

        with torch.no_grad():
            for it, (clauses, sats) in enumerate(self._test_loader):
                generated = self._sat_policy(clauses.to(self._device))
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
            # 'loss_min': loss_meter.min,
            # 'loss_max': loss_meter.max,
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
        'train_dataset_dir',
        type=str, help="train dataset directory",
    )
    parser.add_argument(
        'test_dataset_dir',
        type=str, help="test dataset directory",
    )
    parser.add_argument(
        '--device',
        type=str, help="config override",
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
    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)
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

    train_dataset = SATDataset(
        config,
        os.path.expanduser(args.train_dataset_dir),
    )
    test_dataset = SATDataset(
        config,
        os.path.expanduser(args.test_dataset_dir),
    )

    solver = Solver(config, train_dataset, test_dataset)

    i = 0
    while True:
        solver.batch_train_sat()
        solver.batch_test_sat()
        solver.save_sat()
        i += 1
