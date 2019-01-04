import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataset.dataset import SATDataset

from tensorboardX import SummaryWriter

from transformer.model import SAT

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


class Transformer:
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

        self._save_dir = config.get('transformer_save_dir')
        self._load_dir = config.get('transformer_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._sat_policy = SAT(
            self._config,
            self._train_dataset.variable_count(),
            self._train_dataset.clause_count(),
        ).to(self._device)

        self._sat_optimizer = optim.Adam(
            self._sat_policy.parameters(),
            lr=self._config.get('transformer_learning_rate'),
            betas=(
                self._config.get('transformer_adam_beta_1'),
                self._config.get('transformer_adam_beta_2'),
            ),
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
            batch_size=self._config.get('transformer_batch_size'),
            shuffle=True,
            num_workers=0,
        )
        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._config.get('transformer_batch_size'),
            shuffle=False,
            num_workers=0,
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
            generated = self._sat_policy(clauses)

            loss = F.mse_loss(generated, sats)

            if it == 0:
                print("{}".format(generated[0:2]))

            self._sat_optimizer.zero_grad()
            loss.backward()
            self._sat_optimizer.step()

            loss_meter.update(loss.item())

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

        self._sat_batch_count += 1

    def batch_test_sat(
            self,
    ):
        self._sat_policy.eval()
        loss_meter = Meter()

        with torch.no_grad():
            for it, (clauses, sats) in enumerate(self._test_loader):
                generated = self._sat_policy(clauses)

                loss = F.mse_loss(generated, sats)

                loss_meter.update(loss.item())

        Log.out("SAT TEST", {
            'batch_count': self._sat_batch_count,
            'loss_avg': loss_meter.avg,
            # 'loss_min': loss_meter.min,
            # 'loss_max': loss_meter.max,
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/sat/loss",
                loss_meter.avg, self._sat_batch_count,
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
        '--transformer_save_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--transformer_load_dir',
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
    if args.transformer_load_dir is not None:
        config.override(
            'transformer_load_dir',
            os.path.expanduser(args.transformer_load_dir),
        )
    if args.transformer_save_dir is not None:
        config.override(
            'transformer_save_dir',
            os.path.expanduser(args.transformer_save_dir),
        )

    train_dataset = SATDataset(
        config,
        os.path.expanduser(args.train_dataset_dir),
    )
    test_dataset = SATDataset(
        config,
        os.path.expanduser(args.test_dataset_dir),
    )

    transformer = Transformer(config, train_dataset, test_dataset)
    Log.out(
        "Initializing transformer", {},
    )

    i = 0
    while True:
        if i % 10 == 0:
            transformer.batch_test_sat()
            transformer.save_sat()
        transformer.batch_train_sat()
        i += 1
